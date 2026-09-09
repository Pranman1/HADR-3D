"""Drawing. One sink, used by BOTH the training and the running environment.

Written early on purpose: a fire model you cannot watch is a fire model you cannot
debug.  Everything built after this is watchable from the moment it exists.

Rerun splits in two.  The logging half runs inside the simulation, on whatever box
has the GPU.  The viewer half is just a window and needs no GPU at all, so it can
live on a laptop over SSH.  See `serve()` below.

Never enabled during training.  Four thousand simulations logging to a viewer would
cost more than the training does.  This is for one episode at a time.
"""

from __future__ import annotations

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

# muted city, hot fire, cool drones -- the same split the project's website uses
C_SOLID = [110, 110, 118]
C_CANOPY = [190, 160, 90]
C_GROUND = [38, 38, 42]
C_FUEL = [60, 72, 58]


GRPC_PORT = 9876
WEB_PORT = 9090


def _lan_ip() -> str:
    """This machine's address on the network, for viewing from another computer."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))   # no packet is sent; just picks the route
        return s.getsockname()[0]
    except OSError:
        return "localhost"
    finally:
        s.close()


def viewer_url(host: str = "localhost") -> str:
    """The page to open.

    Rerun uses two ports: one streams the data, one serves the viewer page.  The
    page only knows where to find the data if the stream address is in its query
    string -- opening the bare port shows an empty viewer and no error.

    The stream address must be *fully* percent-encoded, ":" and "/" included.
    Leaving those raw gets you "failed to parse url, url: \'\'", which looks
    identical to omitting the parameter altogether.
    """
    from urllib.parse import quote

    stream = f"rerun+http://{host}:{GRPC_PORT}/proxy"
    return f"http://{host}:{WEB_PORT}?url={quote(stream, safe='')}"


TIMELINE = "sim_time"


def start(name: str = "hadrsim", web: bool = False, save: str | None = None,
          play: bool = True, speed: float = 2.0) -> None:
    """Open a viewer. web=True serves to a browser -- use that over SSH.

    `play` makes the recording start playing on its own timeline and loop.  Without
    it the viewer opens paused at whatever moment it happens to land on, which for a
    finished episode means a still picture of a city and no obvious sign that there
    is anything to scrub -- the natural conclusion being that nothing is happening.
    """
    rr.init(name, spawn=not (web or save))
    if web:
        # The viewer page and the data stream sit on different ports, so the browser
        # treats the stream as cross-origin and blocks the fetch unless we say
        # otherwise.  Without this: "data source failed / TypeError: load failed".
        uri = rr.serve_grpc(grpc_port=GRPC_PORT, cors_allow_origin=["*"])
        rr.serve_web_viewer(web_port=WEB_PORT, open_browser=False, connect_to=uri)
        print(f"\n  on this machine   {viewer_url()}")
        print(f"  from another      {viewer_url(_lan_ip())}\n")
    elif save:
        rr.save(save)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Spatial3DView(origin="/world", name="hadrsim"),
        rrb.TimePanel(
            state="expanded",
            timeline=TIMELINE,
            play_state="playing" if play else "paused",
            loop_mode="all" if play else "off",
            playback_speed=speed,
        ),
    ))


def log_terrain(terrain, path: str = "/world/terrain") -> None:
    """Draw the static city once.

    Two ways in, picked by what the terrain carries.  A map loaded from real building
    outlines is drawn as real extruded buildings; the paper's grid map, which has no
    outlines, falls back to one box per cell.  The simulation sees the same grid
    either way -- this only changes what you look at.
    """
    rr.set_time(TIMELINE, duration=0.0)
    cs = terrain.cell_size_m

    if terrain.footprints:
        _log_footprints(path, terrain.footprints)
    else:
        solid = terrain.solid_top.cpu().numpy()
        z0 = terrain.canopy_z0.cpu().numpy()
        z1 = terrain.canopy_z1.cpu().numpy()
        _log_boxes(f"{path}/solid", solid > 0, np.zeros_like(solid), solid, cs, C_SOLID)
        _log_boxes(f"{path}/canopy", z1 > 0, z0, z1, cs, C_CANOPY)

    _log_ground(f"{path}/ground", terrain)


def _log_footprints(path: str, footprints) -> None:
    """Real buildings, merged into one mesh per kind so the viewer stays responsive.

    Colour comes from each building's own OSM tags, and the window pattern from one
    atlas shared by the whole city -- so a few hundred buildings are still two mesh
    objects and one image.
    """
    from hadrsim.viz import facade, mesh

    for kind in ("solid", "canopy"):
        pieces, colours = [], []
        for i, f in enumerate(x for x in footprints if x.kind == kind):
            v, t, uv = mesh.extrude(
                f.xy.cpu().numpy().astype(np.float64), f.base_m, f.top_m, uv=True
            )
            if len(v) == 0:
                continue
            pieces.append((v, t, uv))
            # Seeded by the outline, so a building keeps its colour between runs.
            seed = hash((round(float(f.xy[0, 0]), 2), round(float(f.xy[0, 1]), 2), i))
            base = np.array(facade.colour_for(f.tags or {}, seed=seed), dtype=float)
            # Darken toward the ground: cheap ambient occlusion, and it makes a
            # skyline read as a skyline instead of a flat slab of grey.
            lift = 0.72 + 0.28 * np.clip(v[:, 2] / 40.0, 0, 1)
            colours.append((base * lift[:, None]).clip(0, 255).astype(np.uint8))

        if not pieces:
            continue
        verts, faces, uvs = mesh.merge(pieces)
        rr.log(
            f"{path}/{kind}",
            rr.Mesh3D(
                vertex_positions=verts,
                triangle_indices=faces,
                vertex_texcoords=uvs,
                albedo_texture=facade.window_atlas(),
                vertex_colors=np.concatenate(colours),
            ),
        )


def _log_ground(path: str, terrain, overlay: np.ndarray | None = None) -> None:
    """The ground, as one textured quad lying in the world.

    A quad rather than `rr.Image`, because an image logged into a 3D view floats as a
    flat picture instead of being the floor.  It also means the fire can be painted
    into the same texture: one image at cell resolution gives crisp, pixel-sharp burn
    scars however large the map, where a sphere per burning cell gives neither.
    """
    img = overlay if overlay is not None else ground_image(terrain)
    w_m, h_m = terrain.extent_m

    rr.log(path, rr.Mesh3D(
        vertex_positions=np.array([[0, 0, 0], [w_m, 0, 0], [w_m, h_m, 0], [0, h_m, 0]],
                                  dtype=np.float32),
        triangle_indices=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        vertex_texcoords=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        albedo_texture=img,
    ))


def ground_image(terrain, fire=None, env: int = 0) -> np.ndarray:
    """The floor texture: what will burn, what is burning, and what already did.

    Indexed [y, x] because that is how an image is laid out, while the simulation's
    fields are [x, y] -- hence the transposes.
    """
    fuel = terrain.fuel.cpu().numpy().T
    hi = max(float(fuel.max()), 1e-6)
    img = (np.array(C_GROUND) + (fuel / hi)[..., None]
           * (np.array(C_FUEL) - np.array(C_GROUND)))

    if fire is not None:
        burnt = fire.burnt[env].cpu().numpy().T
        img[burnt] = C_BURNT_GROUND

        heat = fire.intensity[env].cpu().numpy().T
        alight = heat > 0
        if alight.any():
            # Hot cores go yellow-white, dying edges deep red, so the eye finds the
            # front of the fire without being told where it is.
            t = heat[alight][:, None]
            img[alight] = C_FIRE_COOL + t * (C_FIRE_HOT - C_FIRE_COOL)

    return img.clip(0, 255).astype(np.uint8)


def _log_boxes(path, mask, bottom, top, cell, color) -> None:
    """One box per marked cell. Cheap enough for a 25x25 map, logged once."""
    xs, ys = np.nonzero(mask)
    if len(xs) == 0:
        rr.log(path, rr.Clear(recursive=True))
        return
    half_z = (top[xs, ys] - bottom[xs, ys]) / 2.0
    centres = np.stack(
        [(xs + 0.5) * cell, (ys + 0.5) * cell, bottom[xs, ys] + half_z], axis=-1
    )
    halves = np.stack(
        [np.full_like(half_z, cell / 2), np.full_like(half_z, cell / 2), half_z], axis=-1
    )
    rr.log(path, rr.Boxes3D(centers=centres, half_sizes=halves, colors=[color]))


# ------------------------------------------------------------------ the world

C_FIRE_HOT = np.array([255, 214, 120])
C_FIRE_COOL = np.array([200, 60, 24])
C_BURNT_GROUND = np.array([34, 28, 28])
C_CALM = [225, 228, 235]
C_PANIC = [235, 92, 58]
C_SMOKE = [176, 176, 180]


def log_time(t_s: float) -> None:
    rr.set_time(TIMELINE, duration=float(t_s))


def log_fire(fire, path: str = "/world/fire", env: int = 0, max_flames: int = 4000) -> None:
    """Fire, painted into the floor texture, with flames standing above the front.

    The burn scar is a texture rather than a sphere per cell: at 1 m cells a real city
    block has tens of thousands of burnt cells, and spheres for all of them are both
    slow and unreadable.  A texture is one image at any map size, and it comes out
    pixel-sharp, which is what fire on a grid should look like.

    Flames are the exception -- fire has height, and a flat scar alone reads as paint.
    Only the hottest cells get one, capped, so a city-wide fire stays drawable.
    """
    cell = fire.terrain.cell_size_m
    _log_ground("/world/terrain/ground", fire.terrain, ground_image(fire.terrain, fire, env))

    heat = fire.intensity[env].cpu().numpy()
    xs, ys = np.nonzero(heat > 0.02)
    if not len(xs):
        rr.log(f"{path}/flame", rr.Clear(recursive=True))
        return

    v = heat[xs, ys]
    if len(xs) > max_flames:                        # keep the fiercest
        keep = np.argpartition(-v, max_flames)[:max_flames]
        xs, ys, v = xs[keep], ys[keep], v[keep]

    # Flame height tracks intensity, so the active front stands taller than the embers.
    height = 0.4 + 2.2 * v
    centres = np.stack([(xs + 0.5) * cell, (ys + 0.5) * cell, height / 2], -1)
    halves = np.stack([np.full(len(xs), cell * 0.42), np.full(len(xs), cell * 0.42),
                       height / 2], -1)
    colour = (C_FIRE_COOL + v[:, None] * (C_FIRE_HOT - C_FIRE_COOL)).astype(np.uint8)
    rr.log(f"{path}/flame", rr.Boxes3D(centers=centres, half_sizes=halves, colors=colour))


def log_smoke(smoke, path: str = "/world/smoke", env: int = 0, floor: float = 0.02) -> None:
    """The plume, as one translucent point per occupied voxel.

    Thin smoke is dropped entirely rather than drawn faintly -- a few thousand visible
    points read as a plume, a hundred thousand nearly-invisible ones read as fog and
    cost a great deal to draw.
    """
    cell = smoke.terrain.cell_size_m
    d = smoke.density[env].cpu().numpy()
    alt = smoke.layer_altitudes.cpu().numpy()

    ks, xs, ys = np.nonzero(d > floor)
    if not len(ks):
        rr.log(path, rr.Clear(recursive=False))
        return
    v = np.clip(d[ks, xs, ys] / max(d.max(), 1e-6), 0, 1)
    # Many small faint puffs read as a plume; a few big opaque ones read as bubbles.
    # Higher, older smoke is thinner and wider, so radius grows and alpha falls.
    colour = np.concatenate([np.tile(C_SMOKE, (len(ks), 1)),
                             (25 + 110 * v)[:, None]], -1).astype(np.uint8)
    radii = cell * (0.35 + 0.10 * ks) * (0.5 + 0.5 * v)
    rr.log(path, rr.Points3D(
        positions=np.stack([(xs + 0.5) * cell, (ys + 0.5) * cell, alt[ks]], -1),
        colors=colour, radii=radii,
    ))


def log_human(human, path: str = "/world/human", env: int = 0) -> None:
    """The person, as a figure rather than a dot, facing the way they are walking.

    A sphere is unreadable at city scale: you cannot tell it from a marker, and you
    cannot tell which way it is going.  A 1.75 m figure with a head reads instantly
    as a person, gives the scene its sense of scale, and costs four boxes.

    Coloured by panic -- pale and calm through to hot and frantic -- so their state is
    visible in the scene itself rather than only on a plot.
    """
    p3 = human.pos[env].cpu().numpy()
    goal = human.goal[env].cpu().numpy()
    panic = float(human.gamma[env])
    vel = human.vel[env].cpu().numpy()

    body = (np.array(C_CALM) * (1 - panic) + np.array(C_PANIC) * panic)
    speed = float(np.linalg.norm(vel))
    heading = vel / speed if speed > 1e-3 else np.array([1.0, 0.0])

    # torso, head, and a stub in front so the direction of travel is obvious
    nose = p3 + heading * 0.35
    centres = np.array([
        [p3[0], p3[1], 0.70],
        [p3[0], p3[1], 1.58],
        [nose[0], nose[1], 1.00],
    ])
    halves = np.array([
        [0.22, 0.22, 0.55],
        [0.13, 0.13, 0.16],
        [0.08, 0.08, 0.08],
    ])
    colours = np.array([body, body * 0.92, body * 0.7]).clip(0, 255).astype(np.uint8)
    rr.log(f"{path}/person", rr.Boxes3D(centers=centres, half_sizes=halves, colors=colours))

    rr.log(f"{path}/goal", rr.Boxes3D(
        centers=[[goal[0], goal[1], 1.0]], half_sizes=[[0.9, 0.9, 1.0]],
        colors=[[90, 200, 130]]))
    rr.log(f"{path}/panic", rr.Scalars(panic))
    rr.log(f"{path}/speed", rr.Scalars(speed))
