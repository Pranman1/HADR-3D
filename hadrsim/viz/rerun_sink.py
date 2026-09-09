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


def start(name: str = "hadrsim", web: bool = False, save: str | None = None) -> None:
    """Open a viewer. web=True serves to a browser -- use that over SSH."""
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
    rr.send_blueprint(rrb.Spatial3DView(origin="/world", name="hadrsim"))


def log_terrain(terrain, path: str = "/world/terrain") -> None:
    """Draw the static city once.

    Two ways in, picked by what the terrain carries.  A map loaded from real building
    outlines is drawn as real extruded buildings; the paper's grid map, which has no
    outlines, falls back to one box per cell.  The simulation sees the same grid
    either way -- this only changes what you look at.
    """
    rr.set_time("sim_time", duration=0.0)
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


def _log_ground(path: str, terrain) -> None:
    """The ground, seen from above: dark where nothing burns, green where much does."""
    fuel = terrain.fuel.cpu().numpy().T          # (H, W) for image orientation
    hi = max(float(fuel.max()), 1e-6)
    frac = (fuel / hi)[..., None]
    img = (np.array(C_GROUND) + frac * (np.array(C_FUEL) - np.array(C_GROUND)))
    rr.log(path, rr.Image(img.astype(np.uint8)))


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
C_BURNT = [46, 42, 44]
C_HUMAN = [240, 236, 228]
C_SMOKE = [176, 176, 180]


def log_time(t_s: float) -> None:
    rr.set_time("sim_time", duration=float(t_s))


def log_fire(fire, path: str = "/world/fire", env: int = 0) -> None:
    """Burning cells, coloured by how fiercely, plus the scar left behind."""
    cell = fire.terrain.cell_size_m
    intensity = fire.intensity[env].cpu().numpy()
    burning = intensity > 0

    xs, ys = np.nonzero(burning)
    if len(xs):
        i = intensity[xs, ys][:, None]
        # Hot cores read yellow-white, dying edges red: the eye picks out the front.
        colour = (C_FIRE_COOL + i * (C_FIRE_HOT - C_FIRE_COOL)).astype(np.uint8)
        rr.log(f"{path}/flame", rr.Points3D(
            positions=np.stack([(xs + 0.5) * cell, (ys + 0.5) * cell,
                                np.full(len(xs), 0.3)], -1),
            colors=colour, radii=cell * 0.45,
        ))
    else:
        rr.log(f"{path}/flame", rr.Clear(recursive=False))

    xs, ys = np.nonzero(fire.burnt[env].cpu().numpy())
    rr.log(f"{path}/burnt", rr.Points3D(
        positions=np.stack([(xs + 0.5) * cell, (ys + 0.5) * cell,
                            np.full(len(xs), 0.05)], -1),
        colors=[C_BURNT], radii=cell * 0.4,
    ) if len(xs) else rr.Clear(recursive=False))


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
    colour = np.concatenate([np.tile(C_SMOKE, (len(ks), 1)),
                             (40 + 180 * v)[:, None]], -1).astype(np.uint8)
    rr.log(path, rr.Points3D(
        positions=np.stack([(xs + 0.5) * cell, (ys + 0.5) * cell, alt[ks]], -1),
        colors=colour, radii=cell * 0.8,
    ))


def log_human(human, path: str = "/world/human", env: int = 0) -> None:
    """The person, coloured by how panicked, with a goal marker."""
    p = human.pos[env].cpu().numpy()
    g = human.goal[env].cpu().numpy()
    panic = float(human.gamma[env])
    # Calm reads pale, panicked reads hot -- so you can see the state without a plot.
    colour = (np.array(C_HUMAN) * (1 - panic) + np.array([230, 90, 60]) * panic)

    rr.log(f"{path}/person", rr.Points3D(
        positions=[[p[0], p[1], 0.9]], colors=[colour.astype(np.uint8)], radii=0.45))
    rr.log(f"{path}/goal", rr.Points3D(
        positions=[[g[0], g[1], 0.5]], colors=[[110, 210, 140]], radii=1.2))
    rr.log(f"{path}/panic", rr.Scalars(panic))
    rr.log(f"{path}/speed", rr.Scalars(float(human.vel[env].norm())))
