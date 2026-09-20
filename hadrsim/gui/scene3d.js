// The 3D view: the same scenario, seen from inside it.
//
// The top-down canvas is the better tool for reading a run and it stays.  This exists
// for what a plan view cannot show: that the high rescuer is eighty metres up and the
// low one is under the rooftops, that a roof is between a lens and the person, and what
// each camera's cone actually covers from where the airframe is pointing it.
//
// Everything drawn here is *true to the simulator*, not decorative:
//   - the quad is an X500 (0.25 m arms, 10-inch props) at the attitude the physics
//     has it at, because a leaning airframe points its camera somewhere else;
//   - each cone runs along the camera axis the detector actually tests against --
//     forward and 20 degrees down on the low rescuer, straight down on the high one;
//   - smoke leans with the live wind vector the fire is spreading under.
// The one concession is size: a 0.5 m aircraft is invisible on a 200 m block, so the
// airframe is drawn scaled up (default 8x) and says so.  The cones are never scaled.
//
// three.js is vendored beside this file, so the panel works with no internet.
// Orbiting is thirty lines here instead of another dependency.

const V = THREE.Vector3;
// The simulator is z-up; three.js is y-up.  One mapping, used everywhere.
const W3 = (x, y, z) => new V(x, z, y);

export class Scene3D {
  constructor(container) {
    this.el = container;
    this.droneScale = 8;                 // visual only; see the note above
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputEncoding = THREE.sRGBEncoding;
    this.el.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = this._sky();
    this.scene.fog = new THREE.Fog(0x1a1a22, 350, 1500);

    this.camera = new THREE.PerspectiveCamera(52, 1, 0.5, 4000);
    this.target = new V(0, 0, 0);
    this.orbit = { yaw: -0.7, pitch: 0.66, dist: 400 };

    // Dusk: low ambient, one warm low sun that throws long shadows, so the fire is
    // the brightest thing in the frame -- which is how it reads on the ground too.
    // Levels are set for sRGB output, where linear values read brighter than they
    // look on paper: a first pass at 1.1 / 0.55 / 0.45 blew every facade out white.
    this.scene.add(new THREE.AmbientLight(0x707890, 0.28));
    this.scene.add(new THREE.HemisphereLight(0x8aa0c8, 0x2a2622, 0.22));
    this.sun = new THREE.DirectionalLight(0xffd9b0, 0.55);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0008;
    this.scene.add(this.sun, this.sun.target);
    this.fireLight = new THREE.PointLight(0xff7a30, 0, 140, 2);
    this.scene.add(this.fireLight);

    this.groups = {};
    this.rotors = [];
    this._puff = this._puffTexture();
    this._bindMouse();
    this._loop();
  }

  // ------------------------------------------------------------------ input

  _bindMouse() {
    const c = this.renderer.domElement;
    let down = null;
    c.style.cursor = "grab";
    c.addEventListener("mousedown", e => { down = { x: e.clientX, y: e.clientY, ...this.orbit };
                                           c.style.cursor = "grabbing"; });
    addEventListener("mouseup", () => { down = null; c.style.cursor = "grab"; });
    addEventListener("mousemove", e => {
      if (!down) return;
      this.orbit.yaw = down.yaw + (e.clientX - down.x) * 0.006;
      // Stopped short of straight down: the up vector degenerates there.
      this.orbit.pitch = Math.min(1.45, Math.max(0.05, down.pitch + (e.clientY - down.y) * 0.005));
    });
    c.addEventListener("wheel", e => {
      e.preventDefault();
      this.orbit.dist = Math.min(2200, Math.max(12, this.orbit.dist * (1 + Math.sign(e.deltaY) * 0.12)));
    }, { passive: false });
  }

  resize() {
    const b = this.el.getBoundingClientRect();
    if (!b.width || !b.height) return;
    this.renderer.setSize(b.width, b.height, false);
    this.camera.aspect = b.width / b.height;
    this.camera.updateProjectionMatrix();
  }

  _loop() {
    requestAnimationFrame(() => this._loop());
    const { yaw, pitch, dist } = this.orbit;
    this.camera.position.set(this.target.x + dist * Math.cos(pitch) * Math.sin(yaw),
                             this.target.y + dist * Math.sin(pitch),
                             this.target.z + dist * Math.cos(pitch) * Math.cos(yaw));
    this.camera.lookAt(this.target);
    // Props spin; adjacent rotors counter-rotate, as on the real aircraft.
    for (let i = 0; i < this.rotors.length; i++) this.rotors[i].rotation.y += (i % 2 ? -1 : 1) * 0.7;
    if (this.fireLight.intensity > 0)
      this.fireLight.intensity = this._fireBase * (0.85 + 0.3 * Math.random());
    this.renderer.render(this.scene, this.camera);
  }

  // --------------------------------------------------------------- scenario

  _swap(name, object) {
    const old = this.groups[name];
    if (old) {
      this.scene.remove(old);
      old.traverse?.(o => { o.geometry?.dispose?.(); if (o.material && !o.material._shared) o.material.dispose?.(); });
    }
    this.groups[name] = object;
    if (object) this.scene.add(object);
  }

  _sky() {
    const c = document.createElement("canvas"); c.width = 4; c.height = 256;
    const g = c.getContext("2d"), grad = g.createLinearGradient(0, 0, 0, 256);
    grad.addColorStop(0, "#0a0c14"); grad.addColorStop(0.55, "#181a26");
    grad.addColorStop(0.8, "#2c2530"); grad.addColorStop(1, "#3a2a28");   // ember horizon
    g.fillStyle = grad; g.fillRect(0, 0, 4, 256);
    const t = new THREE.CanvasTexture(c); t.encoding = THREE.sRGBEncoding;
    return t;
  }

  setScenario(m) {
    const cell = m.cell, W = m.width * cell, H = m.height * cell;
    this.cell = cell;
    this.target.set(W / 2, 0, H / 2);
    this.orbit.dist = Math.max(W, H) * 1.4;

    const tex = new THREE.TextureLoader().load(m.map);
    tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.LinearFilter;
    tex.encoding = THREE.sRGBEncoding;
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(W, H),
                                  new THREE.MeshStandardMaterial({ map: tex, roughness: 1 }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(W / 2, 0, H / 2);
    ground.receiveShadow = true;
    this._swap("ground", ground);

    // Fire, burn scar and smoke density as a transparent sheet just above the ground.
    const overlay = new THREE.Mesh(new THREE.PlaneGeometry(W, H),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.95, depthWrite: false }));
    overlay.rotation.x = -Math.PI / 2;
    overlay.position.set(W / 2, 0.2, H / 2);
    this._swap("overlay", overlay);

    this._buildings(m.heights, cell);

    // A low sun from the south-west, shadow frustum sized to the block.
    const span = Math.max(W, H) * 0.8;
    this.sun.position.set(W / 2 - 420, 300, H / 2 + 260);
    this.sun.target.position.copy(this.target);
    Object.assign(this.sun.shadow.camera, { left: -span, right: span, top: span, bottom: -span, near: 10, far: 1400 });
    this.sun.shadow.camera.updateProjectionMatrix();

    for (const k of ["flames", "smoke", "agents"]) this._swap(k, new THREE.Group());
    this.rotors = [];
    this.resize();
  }

  _buildings(hs, cell) {
    // One instanced box per built cell.  A 150x150 block is a few thousand of them;
    // as separate meshes that is a few thousand draw calls and the view crawls.
    const runs = hs.runs, Wc = hs.w, boxes = [];
    let i = 0;
    for (const [value, count] of runs) {
      if (value > 0) for (let k = 0; k < count; k++) {
        const n = i + k; boxes.push([(n % Wc) * cell, Math.floor(n / Wc) * cell, value]);
      }
      i += count;
    }
    if (!boxes.length) { this._swap("buildings", new THREE.Group()); return; }

    const mesh = new THREE.InstancedMesh(
      new THREE.BoxGeometry(cell, 1, cell),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.92, metalness: 0.0 }),
      boxes.length);
    const dummy = new THREE.Object3D(), colour = new THREE.Color();
    boxes.forEach(([x, z, h], n) => {
      dummy.position.set(x + cell / 2, h / 2, z + cell / 2);
      dummy.scale.set(1, h, 1);
      dummy.updateMatrix();
      mesh.setMatrixAt(n, dummy.matrix);
      // Taller is lighter and cooler, so the skyline reads as depth rather than mass.
      const t = Math.min(1, h / 30);
      colour.setRGB(0.30 + 0.22 * t, 0.31 + 0.22 * t, 0.35 + 0.22 * t).convertSRGBToLinear();
      mesh.setColorAt(n, colour);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.castShadow = mesh.receiveShadow = true;
    this._swap("buildings", mesh);
  }

  // ------------------------------------------------------------------ frame

  setFrame(state, fireURI, flames, wind) {
    this._last = [state, fireURI, flames, wind];
    const cell = this.cell;
    if (fireURI && this.groups.overlay) {
      const t = new THREE.TextureLoader().load(fireURI);
      t.magFilter = THREE.NearestFilter; t.encoding = THREE.sRGBEncoding;
      const mat = this.groups.overlay.material;
      mat.map?.dispose(); mat.map = t; mat.needsUpdate = true;
    }
    const w = wind || [0, 0];
    this._flames(flames || [], cell);
    this._smoke(flames || [], cell, w);

    const g = new THREE.Group();
    this.rotors = [];
    for (const h of state.humans || []) g.add(this._person(h, cell));
    for (const d of state.drones || []) g.add(this._drone(d, cell));
    this._swap("agents", g);
  }

  redraw() { if (this._last) this.setFrame(...this._last); }

  // Slide the agents built by the last setFrame to interpolated positions.  Cheap:
  // no geometry is rebuilt, only transforms move, so this can run at display rate.
  moveAgents(state) {
    const cell = this.cell, g = this.groups.agents;
    if (!g || !cell) return;
    const kids = g.children, humans = state.humans || [], drones = state.drones || [];
    let i = 0;
    for (const h of humans) { const o = kids[i++]; if (o) o.position.set(h.x * cell, 0, h.y * cell); }
    for (const d of drones) { const o = kids[i++]; if (o) o.position.set(d.x * cell, d.z, d.y * cell); }
  }

  // ------------------------------------------------------------ the person

  _person(h, cell) {
    const g = new THREE.Group();
    const c = new THREE.Color().setRGB(0.89 + h.panic * 0.03, 0.90 - h.panic * 0.54, 0.93 - h.panic * 0.70);
    const skin = new THREE.MeshStandardMaterial({ color: c, roughness: 0.7 });
    const s = 3;                                             // drawn at 3x, like the aircraft
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.16 * s, 0.2 * s, 1.3 * s, 10), skin);
    body.position.y = 0.85 * s; body.castShadow = true;
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.14 * s, 12, 10), skin);
    head.position.y = 1.62 * s; head.castShadow = true;
    // A column of light: a person is findable on a 300 m block, and this is where
    // "reached safety" turns quiet.
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.6, 50, 8, 1, true),
      new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: h.reached ? 0.06 : 0.16,
                                    side: THREE.DoubleSide, depthWrite: false }));
    beam.position.y = 25;
    g.add(body, head, beam);
    g.position.set(h.x * cell, 0, h.y * cell);
    return g;
  }

  // ---------------------------------------------------------- the aircraft

  _drone(d, cell) {
    const hot = d.role === "hlr";
    const tint = hot ? 0x5aa9e6 : 0x8fd0ff;
    const g = new THREE.Group();
    g.position.set(d.x * cell, d.z, d.y * cell);

    // Attitude from the physics' own forward and up vectors, mapped into this frame.
    // Local: +x forward, +y up, +z right -- the frame the model below is built in.
    if (d.fwd && d.up) {
      const f = W3(...d.fwd).normalize(), u = W3(...d.up).normalize();
      const r = new V().crossVectors(f, u).normalize();
      g.setRotationFromMatrix(new THREE.Matrix4().makeBasis(f, u, r));
    }

    const body = this._airframe(d.arm_m || 0.25, tint, d.cam?.kind || (hot ? "downward" : "forward"));
    body.scale.setScalar(this.droneScale);
    g.add(body);

    // The camera's cone, along the axis the detector tests against.  Unscaled: it is
    // real geometry.  Clipped where the axis meets the ground or at the sensor's
    // range, whichever comes first; the ground plane hides whatever dips below.
    if (d.cam) {
      const axis = W3(...d.cam.axis).normalize();
      const toGround = axis.y < -1e-3 ? d.z / -axis.y : Infinity;
      const L = Math.max(0.5, Math.min(d.cam.range, toGround));
      const half = (d.cam.fov * Math.PI / 180) / 2;
      const geo = new THREE.ConeGeometry(L * Math.tan(half), L, 40, 1, true);
      geo.translate(0, -L / 2, 0);                             // apex at the lens
      const cone = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
        color: d.sees ? 0x7ce8b0 : tint, transparent: true, opacity: d.sees ? 0.22 : 0.10,
        side: THREE.DoubleSide, depthWrite: false }));
      // Built in the *world* group so the airframe's own scale never touches it.
      const holder = new THREE.Group();
      holder.quaternion.setFromUnitVectors(new V(0, -1, 0), axis);
      holder.add(cone);
      // Edge line, so the cone still reads when the fill is faint against sky.
      const edge = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 89),
        new THREE.LineBasicMaterial({ color: d.sees ? 0x7ce8b0 : tint, transparent: true, opacity: 0.35 }));
      holder.add(edge);
      const wrap = new THREE.Group();          // undo the body rotation: axis is already world
      wrap.quaternion.copy(g.quaternion).invert();
      wrap.add(holder);
      g.add(wrap);
    }

    if (d.trail && d.trail.length > 1) {
      const pts = d.trail.map(p => new V(p[0] * cell - g.position.x, 0, p[1] * cell - g.position.z));
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: tint, transparent: true, opacity: 0.45 }));
      const wrap = new THREE.Group(); wrap.quaternion.copy(g.quaternion).invert(); wrap.add(line); g.add(wrap);
    }
    return g;
  }

  _airframe(arm, tint, kind) {
    // An X500-class quad at true scale: 0.25 m arms, 10-inch (0.254 m) props, a
    // gimballed camera slung underneath.  Built in metres; the caller scales it.
    const g = new THREE.Group();
    const dark = this._mat("dark", 0x2a2d33, 0.6, 0.3);
    const carbon = this._mat("carbon", 0x141618, 0.45, 0.5);
    const accent = new THREE.MeshStandardMaterial({ color: tint, emissive: tint, emissiveIntensity: 0.55, roughness: 0.5 });

    const plate = new THREE.Mesh(new THREE.BoxGeometry(0.17, 0.035, 0.12), dark); plate.castShadow = true;
    const top = new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.03, 0.08), accent); top.position.y = 0.032;
    const battery = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.04, 0.05), carbon); battery.position.y = -0.037;
    g.add(plate, top, battery);

    const propR = 0.127, motorH = 0.03;
    [45, 135, 225, 315].forEach((deg, i) => {
      const a = deg * Math.PI / 180, ex = Math.cos(a) * arm, ez = Math.sin(a) * arm;
      const armMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.011, arm, 8), carbon);
      armMesh.rotation.z = Math.PI / 2; armMesh.rotation.y = -a;
      armMesh.position.set(ex / 2, 0, ez / 2);
      const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.017, 0.019, motorH, 12), dark);
      motor.position.set(ex, motorH / 2, ez);
      // A spinning prop: two blades plus the translucent disc a camera actually sees.
      const rotor = new THREE.Group(); rotor.position.set(ex, motorH + 0.008, ez);
      const blade = new THREE.Mesh(new THREE.BoxGeometry(propR * 2, 0.003, 0.022), carbon);
      const disc = new THREE.Mesh(new THREE.CircleGeometry(propR, 24),
        new THREE.MeshBasicMaterial({ color: 0xc8ccd4, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false }));
      disc.rotation.x = -Math.PI / 2;
      rotor.add(blade, disc);
      this.rotors.push(rotor);
      // Nav lights: red left, green right, as on the real aircraft.
      const led = new THREE.Mesh(new THREE.SphereGeometry(0.008, 6, 6),
        new THREE.MeshBasicMaterial({ color: ez < 0 ? 0xff3b30 : 0x30ff6a }));
      led.position.set(ex, -0.005, ez);
      g.add(armMesh, motor, rotor, led);
    });

    for (const zs of [-0.06, 0.06]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.012, 0.012), carbon);
      leg.position.set(0, -0.10, zs);
      const strut = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, 0.08, 6), carbon);
      strut.position.set(0, -0.06, zs);
      g.add(leg, strut);
    }

    // The gimbal.  Forward camera: on the nose, tilted 20 degrees down.  Downward
    // camera: under the centre, looking straight at the ground.
    const gimbal = new THREE.Group();
    const housing = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.035, 0.04), dark);
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.013, 0.022, 12), this._mat("lens", 0x0a0c10, 0.2, 0.8));
    const glass = new THREE.Mesh(new THREE.CircleGeometry(0.009, 12), new THREE.MeshBasicMaterial({ color: 0x9ad7ff }));
    if (kind === "downward") {
      gimbal.position.set(0, -0.075, 0);
      lens.position.y = -0.026; glass.position.y = -0.038; glass.rotation.x = -Math.PI / 2;
    } else {
      gimbal.position.set(0.085, -0.055, 0);
      const t = 20 * Math.PI / 180;
      lens.rotation.z = Math.PI / 2 + t;                     // barrel along (cos t, -sin t)
      lens.position.set(0.028 * Math.cos(t), -0.028 * Math.sin(t), 0);
      glass.position.set(0.04 * Math.cos(t), -0.04 * Math.sin(t), 0);
      glass.rotation.y = Math.PI / 2; glass.rotation.x = t;
    }
    gimbal.add(housing, lens, glass);
    g.add(gimbal);
    return g;
  }

  _mat(key, color, roughness, metalness) {
    // Shared across every airframe part and frame, and flagged so _swap keeps it.
    this._mats ??= {};
    if (!this._mats[key]) {
      this._mats[key] = new THREE.MeshStandardMaterial({ color, roughness, metalness });
      this._mats[key]._shared = true;
    }
    return this._mats[key];
  }

  // ------------------------------------------------------------- the fire

  _flames(cells, cell) {
    const g = new THREE.Group();
    let cx = 0, cz = 0, sum = 0;
    for (const [x, y, v] of cells) {
      const h = 1.5 + 6 * v, px = (x + 0.5) * cell, pz = (y + 0.5) * cell;
      const mat = new THREE.MeshBasicMaterial({
        color: new THREE.Color().setRGB(1, 0.32 + 0.5 * v, 0.08 + 0.25 * v),
        transparent: true, opacity: 0.45 + 0.4 * v, side: THREE.DoubleSide,
        depthWrite: false, blending: THREE.AdditiveBlending });
      for (const rot of [0, Math.PI / 2]) {
        const q = new THREE.Mesh(new THREE.PlaneGeometry(cell * 1.4, h), mat);
        q.position.set(px, h / 2, pz); q.rotation.y = rot;
        g.add(q);
      }
      cx += px * v; cz += pz * v; sum += v;
    }
    this._swap("flames", g);
    // One warm light at the fire's centroid rather than one per cell.
    this._fireBase = sum ? Math.min(4.5, 0.6 + sum / 12) : 0;
    this.fireLight.intensity = this._fireBase;
    if (sum) this.fireLight.position.set(cx / sum, 6, cz / sum);
  }

  _puffTexture() {
    const c = document.createElement("canvas"); c.width = c.height = 64;
    const g = c.getContext("2d"), r = g.createRadialGradient(32, 32, 2, 32, 32, 30);
    r.addColorStop(0, "rgba(200,200,205,0.55)"); r.addColorStop(1, "rgba(200,200,205,0)");
    g.fillStyle = r; g.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(c);
  }

  _smoke(cells, cell, wind) {
    // Columns of puffs over the hottest cells, leaning with the wind: a puff that has
    // risen h metres at about 2 m/s has been blown wind * h/2 metres downrange.
    const g = new THREE.Group();
    const hot = cells.slice().sort((a, b) => b[2] - a[2]).slice(0, 120);
    const wx = wind[0] || 0, wz = wind[1] || 0;
    for (const [x, y, v] of hot) {
      const seed = ((x * 73856093) ^ (y * 19349663)) >>> 0;
      const jit = ((seed % 1000) / 1000 - 0.5) * cell;
      for (let k = 0; k < 4; k++) {
        const h = 4 + k * 7 + (seed % 7);
        const sp = new THREE.Sprite(new THREE.SpriteMaterial({
          map: this._puff, color: 0xb9bcc4, transparent: true, depthWrite: false,
          opacity: (0.32 - 0.06 * k) * Math.min(1, 0.4 + v) }));
        const size = 5 + 4.5 * k;
        sp.scale.set(size, size, 1);
        sp.position.set((x + 0.5) * cell + jit + wx * h / 2, h, (y + 0.5) * cell - jit + wz * h / 2);
        g.add(sp);
      }
    }
    this._swap("smoke", g);
  }
}
