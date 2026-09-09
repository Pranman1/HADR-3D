"""What a building looks like: its colour, and the window pattern on its walls.

Drawing only.  Nothing the simulation reads comes from this file -- occlusion, fire
and movement are all grid fields, and the policy never sees pixels.  This exists
because a city of identical grey blobs is hard to read, and being able to see what
you built is how you notice it is wrong.

**Colour comes from building type, not colour tags.**  Surveying the cached extracts
(Berkeley, SF, Manhattan) finds *zero* `building:colour` tags and six
`building:material`, against 30-odd distinct `building=*` values on nearly every
outline.  So a type palette is the thing that actually pays; explicit colour and
material are honoured where present, which is rarely.

**The texture is one shared atlas.**  A grid of ATLAS_CELLS x ATLAS_CELLS window
bays, and each wall maps into the corner of it -- a building of 8 floors and 12 bays
uses u in [0, 12/N], v in [0, 8/N].  Every building samples the same image, so the
whole city is one texture and one mesh, and nothing depends on the renderer
repeating UVs outside [0, 1].
"""

from __future__ import annotations

import numpy as np

FLOOR_M = 3.5      # vertical spacing of window bands
BAY_M = 4.0        # horizontal spacing of windows
ATLAS_CELLS = 128  # tallest supported building is 128 floors ~ 450 m

# Muted and desaturated on purpose: fire is red-orange and the drones are blue, and
# both have to stay legible against the city rather than competing with it.
PALETTE = {
    "apartments":   (150, 138, 126),
    "residential":  (150, 138, 126),
    "house":        (158, 142, 128),
    "detached":     (158, 142, 128),
    "bungalow":     (158, 142, 128),
    "dormitory":    (146, 134, 124),
    "hotel":        (140, 132, 128),
    "office":       (128, 134, 142),
    "commercial":   (134, 132, 136),
    "mixed_use":    (140, 134, 130),
    "retail":       (146, 130, 118),
    "industrial":   (120, 120, 122),
    "warehouse":    (120, 120, 122),
    "university":   (152, 140, 122),
    "school":       (152, 140, 122),
    "civic":        (146, 142, 134),
    "public":       (146, 142, 134),
    "government":   (146, 142, 134),
    "church":       (154, 146, 132),
    "theatre":      (144, 128, 128),
    "train_station": (132, 132, 136),
    "garage":       (112, 112, 114),
    "parking":      (112, 112, 114),
    "shed":         (118, 114, 108),
    "roof":         (176, 150, 96),    # canopies, so they read apart from walls
}
DEFAULT_COLOUR = (138, 134, 130)       # `building=yes`, which is most of OSM

MATERIALS = {
    "glass":     (120, 142, 152),
    "concrete":  (140, 140, 138),
    "brick":     (150, 106, 88),
    "wood":      (146, 118, 88),
    "metal":     (132, 136, 140),
    "stone":     (150, 146, 136),
    "plaster":   (162, 152, 138),
}

NAMED = {
    "white": (235, 233, 228), "grey": (140, 140, 140), "gray": (140, 140, 140),
    "black": (60, 60, 62), "red": (160, 80, 70), "brown": (140, 106, 82),
    "beige": (196, 178, 150), "yellow": (200, 180, 110), "blue": (110, 130, 160),
    "green": (110, 140, 110), "sandstone": (196, 176, 140), "cream": (216, 204, 180),
}


def _parse_colour(v: str):
    v = v.strip().lower()
    if v.startswith("#") and len(v) in (4, 7):
        if len(v) == 4:
            v = "#" + "".join(c * 2 for c in v[1:])
        return tuple(int(v[i:i + 2], 16) for i in (1, 3, 5))
    # `building:colour=brick` is common and means the material, not a named colour.
    return NAMED.get(v) or MATERIALS.get(v)


def colour_for(tags: dict, seed: int = 0) -> tuple[int, int, int]:
    """A building's colour: explicit tag, then material, then type, then default.

    `seed` applies a small deterministic jitter so a street of `building=yes` does
    not come out as one flat slab of identical grey.  Same building, same colour,
    every run.
    """
    base = None
    for key in ("building:colour", "building:color"):
        if key in tags and (c := _parse_colour(tags[key])):
            base = c
            break
    if base is None:
        base = MATERIALS.get(tags.get("building:material", ""))
    if base is None:
        base = PALETTE.get(tags.get("building", ""), DEFAULT_COLOUR)

    # Multiplicative, not additive.  An additive shift with a floor to keep buildings
    # off pure black distorts dark colours badly -- a saddle-brown's blue channel of
    # 19 gets pushed up to the floor and the hue slides toward grey.  Scaling
    # brightness leaves the hue alone at any darkness.
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    factor = rng.uniform(0.88, 1.12) * (1.0 + rng.uniform(-0.06, 0.06, 3))
    # No per-channel floor: any floor at all distorts a dark colour's hue, which is
    # the bug this replaced.  Staying visible against the ground is the palette's
    # job, not the jitter's.
    return tuple(int(np.clip(c * f, 0, 252)) for c, f in zip(base, factor))


def window_atlas(px_per_cell: int = 4) -> np.ndarray:
    """The shared facade image: a grid of lit and unlit windows on a pale wall.

    Kept white-ish so that multiplying by each building's vertex colour tints it --
    one image serves every building in the city.
    """
    n, p = ATLAS_CELLS, px_per_cell
    img = np.full((n * p, n * p, 3), 245, dtype=np.uint8)

    # One window per cell, inset, leaving a mullion of wall between them.
    rng = np.random.default_rng(7)                      # fixed: same city every run
    lit = rng.random((n, n)) < 0.18
    inset = max(1, p // 4)
    for row in range(n):
        for col in range(n):
            y, x = row * p + inset, col * p + inset
            h = w = p - 2 * inset
            img[y:y + h, x:x + w] = (215, 190, 130) if lit[row, col] else (96, 104, 116)

    # A floor line every storey reads as structure even when windows are tiny.
    img[::p, :] = 210
    return img


def roof_uv() -> tuple[float, float]:
    """Where roofs sample the atlas: a plain patch of wall between two floor lines."""
    return (0.5 + 0.5 / ATLAS_CELLS, 0.5 + 0.5 / ATLAS_CELLS)
