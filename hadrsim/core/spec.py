"""The observation contract: shapes and provenance of everything a drone is told.
A policy trained against this runs on anything that fills it."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

SPEC_VERSION = 1

# ---------------------------------------------------------------------- provenance

class Provenance(str, Enum):
    """How a drone came by a piece of information: prior map, observed, shared, or its own memory."""

    PRIOR = "prior"          # known before take-off: the map
    OBSERVED = "observed"    # seen by this drone, subject to range, view and line of sight
    SHARED = "shared"        # told by someone else; may be late or missing
    OWN = "own"              # the drone's own memory of its actions


RASTER_CHANNELS: tuple[tuple[str, Provenance], ...] = (
    ("buildings", Provenance.PRIOR),
    ("fire", Provenance.OBSERVED),      # plus SHARED beyond line of sight
    ("smoke", Provenance.OBSERVED),
    ("searched", Provenance.OWN),
)


# --------------------------------------------------------------------------- action

ACTION_FIELDS = ("v_forward", "v_lateral", "v_up", "yaw_rate")
ACTION_DIM = len(ACTION_FIELDS)


# ----------------------------------------------------------------------- observation

@dataclass(frozen=True)
class ObsSpec:
    """Shapes of one drone's observation."""

    self_dim: int = 10

    # --- detections: rel_pos(3) rel_vel(3) confidence(1) age_s(1), K fixed slots
    n_detection_slots: int = 4
    detection_dim: int = 8

    # --- teammate: rel_pos(3) confidence(1) age_s(1). Arrives via comms, may be stale.
    teammate_dim: int = 5

    raster_channels: int = len(RASTER_CHANNELS)
    raster_size: int = 32
    raster_metres_per_cell: float = 2.0

    # --- task: rel_goal(3) time_remaining_s(1)
    task_dim: int = 4

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "self": (self.self_dim,),
            "detections": (self.n_detection_slots, self.detection_dim),
            "teammate": (self.teammate_dim,),
            "raster": (self.raster_channels, self.raster_size, self.raster_size),
            "task": (self.task_dim,),
        }

    @property
    def flat_dim(self) -> int:
        """Size of everything except the raster, which goes to the CNN separately."""
        return (
            self.self_dim
            + self.n_detection_slots * self.detection_dim
            + self.teammate_dim
            + self.task_dim
        )


DEFAULT_OBS = ObsSpec()


def validate(obs: dict, spec: ObsSpec = DEFAULT_OBS, batch: int | None = None) -> None:
    """Raise if `obs` does not match the contract."""
    expected = spec.shapes
    missing = set(expected) - set(obs)
    extra = set(obs) - set(expected)
    if missing or extra:
        raise ValueError(f"observation keys off contract: missing={missing} extra={extra}")
    for key, shape in expected.items():
        got = tuple(obs[key].shape)
        want = shape if batch is None else (batch, *shape)
        if got != want:
            raise ValueError(f"observation['{key}'] has shape {got}, contract says {want}")
