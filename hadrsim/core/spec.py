"""THE CONTRACT.

The one frozen description of what a drone can see and what a drone can do.

This is the only thing the training environment and the running environment share.
As long as both honour it, a network trained in the fast headless environment runs
unchanged in the real-time ROS 2 one -- and later, on a real drone.

Changing anything here is a deliberate, versioned act: bump SPEC_VERSION, and every
saved checkpoint trained against the old version becomes unloadable on purpose.

Two rules that live here rather than in any one environment:

  1. Detections carry confidence and age.  The oracle detector just sets confidence
     to 1.0 and age to 0.0.  That is what lets a realistic detector be swapped in
     without a single change downstream.

  2. Teammate state is a detection like any other -- it arrives through comms and
     may be stale.  Perfect communication is a *setting*, never an assumption.

And one rule the environments must enforce, stated here because this is where anyone
looking for it will come:

  3. The reward may only read what the policy reads.  If the reward looks up ground
     truth on its own, degrading perception changes nothing, because the answer
     still arrives through the reward channel.
"""

from __future__ import annotations

from dataclasses import dataclass

SPEC_VERSION = 1

# --------------------------------------------------------------------------- action

ACTION_FIELDS = ("v_forward", "v_lateral", "v_up", "yaw_rate")
ACTION_DIM = len(ACTION_FIELDS)

# All four are in [-1, 1] and are scaled by the airframe's limits into a *velocity
# setpoint in the drone's own frame*.  This is deliberately the native PX4 offboard
# message, so the jump to real hardware needs no translation layer.


# ----------------------------------------------------------------------- observation

@dataclass(frozen=True)
class ObsSpec:
    """Shapes of one drone's observation. Frozen: same for every drone, every env."""

    # --- itself: pos(3) vel(3) yaw(1) alt_agl(1) pitch(1) roll(1)
    # pitch and roll are in here because the camera is bolted to the airframe: a real
    # quadrotor tilts to accelerate, so where it looks depends on how it is moving.
    self_dim: int = 10

    # --- detections: rel_pos(3) rel_vel(3) confidence(1) age_s(1), K fixed slots
    n_detection_slots: int = 4
    detection_dim: int = 8

    # --- teammate: rel_pos(3) confidence(1) age_s(1). Arrives via comms, may be stale.
    teammate_dim: int = 5

    # --- surroundings: a picture centred on the drone, always the same size, so map
    # size never touches the network. Channels: occupancy, fire, smoke, already-looked.
    raster_channels: int = 4
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

    @property
    def raster_extent_m(self) -> float:
        return self.raster_size * self.raster_metres_per_cell


DEFAULT_OBS = ObsSpec()


# ------------------------------------------------------------------- sentinel values

# A detection slot that holds nothing. Confidence 0.0 is the signal; the position
# fields are zeroed rather than set to -1, so a network cannot learn to read the
# sentinel itself as a location.
EMPTY_CONFIDENCE = 0.0
MAX_DETECTION_AGE_S = 10.0  # older than this and the slot is cleared


def validate(obs: dict, spec: ObsSpec = DEFAULT_OBS, batch: int | None = None) -> None:
    """Raise if `obs` does not match the contract. Cheap; call it in tests, not loops."""
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
