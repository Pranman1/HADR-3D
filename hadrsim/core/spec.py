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
from enum import Enum

SPEC_VERSION = 1

# ---------------------------------------------------------------------- provenance

class Provenance(str, Enum):
    """How a drone came by a piece of information. The honesty of the whole sim.

    It is easy to make human detection realistic and leave something else quietly
    magic.  Fire is the one that nearly slipped through: ground-truth fire is
    perfectly legitimate for deciding *where the person runs* -- that is just how a
    simulator works -- but the moment a drone reads it, the policy is planning around
    a fire map it never had to go and find.
    """

    PRIOR = "prior"
    """Known before the mission started. A real team has building footprints; so do
    we, from OpenStreetMap. Nothing to earn, and no dishonesty in having it."""

    OBSERVED = "observed"
    """Must be seen. Sensor range, field of view and line of sight all apply, and
    what was seen goes stale. Never read from the simulator's own state."""

    SHARED = "shared"
    """Someone else saw it and told us. Arrives through `comms`, so it is coarse,
    delayed, and may not arrive at all. How a real incident commander learns where a
    fire is: other aircraft, ground crews, satellite."""

    OWN = "own"
    """The drone's own memory of its own actions. Free by definition."""


RASTER_CHANNELS: tuple[tuple[str, Provenance], ...] = (
    ("buildings", Provenance.PRIOR),
    ("fire", Provenance.OBSERVED),      # plus SHARED beyond line of sight
    ("smoke", Provenance.OBSERVED),
    ("searched", Provenance.OWN),
)

# Fire and smoke are OBSERVED rather than needing a learned detector, and that is a
# deliberate, defensible line rather than laziness.  A person at 60 m is 1.7 m of
# low-contrast texture; a fire is a large, bright, hot thing with a plume above it.
# "Reliably detected inside the sensor footprint, absent outside it" is a fair
# description of reality for fire, where the identical claim about a person is not.
#
# Any code that fills these channels must therefore route through `sensing/`, never
# read `world/fire.py` state directly.  That rule is what this table exists to make
# checkable.


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
    # size never touches the network. See RASTER_CHANNELS for what each layer is and,
    # more importantly, what the drone had to do to earn it.
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
