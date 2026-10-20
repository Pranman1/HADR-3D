"""The paper's setup, at the paper's scale, as a control experiment.

Not part of the simulator -- a sandbox, kept separate on purpose.  Its whole job is
to answer one question that months of tuning on a real city cannot:

    on a problem where 71% is known to be achievable, does our PPO reach it?

If it does, the machinery is sound and everything failing on a 300 m city block is
about scale and exploration.  If it does not, the fault is in the learner, and no
amount of reward design on the harder problem was ever going to help.  Tuning
without a reference is what the last several runs were, and it is why this exists.

Numbers taken from the published configuration, not invented:

    map            maps/rust25.yaml, 25 x 25 m               (config/env/map/rust/sm)
    rescuers       llp [13, 11], hlp [12, 12]                 (config/env/deo.yaml)
    fire           one of [5,18] [19,18] [5,6] [19,6]         (config/env/deo.yaml)
    sensors        llp 5x5 forward, hlp 9x9 down              (map/rust/sm.yaml)
    speeds         rescuers 2 cells/step, evacuee 1           (map/rust/sm.yaml)
    fire spread    p = 0.01 on four neighbours, never out     (firegrid_env.py:125)

The deliberate cripplings -- teleporting drones, an oracle detector, perfect comms,
fixed panic -- are the point.  This scenario exists to reproduce the paper, not to
be right.
"""

import math
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from hadrsim.agents.airframes import Airframe
from hadrsim.env import Env
from hadrsim.eval import evaluate
from hadrsim.scenario import Scenario
from hadrsim.sensing.fov import Camera
from hadrsim.world.fire import FireParams

# Two cells per control step at 2 Hz is 4 m/s. Kinematic, so this is exactly what
# they do rather than what they aim for.
PAPER_DRONE = Airframe(
    name="paper rescuer", mass_kg=2.0, arm_m=0.25, inertia=(0.02, 0.02, 0.04),
    thrust_coeff=8.55e-6, moment_coeff=0.016, motor_tau_s=0.033,
    max_rotor_rads=1000.0, drag_coeff=0.1,
    max_speed_mps=4.0, max_climb_mps=2.0, max_yaw_rate_dps=180.0,
)

# A 9x9 window from 12 m up needs a half-angle of atan(4.5/12); a 5x5 window ahead
# needs the same reasoning at the low rescuer's range.
HLR_CAM = Camera(fov_deg=2 * math.degrees(math.atan2(4.5, 12.0)),
                 range_m=20.0, kind="downward")
LLR_CAM = Camera(fov_deg=2 * math.degrees(math.atan2(2.5, 3.0)),
                 range_m=6.0, pitch_deg=25.0, kind="forward")

# The paper's own weights (SpeedScaled: l0, l1, l2, l3 = 1, 1, 1, 10), plus a capture
# payment. None of this project's search terms: on a 25 m block the evacuee is inside
# a footprint from the first step, so there is nothing to search for.
PAPER_WEIGHTS = {
    "capture": 200.0,
    "evacuee_in_hlr_fov": 1.0,
    "evacuee_in_llr_fov": 1.0,
    "proximity": 20.0,
    "goal_progress": 10.0,
    "role_coordination": 1.0,
    "time_penalty": -0.02,
}


# The paper draws each episode from forty origin/destination pairs and four fire
# origins.  Running a single fixed pair instead is a *much* easier problem -- one
# trajectory to memorise rather than a task to generalise -- so a number measured
# that way is not comparable to the published 71%, however close it lands.  Both are
# available here because the single pair is the control that says whether the learner
# still works, and the forty are the comparison that is allowed in a paper.
def build(batch: int = 1024, device: str = "cuda", pairs: int = 40, **over) -> Env:
    sc = Scenario.preset("Paper grid (25 m)", cell_size_m=1.0, episode_seconds=50.0,
                         wind_mps=0.0, seed=0)
    sc.human_start = [[0.5, 0.5]]        # a corner, as in the paper's O-D pairs
    sc.human_goal = [[22.5, 24.5]]
    sc.drone_start = [12.5, 11.5]        # the paper's fixed rescuer base
    sc.fire_at = [5.0, 18.0]             # one of its four origins
    sc.randomise = pairs > 1
    sc.n_routes, sc.n_fire_origins = pairs, 4
    for k, v in over.items():
        setattr(sc, k, v)

    env = Env(
        sc, batch=batch, detector="oracle", fidelity="kinematic",
        weights=PAPER_WEIGHTS, cameras=[HLR_CAM, LLR_CAM],
        start_alt=[12.0, 3.0], capture_radius=1.5, airframe=PAPER_DRONE,
        device=device,
    )
    # The paper's fire: four neighbours, p = 0.01, no fuel, never goes out.
    env.fire.p = FireParams(spread_prob=0.01, spot_prob=0.0, burn_rate=0.0,
                            wind_boost=1.0, fuel_exponent=0.0)
    env.ALT_MAX = 25.0
    return env


def report(env, tag, net=None, deterministic=True):
    """The paper's four metrics, from the shared evaluator."""
    m = evaluate(env, net, deterministic=deterministic)
    print(f"  {tag:<26} capture {m.capture:5.1f}%   time {m.time_to_capture_s / env.clock.control_dt:5.1f}"
          f"   first seen {m.first_seen_s / env.clock.control_dt:5.1f}   in view {m.in_view:5.1f}%")
    return m.capture


if __name__ == "__main__":
    from hadrsim.policy.ppo import PPO, ActorCritic, PPOConfig

    pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    env = build(pairs=pairs)
    print(f"origin/destination pairs: {len(env.scenario.routes or [1])}, "
          f"fire origins: {len(env.scenario.fire_origins or [1])}")
    print(f"map {env.terrain.shape}, {env.clock.max_steps} decisions, "
          f"capture radius {env.capture_radius} m")
    print(f"HLR sees {HLR_CAM.fov_deg:.0f} deg down, LLR {LLR_CAM.fov_deg:.0f} deg ahead\n")

    print("PAPER, for reference        capture  71.0%   time  13.2   "
          "first seen   8.5   in view  30.8%\n")
    report(env, "do nothing")
    report(env, "untrained", ActorCritic().to(env.device))

    ppo = PPO(env, PPOConfig(horizon=25, epochs=3, entropy=0.005),
              encoder="conv", device=str(env.device))
    print("\ntraining")
    ppo.train(int(sys.argv[1]) if len(sys.argv) > 1 else 200, log_every=20)
    # Saved before anything is reported.  Last time a formatting bug in `report`
    # threw after training finished and took two and a half million steps of work
    # with it, because the save came afterwards.
    out = f"checkpoints/paper_parity_{pairs}pair.pt"
    ppo.save(out)
    print(f"\nsaved {out}\n")

    ppo.net.eval()
    report(env, "trained, deterministic", ppo.net)
    report(env, "trained, stochastic", ppo.net, deterministic=False)
