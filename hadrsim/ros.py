"""hadrsim-ros: the gym at batch 1, on ROS 2 topics.  Optional; never used in training.

    pip install -e .[ros]        # needs a sourced ROS 2 (Humble or later) on Linux
    hadrsim-ros --ros-args -p city:="Downtown Berkeley" -p policy:=last_sighting

Publishes, at the policy rate (2 Hz), in a fixed frame `map` with metres east/north/up:

    /hadrsim/hlr/pose, /hadrsim/llr/pose      geometry_msgs/PoseStamped
    /hadrsim/hlr/detection, /hadrsim/llr/detection
                                               geometry_msgs/PointStamped, the believed person,
                                               sent only when that drone has a detection
    /hadrsim/evacuee                           geometry_msgs/PointStamped, the truth (for rviz)
    /hadrsim/fire                              nav_msgs/OccupancyGrid: 100 burning, 50 burnt, 0 clear

Subscribes:

    /hadrsim/hlr/cmd_vel, /hadrsim/llr/cmd_vel geometry_msgs/Twist in the body frame, m/s and
                                               rad/s, scaled by the airframe's limits into the
                                               [-1, 1] setpoint the gym takes (the PX4 offboard shape)

With `policy` set, that policy flies and cmd_vel is ignored -- a demo.  Without it the
drones hover until something publishes commands, which is how an external autonomy
stack is tested against this world.
"""

from __future__ import annotations

import math

import torch


def main(argv=None) -> None:
    try:
        import rclpy
        from geometry_msgs.msg import PointStamped, PoseStamped, Twist
        from nav_msgs.msg import OccupancyGrid
        from rclpy.node import Node
    except ImportError as e:          # pragma: no cover
        raise SystemExit("hadrsim-ros needs a sourced ROS 2 installation (rclpy not importable)") from e

    from hadrsim.env import Env
    from hadrsim.policy import base
    from hadrsim.scenario import Scenario, autoplace

    class Bridge(Node):
        def __init__(self):
            super().__init__("hadrsim")
            p = lambda name, default: self.declare_parameter(name, default).value
            sc = Scenario.preset(p("city", "Downtown Berkeley"), radius_m=float(p("radius", 100.0)),
                                 cell_size_m=2.0, episode_seconds=float(p("seconds", 120.0)), wind_mps=0.0, seed=int(p("seed", 0)))
            self.env = Env(autoplace(sc), batch=1, fidelity=p("fidelity", "quadrotor"), detector=p("detector", "geometric"))
            self.env.launch_offset_m = float(p("launch", 25.0))
            name = p("policy", "")
            self.policy = base.load(name, "cpu") if name else None
            self.obs, self.state = self.env.reset(), None
            self.cmd = torch.zeros(1, 2, 4)
            self.roles = ("hlr", "llr")
            self.pose_pub = [self.create_publisher(PoseStamped, f"/hadrsim/{r}/pose", 10) for r in self.roles]
            self.det_pub = [self.create_publisher(PointStamped, f"/hadrsim/{r}/detection", 10) for r in self.roles]
            self.person_pub = self.create_publisher(PointStamped, "/hadrsim/evacuee", 10)
            self.fire_pub = self.create_publisher(OccupancyGrid, "/hadrsim/fire", 1)
            for i, r in enumerate(self.roles):
                self.create_subscription(Twist, f"/hadrsim/{r}/cmd_vel", lambda m, i=i: self._cmd(i, m), 10)
            self.create_timer(self.env.clock.control_dt / float(p("speed", 1.0)), self.tick)
            self.get_logger().info(f"{sc.place}: publishing under /hadrsim at {self.env.clock.control_hz:g} Hz"
                                   + (f", flown by {name}" if name else ", waiting for cmd_vel"))

        def _cmd(self, i: int, m) -> None:
            af = self.env.drones[i].af
            self.cmd[0, i] = torch.tensor([m.linear.x / af.max_speed_mps, m.linear.y / af.max_speed_mps,
                                           m.linear.z / af.max_climb_mps,
                                           math.degrees(m.angular.z) / af.max_yaw_rate_dps]).clamp(-1, 1)

        def _stamp(self, msg):
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            return msg

        @torch.no_grad()
        def tick(self) -> None:
            e = self.env
            if int(e.step_count[0]) >= e.clock.max_steps or bool(e.human.reached[0] | e.human.harmed[0]):
                self.obs, self.state = e.reset(), None
            act = self.cmd
            if self.policy is not None:
                act, _, _, self.state = self.policy.act(self.obs, self.state, deterministic=True)
            self.obs, _, _, info = e.step(act)

            for i, d in enumerate(e.drones):
                ps = self._stamp(PoseStamped())
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in d.pos[0])
                ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = (float(v) for v in d.quat[0])
                self.pose_pub[i].publish(ps)
                det = e._last[0][i]
                if float(det.confidence[0, 0]) > 0:
                    pt = self._stamp(PointStamped())
                    pt.point.x, pt.point.y, pt.point.z = (float(v) for v in det.position[0, 0])
                    self.det_pub[i].publish(pt)
            truth = self._stamp(PointStamped())
            truth.point.x, truth.point.y = float(e.human.pos[0, 0]), float(e.human.pos[0, 1])
            self.person_pub.publish(truth)

            grid = self._stamp(OccupancyGrid())
            grid.info.resolution = e.terrain.cell_size_m
            grid.info.width, grid.info.height = e.terrain.shape
            cells = torch.zeros(e.terrain.shape, dtype=torch.int8)
            cells[e.fire.burnt[0]] = 50
            cells[e.fire.burning[0]] = 100
            grid.data = cells.T.reshape(-1).tolist()          # row-major in y, as the message wants
            self.fire_pub.publish(grid)

    rclpy.init(args=argv)
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
