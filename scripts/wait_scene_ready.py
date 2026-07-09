#!/usr/bin/env python3
"""
Wait until a multi-drone scene is genuinely ready for formal recording.

Readiness is defined conservatively:
- /clock is advancing
- every /droneN/engage service is advertised
- every required topic has published at least one message
- every drone has finite odom and has climbed above a minimum altitude
- sim time has reached a minimum threshold so takeoff/startup transients are excluded
"""

import argparse
import math
import sys
import time

import rospy
import rosservice
from geometry_msgs.msg import PointStamped, Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock


class SceneReadyWatcher:
    def __init__(self, num_drones: int):
        self.num_drones = num_drones
        self.sim_time = None
        self.odom = {i: None for i in range(1, num_drones + 1)}

        rospy.Subscriber("/clock", Clock, self._clock_cb, queue_size=1)
        for drone_id in range(1, num_drones + 1):
            rospy.Subscriber(
                f"/drone{drone_id}/ground_truth/state",
                Odometry,
                self._make_odom_cb(drone_id),
                queue_size=1,
            )

    def _clock_cb(self, msg: Clock):
        self.sim_time = msg.clock.to_sec()

    def _make_odom_cb(self, drone_id: int):
        def cb(msg: Odometry):
            self.odom[drone_id] = msg

        return cb

    def current_sim_time(self):
        return 0.0 if self.sim_time is None else float(self.sim_time)

    def drone_is_airborne_and_finite(self, drone_id: int, min_altitude: float) -> bool:
        msg = self.odom[drone_id]
        if msg is None:
            return False
        p = msg.pose.pose.position
        coords = (p.x, p.y, p.z)
        return all(math.isfinite(v) for v in coords) and p.z >= min_altitude


def remaining_timeout(deadline_wall: float) -> float:
    return max(0.1, deadline_wall - time.time())


def wait_for_services(num_drones: int, deadline_wall: float):
    pending = {f"/drone{i}/engage" for i in range(1, num_drones + 1)}
    while pending and time.time() < deadline_wall and not rospy.is_shutdown():
        try:
            services = set(rosservice.get_service_list())
        except Exception:
            services = set()
        pending = {srv for srv in pending if srv not in services}
        if pending:
            rospy.loginfo_throttle(
                2.0,
                f"[wait_scene_ready] waiting for engage services: {sorted(pending)}",
            )
            time.sleep(0.2)
    if pending:
        raise TimeoutError(f"engage services not ready before timeout: {sorted(pending)}")


def wait_for_required_topics(num_drones: int, deadline_wall: float):
    topic_specs = []
    for drone_id in range(1, num_drones + 1):
        topic_specs.extend(
            [
                (f"/drone{drone_id}/ground_truth/state", Odometry),
                (f"/drone{drone_id}/p_des", PointStamped),
                (f"/drone{drone_id}/v_des", Vector3Stamped),
                (f"/drone{drone_id}/cmd_vel_teacher", Twist),
            ]
        )

    for topic, msg_type in topic_specs:
        timeout = remaining_timeout(deadline_wall)
        rospy.loginfo(f"[wait_scene_ready] waiting for topic {topic} (timeout={timeout:.1f}s)")
        rospy.wait_for_message(topic, msg_type, timeout=timeout)


def main():
    ap = argparse.ArgumentParser(description="Wait for a formal multi-drone recording preflight state")
    ap.add_argument("--num-drones", type=int, default=4)
    ap.add_argument("--min-sim-time", type=float, default=8.0)
    ap.add_argument("--min-altitude", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=120.0, help="wall-clock timeout")
    args = ap.parse_args()

    rospy.init_node("wait_scene_ready", anonymous=True, disable_signals=True)
    watcher = SceneReadyWatcher(args.num_drones)
    deadline_wall = time.time() + args.timeout

    rospy.loginfo("[wait_scene_ready] waiting for /clock to advance...")
    while watcher.sim_time is None and time.time() < deadline_wall and not rospy.is_shutdown():
        time.sleep(0.1)
    if watcher.sim_time is None:
        print("[wait_scene_ready] ERROR: /clock did not start before timeout", file=sys.stderr)
        sys.exit(1)

    try:
        wait_for_services(args.num_drones, deadline_wall)
        wait_for_required_topics(args.num_drones, deadline_wall)
    except Exception as exc:
        print(f"[wait_scene_ready] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    rate = rospy.Rate(5)
    while time.time() < deadline_wall and not rospy.is_shutdown():
        sim_time = watcher.current_sim_time()
        ready_ids = [
            drone_id
            for drone_id in range(1, args.num_drones + 1)
            if watcher.drone_is_airborne_and_finite(drone_id, args.min_altitude)
        ]
        if sim_time >= args.min_sim_time and len(ready_ids) == args.num_drones:
            print(
                f"[wait_scene_ready] READY sim_time={sim_time:.3f}s "
                f"altitude>={args.min_altitude:.2f}m for all {args.num_drones} drones"
            )
            return
        rospy.loginfo_throttle(
            2.0,
            f"[wait_scene_ready] sim_time={sim_time:.2f}s "
            f"airborne={len(ready_ids)}/{args.num_drones} min_altitude={args.min_altitude:.2f}",
        )
        try:
            rate.sleep()
        except rospy.exceptions.ROSTimeMovedBackwardsException:
            pass

    print("[wait_scene_ready] ERROR: scene did not reach ready state before timeout", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
