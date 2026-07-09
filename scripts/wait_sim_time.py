#!/usr/bin/env python3
"""
Wait for a target amount of simulated time to elapse.

This is used instead of wall-clock sleeping so paper data collection is measured
in the same time base as the recorded messages.
"""

import argparse
import sys
import time

import rospy
from rosgraph_msgs.msg import Clock


class SimClockWatcher:
    def __init__(self):
        self.sim_time = None
        rospy.Subscriber("/clock", Clock, self._clock_cb, queue_size=1)

    def _clock_cb(self, msg: Clock):
        self.sim_time = msg.clock.to_sec()

    def now(self):
        return None if self.sim_time is None else float(self.sim_time)


def main():
    ap = argparse.ArgumentParser(description="Wait until a simulated-time duration has elapsed")
    ap.add_argument("--duration", type=float, required=True, help="sim seconds to wait")
    ap.add_argument("--start-time", type=float, default=None, help="optional sim start time")
    ap.add_argument("--timeout", type=float, default=300.0, help="wall-clock timeout")
    args = ap.parse_args()

    rospy.init_node("wait_sim_time", anonymous=True, disable_signals=True)
    watcher = SimClockWatcher()
    deadline_wall = time.time() + args.timeout

    rospy.loginfo("[wait_sim_time] waiting for /clock...")
    while watcher.now() is None and time.time() < deadline_wall and not rospy.is_shutdown():
        time.sleep(0.1)
    if watcher.now() is None:
        print("[wait_sim_time] ERROR: /clock did not start before timeout", file=sys.stderr)
        sys.exit(1)

    start_time = watcher.now() if args.start_time is None else float(args.start_time)
    target_time = start_time + args.duration
    rate = rospy.Rate(5)

    while time.time() < deadline_wall and not rospy.is_shutdown():
        now = watcher.now()
        if now is not None and now >= target_time:
            print(
                f"[wait_sim_time] DONE start={start_time:.3f}s target={target_time:.3f}s now={now:.3f}s"
            )
            return
        rospy.loginfo_throttle(
            5.0,
            f"[wait_sim_time] sim_now={0.0 if now is None else now:.2f}s target={target_time:.2f}s",
        )
        try:
            rate.sleep()
        except rospy.exceptions.ROSTimeMovedBackwardsException:
            pass

    print(
        f"[wait_sim_time] ERROR: timed out waiting for sim time target {target_time:.3f}s",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
