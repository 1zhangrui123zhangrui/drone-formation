#!/usr/bin/env python3
"""Template ROS node for online end-to-end formation control."""

from collections import deque

import rospy
import yaml
from std_msgs.msg import Float64MultiArray


class OnlineLstmController:
    def __init__(self):
        self.config_path = rospy.get_param("~controller_config", "")
        self.config = self._load_config(self.config_path)
        self.sequence_length = int(self.config.get("sequence_length", 20))
        self.input_dim = int(self.config.get("input_dim", 9))
        self.state_buffer = deque(maxlen=self.sequence_length)
        self.command_pub = rospy.Publisher("/formation/cmd", Float64MultiArray, queue_size=10)
        self.state_sub = rospy.Subscriber("/formation/state_vector", Float64MultiArray, self._state_callback)
        rospy.loginfo("Online controller loaded config: %s", self.config.get("controller_name", "unknown"))

    @staticmethod
    def _load_config(config_path):
        if not config_path:
            rospy.logwarn("No controller_config provided; using defaults.")
            return {}
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _state_callback(self, msg):
        if len(msg.data) < self.input_dim:
            rospy.logwarn_throttle(5.0, "State vector shorter than expected input_dim=%d", self.input_dim)
            return

        self.state_buffer.append(list(msg.data[: self.input_dim]))
        if len(self.state_buffer) < self.sequence_length:
            return

        command = self._infer_command()
        output = Float64MultiArray()
        output.data = command
        self.command_pub.publish(output)

    def _infer_command(self):
        # Placeholder inference: replace with MATLAB-exported model or Torch runtime.
        latest_state = self.state_buffer[-1]
        return [0.1 * value for value in latest_state[:4]]


def main():
    rospy.init_node("online_lstm_controller", anonymous=False)
    OnlineLstmController()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
