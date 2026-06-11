#!/usr/bin/env python3
import rospy
from hector_uav_msgs.srv import EnableMotors
import time

if __name__ == '__main__':
    rospy.init_node('enable_motors_delayed')
    # 等 8 秒,确保 Gazebo + controllers 完全启动
    time.sleep(8)
    try:
        rospy.wait_for_service('/enable_motors', timeout=10)
        srv = rospy.ServiceProxy('/enable_motors', EnableMotors)
        resp = srv(True)
        rospy.loginfo(f'[enable_motors_delayed] motors enabled: {resp.success}')
    except Exception as e:
        rospy.logerr(f'[enable_motors_delayed] failed: {e}')