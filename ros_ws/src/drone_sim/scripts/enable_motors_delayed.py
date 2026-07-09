#!/usr/bin/env python3
"""
延时使能电机 (4 机版,RAFALAMAO fork 适配)
RAFALAMAO 版 hector_quadrotor 提供的服务名是 /droneN/engage,
类型是 std_srvs/Empty (无参数,直接触发)。
"""
import rospy
import time
from std_srvs.srv import Empty


if __name__ == '__main__':
    rospy.init_node('enable_motors_delayed')
    num_drones = rospy.get_param('~num_drones', 4)
    wait_sec = rospy.get_param('~wait_sec', 8.0)
    retry_period = rospy.get_param('~retry_period', 0.5)
    per_drone_timeout = rospy.get_param('~per_drone_timeout', 20.0)

    rospy.loginfo(f'[enable_motors] waiting {wait_sec}s before engaging {num_drones} drones...')
    time.sleep(wait_sec)

    success_count = 0
    for i in range(num_drones):
        srv_name = f'/drone{i+1}/engage'
        deadline = time.time() + per_drone_timeout
        last_error = None
        while time.time() < deadline and not rospy.is_shutdown():
            try:
                rospy.wait_for_service(srv_name, timeout=min(2.0, max(0.1, deadline - time.time())))
                srv = rospy.ServiceProxy(srv_name, Empty)
                srv()  # Empty 服务无参数
                rospy.loginfo(f'[enable_motors] {srv_name}: engaged OK')
                success_count += 1
                last_error = None
                break
            except Exception as e:
                last_error = e
                rospy.logwarn_throttle(2.0, f'[enable_motors] retrying {srv_name}: {e}')
                time.sleep(retry_period)
        if last_error is not None:
            rospy.logerr(f'[enable_motors] {srv_name} FAILED after retries: {last_error}')
    
    rospy.loginfo(f'[enable_motors] DONE: {success_count}/{num_drones} drones engaged')
