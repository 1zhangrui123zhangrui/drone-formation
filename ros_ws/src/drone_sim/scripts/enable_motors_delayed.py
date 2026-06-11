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

    rospy.loginfo(f'[enable_motors] waiting {wait_sec}s before engaging {num_drones} drones...')
    time.sleep(wait_sec)

    success_count = 0
    for i in range(num_drones):
        srv_name = f'/drone{i+1}/engage'
        try:
            rospy.wait_for_service(srv_name, timeout=5)
            srv = rospy.ServiceProxy(srv_name, Empty)
            srv()  # Empty 服务无参数
            rospy.loginfo(f'[enable_motors] {srv_name}: engaged OK')
            success_count += 1
        except Exception as e:
            rospy.logerr(f'[enable_motors] {srv_name} FAILED: {e}')
    
    rospy.loginfo(f'[enable_motors] DONE: {success_count}/{num_drones} drones engaged')