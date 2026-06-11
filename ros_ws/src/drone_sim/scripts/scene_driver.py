#!/usr/bin/env python3
"""
scene_driver - 任务驱动器 (4 机版,含 S3 队形切换)
论文 §3.2 公式 (3.5)-(3.9): 中心轨迹 c(t) + 编队偏移 oᵢ(t) → 各机期望位置 pᵢ_des
"""
import rospy
import math
import time
import numpy as np
from geometry_msgs.msg import PointStamped, Vector3Stamped


class SceneDriver:
    def __init__(self):
        rospy.init_node('scene_driver')

        self.scenario = rospy.get_param('~scenario', 'circle')
        self.radius = rospy.get_param('~radius', 3.0)
        self.omega = rospy.get_param('~omega', 0.4)
        self.z_target = rospy.get_param('~z_target', 1.8)
        self.v_line = rospy.get_param('~v_line', 1.0)
        self.rate_hz = rospy.get_param('~rate', 10)
        self.num_drones = rospy.get_param('~num_drones', 4)
        self.formation_size = rospy.get_param('~formation_size', 0.5)
        # S3 队形切换时刻
        self.t_switch_1 = rospy.get_param('~t_switch_1', 40.0)   # 矩形→菱形
        self.t_switch_2 = rospy.get_param('~t_switch_2', 80.0)   # 菱形→三角形+中心
        startup_wait = rospy.get_param('~startup_wait', 5.0)

        rospy.loginfo(f'[scene_driver] init: scenario={self.scenario}, N={self.num_drones}')
        rospy.loginfo(f'[scene_driver] sleeping {startup_wait}s (wall time)...')
        time.sleep(startup_wait)

        self.pub_p = []
        self.pub_v = []
        for i in range(self.num_drones):
            ns = f'/drone{i+1}'
            self.pub_p.append(rospy.Publisher(f'{ns}/p_des', PointStamped, queue_size=10))
            self.pub_v.append(rospy.Publisher(f'{ns}/v_des', Vector3Stamped, queue_size=10))

        time.sleep(1.0)
        self.t0 = rospy.Time.now().to_sec()
        rospy.loginfo(f'[scene_driver] starting publish loop: R={self.radius}, '
                      f'omega={self.omega}, z={self.z_target}, formation_size={self.formation_size}')

    def get_formation_offsets(self, t):
        """根据场景与时间返回 N 个偏移向量 (论文 §3.2 公式 3.8)"""
        s = self.formation_size

        if self.scenario != 'reconfig':
            # 默认矩形编队 (S1/S2/S2'/S4/S5 共用)
            return [np.array([ s,  s]), np.array([-s,  s]),
                    np.array([-s, -s]), np.array([ s, -s])][:self.num_drones]

        # S3 reconfig: 矩形 → 菱形 → 三角形+中心
        if t < self.t_switch_1:
            # 阶段 1: 矩形 (与默认一致)
            return [np.array([ s,  s]), np.array([-s,  s]),
                    np.array([-s, -s]), np.array([ s, -s])]
        elif t < self.t_switch_2:
            # 阶段 2: 菱形 (顶点向外,边长保持相同)
            d = s * 1.414
            return [np.array([ d,  0]), np.array([ 0,  d]),
                    np.array([-d,  0]), np.array([ 0, -d])]
        else:
            # 阶段 3: 三角形 + 中心机
            d = s * 1.2
            return [np.array([ 0,  d]),
                    np.array([-d, -d * 0.6]),
                    np.array([ d, -d * 0.6]),
                    np.array([ 0,  0])]

    def center_trajectory(self, t):
        if self.scenario == 'hover':
            return np.array([0.0, 0.0]), np.array([0.0, 0.0])
        elif self.scenario in ('circle', 'reconfig'):
            # reconfig 中心轨迹也走圆周,只是编队偏移在变
            cx = self.radius * math.cos(self.omega * t)
            cy = self.radius * math.sin(self.omega * t)
            vx = -self.radius * self.omega * math.sin(self.omega * t)
            vy = self.radius * self.omega * math.cos(self.omega * t)
            return np.array([cx, cy]), np.array([vx, vy])
        elif self.scenario == 'lemniscate':
            cx = self.radius * math.sin(self.omega * t)
            cy = (self.radius / 2.0) * math.sin(2.0 * self.omega * t)
            vx = self.radius * self.omega * math.cos(self.omega * t)
            vy = self.radius * self.omega * math.cos(2.0 * self.omega * t)
            return np.array([cx, cy]), np.array([vx, vy])
        elif self.scenario == 'line':
            return np.array([self.v_line * t, 0.0]), np.array([self.v_line, 0.0])
        else:
            rospy.logwarn_throttle(5, f'Unknown scenario: {self.scenario}')
            return np.array([0.0, 0.0]), np.array([0.0, 0.0])

    def publish_once(self, t):
        c, cdot = self.center_trajectory(t)
        offsets = self.get_formation_offsets(t)
        ts = rospy.Time.now()
        for i in range(self.num_drones):
            msg_p = PointStamped()
            msg_p.header.stamp = ts
            msg_p.header.frame_id = 'world'
            msg_p.point.x = float(c[0] + offsets[i][0])
            msg_p.point.y = float(c[1] + offsets[i][1])
            msg_p.point.z = float(self.z_target)
            self.pub_p[i].publish(msg_p)

            msg_v = Vector3Stamped()
            msg_v.header = msg_p.header
            msg_v.vector.x = float(cdot[0])
            msg_v.vector.y = float(cdot[1])
            msg_v.vector.z = 0.0
            self.pub_v[i].publish(msg_v)

    def run(self):
        r = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec() - self.t0
            self.publish_once(t)
            try:
                r.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                pass


if __name__ == '__main__':
    try:
        SceneDriver().run()
    except rospy.ROSInterruptException:
        pass