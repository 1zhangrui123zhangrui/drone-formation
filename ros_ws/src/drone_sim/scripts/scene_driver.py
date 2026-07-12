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
        self.prehover_duration = rospy.get_param('~prehover_duration', 0.0)
        self.transition_duration = rospy.get_param('~transition_duration', 0.0)
        # S3 队形切换时刻
        self.t_switch_1 = rospy.get_param('~t_switch_1', 40.0)   # 矩形→菱形
        self.t_switch_2 = rospy.get_param('~t_switch_2', 80.0)   # 菱形→三角形+中心
        self.formation_transition_duration = rospy.get_param('~formation_transition_duration', 8.0)
        self.reconfig_stage3_scale = rospy.get_param('~reconfig_stage3_scale', 1.8)
        self.reconfig_via_scale = rospy.get_param('~reconfig_via_scale', 3.6)
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
                      f'omega={self.omega}, z={self.z_target}, formation_size={self.formation_size}, '
                      f'prehover={self.prehover_duration}, transition={self.transition_duration}')

    @staticmethod
    def _smoothstep(x):
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _reconfig_offset_sets(self):
        s = self.formation_size
        rectangle = [np.array([ s,  s]), np.array([-s,  s]),
                     np.array([-s, -s]), np.array([ s, -s])]

        d_diamond = s * 1.414
        diamond = [np.array([ d_diamond,  0]), np.array([ 0,  d_diamond]),
                   np.array([-d_diamond,  0]), np.array([ 0, -d_diamond])]

        d_tri = s * self.reconfig_stage3_scale
        triangle_center = [np.array([ 0,  d_tri]),
                           np.array([-d_tri, -d_tri * 0.6]),
                           np.array([ d_tri, -d_tri * 0.6]),
                           np.array([ 0,  0])]

        d_via = s * self.reconfig_via_scale
        triangle_via = [np.array([ 0,  d_via]),
                        np.array([-d_via, -d_via * 0.6]),
                        np.array([ d_via, -d_via * 0.6]),
                        np.array([ 0, -d_via])]
        return rectangle, diamond, triangle_via, triangle_center

    @staticmethod
    def _blend_offsets(offsets_a, offsets_b, alpha):
        return [(1.0 - alpha) * a + alpha * b for a, b in zip(offsets_a, offsets_b)]

    def get_formation_offsets(self, t):
        """根据场景与时间返回 N 个偏移向量 (论文 §3.2 公式 3.8)"""
        s = self.formation_size

        if self.scenario != 'reconfig':
            # 默认矩形编队 (S1/S2/S2'/S4/S5 共用)
            return [np.array([ s,  s]), np.array([-s,  s]),
                    np.array([-s, -s]), np.array([ s, -s])][:self.num_drones]

        # S3 reconfig: 矩形 → 菱形 → 三角形+中心.
        # Offset changes are smoothed; hard jumps at 40/80 s can excite the
        # repulsion term and destabilize the center drone during stage 3.
        rectangle, diamond, triangle_via, triangle_center = self._reconfig_offset_sets()
        dt = max(float(self.formation_transition_duration), 1e-3)
        half_dt = 0.5 * dt

        if t < self.t_switch_1:
            return rectangle[:self.num_drones]
        if t < self.t_switch_1 + dt:
            alpha = self._smoothstep((t - self.t_switch_1) / dt)
            return self._blend_offsets(rectangle, diamond, alpha)[:self.num_drones]
        if t < self.t_switch_2:
            return diamond[:self.num_drones]
        if t < self.t_switch_2 + half_dt:
            alpha = self._smoothstep((t - self.t_switch_2) / half_dt)
            return self._blend_offsets(diamond, triangle_via, alpha)[:self.num_drones]
        if t < self.t_switch_2 + dt:
            alpha = self._smoothstep((t - self.t_switch_2 - half_dt) / half_dt)
            return self._blend_offsets(triangle_via, triangle_center, alpha)[:self.num_drones]
        return triangle_center[:self.num_drones]

    def base_center_trajectory(self, t):
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

    @staticmethod
    def _cubic_hermite(p0, v0, p1, v1, s, duration):
        h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        h10 = s**3 - 2.0 * s**2 + s
        h01 = -2.0 * s**3 + 3.0 * s**2
        h11 = s**3 - s**2

        dh00 = 6.0 * s**2 - 6.0 * s
        dh10 = 3.0 * s**2 - 4.0 * s + 1.0
        dh01 = -6.0 * s**2 + 6.0 * s
        dh11 = 3.0 * s**2 - 2.0 * s

        pos = h00 * p0 + h10 * duration * v0 + h01 * p1 + h11 * duration * v1
        vel = (
            dh00 * p0 + dh10 * duration * v0 + dh01 * p1 + dh11 * duration * v1
        ) / duration
        return pos, vel

    def center_trajectory(self, t):
        dynamic_scene = self.scenario != 'hover'
        has_smooth_entry = dynamic_scene and (
            self.prehover_duration > 0.0 or self.transition_duration > 0.0
        )
        if not has_smooth_entry:
            return self.base_center_trajectory(t)

        if t < self.prehover_duration:
            return np.array([0.0, 0.0]), np.array([0.0, 0.0])

        transition_start = self.prehover_duration
        transition_end = self.prehover_duration + self.transition_duration
        if self.transition_duration > 0.0 and t < transition_end:
            s = np.clip((t - transition_start) / self.transition_duration, 0.0, 1.0)
            p0 = np.array([0.0, 0.0])
            v0 = np.array([0.0, 0.0])
            p1, v1 = self.base_center_trajectory(0.0)
            return self._cubic_hermite(p0, v0, p1, v1, s, self.transition_duration)

        return self.base_center_trajectory(t - transition_end)

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
