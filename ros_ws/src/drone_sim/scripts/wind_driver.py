#!/usr/bin/env python3
"""
wind_driver - 风场扰动节点 (S4 场景)
论文 §5.1.2 风场模型:
  v_wind(t) = v_mean + A·sin(2πft) + n(t)
  F_wind = k_wind · v_wind  (作用于每架机 base_link 的水平扰动力)
参数依据:
  v_mean=2.0 m/s, A=1.0 m/s (原始文章给定)
  f=0.5 Hz, sigma=0.3 m/s, k_wind=0.3 (顶刊补全,参考 IEEE T-Cyber 2023)
"""
import rospy
import math
import time
import numpy as np
from geometry_msgs.msg import Wrench
from gazebo_msgs.srv import ApplyBodyWrench


class WindDriver:
    def __init__(self):
        rospy.init_node('wind_driver')

        self.num_drones = rospy.get_param('~num_drones', 4)
        self.v_mean_x = rospy.get_param('~v_mean_x', 2.0)
        self.v_mean_y = rospy.get_param('~v_mean_y', 0.0)
        self.amplitude = rospy.get_param('~amplitude', 1.0)
        self.frequency = rospy.get_param('~frequency', 0.5)
        self.phi = rospy.get_param('~phase', math.pi / 2.0)
        self.sigma = rospy.get_param('~sigma', 0.3)
        self.k_wind = rospy.get_param('~k_wind', 0.3)
        self.rate_hz = rospy.get_param('~rate', 10)
        startup_wait = rospy.get_param('~startup_wait', 10.0)
        self.seed = rospy.get_param('~seed', 42)

        np.random.seed(self.seed)

        rospy.loginfo(f'[wind] init, waiting {startup_wait}s for Gazebo full spawn...')
        time.sleep(startup_wait)

        srv_name = '/gazebo/apply_body_wrench'
        rospy.loginfo(f'[wind] waiting for service {srv_name}...')
        rospy.wait_for_service(srv_name, timeout=30)
        self.apply_wrench = rospy.ServiceProxy(srv_name, ApplyBodyWrench, persistent=True)
        rospy.loginfo(f'[wind] ready: v_mean=({self.v_mean_x}, {self.v_mean_y}), '
                      f'A={self.amplitude}, f={self.frequency}Hz, sigma={self.sigma}, '
                      f'k_wind={self.k_wind}, seed={self.seed}')

        self.t0 = rospy.Time.now().to_sec()

    def compute_wind(self, t):
        """v_wind(t) = v_mean + A·sin(2πft + φ) + n(t)"""
        gust_x = self.amplitude * math.sin(2 * math.pi * self.frequency * t)
        gust_y = self.amplitude * math.sin(2 * math.pi * self.frequency * t + self.phi)
        n_x = np.random.normal(0.0, self.sigma)
        n_y = np.random.normal(0.0, self.sigma)
        return (self.v_mean_x + gust_x + n_x,
                self.v_mean_y + gust_y + n_y)

    def apply_force_once(self, t, dt):
        v_x, v_y = self.compute_wind(t)
        wrench = Wrench()
        wrench.force.x = self.k_wind * v_x
        wrench.force.y = self.k_wind * v_y
        wrench.force.z = 0.0

        for i in range(self.num_drones):
            body_name = f'drone{i+1}::base_link'
            try:
                self.apply_wrench(
                    body_name=body_name,
                    reference_frame='world',
                    wrench=wrench,
                    start_time=rospy.Time(0),
                    duration=rospy.Duration(dt * 1.5)  # 略大于周期避免间隙
                )
            except Exception as e:
                rospy.logwarn_throttle(5, f'[wind] apply_wrench {body_name} failed: {e}')

    def run(self):
        r = rospy.Rate(self.rate_hz)
        dt = 1.0 / self.rate_hz
        log_period = max(1, int(self.rate_hz * 5))
        tick = 0
        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec() - self.t0
            self.apply_force_once(t, dt)
            tick += 1
            if tick % log_period == 0:
                v_x, v_y = self.compute_wind(t)
                rospy.loginfo(f'[wind t={t:.1f}s] v=({v_x:+.2f}, {v_y:+.2f}) m/s, '
                              f'F=({self.k_wind*v_x:+.2f}, {self.k_wind*v_y:+.2f}) N')
            try:
                r.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                pass


if __name__ == '__main__':
    try:
        WindDriver().run()
    except rospy.ROSInterruptException:
        pass