#!/usr/bin/env python3
"""
teacher_controller - Teacher PD 控制器 (4 机版)
论文 §3.3 实现:
  水平: PD + 碰撞分离 (公式 3.10, 3.11)
  垂直: PD (公式 3.12)
  偏航: 阻尼 (公式 3.13)
  输出: 经分量饱和后 (公式 3.14)
设计原则: 保持 PD 控制律纯净(不加前馈/积分),
          以保留 Student 网络通过 p_des/v_des 学习隐式前馈的提升空间。

修复点 (4 机版本):
1. 启动后等待 5 秒,确保 4 机 spawn 完成,subscriber 不漏连接
2. 周期性日志:每 3 秒打印每架机数据接收状态
3. 加 wall time 兜底:避免 sim time 卡死
"""
import rospy
import time
import numpy as np
from geometry_msgs.msg import PointStamped, Vector3Stamped, Twist, TwistStamped
from nav_msgs.msg import Odometry


class TeacherController:
    def __init__(self):
        rospy.init_node('teacher_controller')

        ns = '/teacher_controller'
        self.kp_h = rospy.get_param(f'{ns}/kp_h', 2.0)
        self.kd_h = rospy.get_param(f'{ns}/kd_h', 1.0)
        self.kp_z = rospy.get_param(f'{ns}/kp_z', 2.5)
        self.kd_z = rospy.get_param(f'{ns}/kd_z', 1.2)
        self.kyaw = rospy.get_param(f'{ns}/kyaw', 0.5)
        self.k_rep = rospy.get_param(f'{ns}/k_rep', 1.0)
        self.d_min = rospy.get_param(f'{ns}/d_min', 0.5)
        self.umax_xy = rospy.get_param(f'{ns}/umax_xy', 2.5)
        self.umax_z = rospy.get_param(f'{ns}/umax_z', 1.5)
        self.umax_yaw = rospy.get_param(f'{ns}/umax_yaw', 1.5)
        self.takeoff_duration = rospy.get_param(f'{ns}/takeoff_duration', 3.0)

        self.rate_hz = rospy.get_param('~rate', 10)
        self.num_drones = rospy.get_param('~num_drones', 4)
        self.startup_wait = rospy.get_param('~startup_wait', 5.0)
        self.publish_world_cmd = rospy.get_param('~publish_world_cmd', True)

        # 状态缓存
        self.odoms = [None] * self.num_drones
        self.p_des = [None] * self.num_drones
        self.v_des = [None] * self.num_drones

        rospy.loginfo(f'[teacher] init: N={self.num_drones}, kp_h={self.kp_h}, kd_h={self.kd_h}, '
                      f'kp_z={self.kp_z}, kd_z={self.kd_z}, publish_world_cmd={self.publish_world_cmd}')
        rospy.loginfo(f'[teacher] sleeping {self.startup_wait}s (wall time) to let 4-drone spawn complete...')
        time.sleep(self.startup_wait)
        rospy.loginfo('[teacher] setting up subscribers...')

        # 订阅 N 套 (用 _make_xxx_cb 闭包避免 lambda 陷阱)
        for i in range(self.num_drones):
            ns_d = f'/drone{i+1}'
            rospy.Subscriber(f'{ns_d}/ground_truth/state', Odometry, self._make_odom_cb(i))
            rospy.Subscriber(f'{ns_d}/p_des', PointStamped, self._make_pdes_cb(i))
            rospy.Subscriber(f'{ns_d}/v_des', Vector3Stamped, self._make_vdes_cb(i))

        # 发布 N 套
        self.pub_cmd_world = []
        self.pub_teacher = []
        for i in range(self.num_drones):
            ns_d = f'/drone{i+1}'
            self.pub_cmd_world.append(
                rospy.Publisher(f'{ns_d}/command/twist', TwistStamped, queue_size=10)
                if self.publish_world_cmd else None
            )
            self.pub_teacher.append(rospy.Publisher(f'{ns_d}/cmd_vel_teacher', Twist, queue_size=10))

        rospy.loginfo('[teacher] subscribers/publishers ready, starting control loop')
        self.t0 = rospy.Time.now().to_sec()

    def _make_odom_cb(self, idx):
        def cb(msg): self.odoms[idx] = msg
        return cb

    def _make_pdes_cb(self, idx):
        def cb(msg): self.p_des[idx] = np.array([msg.point.x, msg.point.y, msg.point.z])
        return cb

    def _make_vdes_cb(self, idx):
        def cb(msg): self.v_des[idx] = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
        return cb

    def compute_repulsion(self, i):
        """论文公式 3.11: f_rep,ij = k_rep · (p_i - p_j) / ||p_i - p_j||³"""
        if self.odoms[i] is None:
            return np.array([0.0, 0.0])
        p_i = np.array([self.odoms[i].pose.pose.position.x,
                        self.odoms[i].pose.pose.position.y])
        f = np.array([0.0, 0.0])
        for j in range(self.num_drones):
            if j == i or self.odoms[j] is None:
                continue
            p_j = np.array([self.odoms[j].pose.pose.position.x,
                            self.odoms[j].pose.pose.position.y])
            d_vec = p_i - p_j
            d = np.linalg.norm(d_vec)
            if 1e-6 < d < self.d_min:
                f += self.k_rep * d_vec / (d ** 3)
        return f

    def control_step(self, i, t_elapsed):
        if self.odoms[i] is None or self.p_des[i] is None or self.v_des[i] is None:
            return None
        odom = self.odoms[i]
        p = np.array([odom.pose.pose.position.x,
                      odom.pose.pose.position.y,
                      odom.pose.pose.position.z])
        v = np.array([odom.twist.twist.linear.x,
                      odom.twist.twist.linear.y,
                      odom.twist.twist.linear.z])
        omega_z = odom.twist.twist.angular.z

        # NaN/Inf 保护：odom 异常时停发指令，防止 NaN 传播
        if not np.isfinite(p).all() or not np.isfinite(v).all() or not np.isfinite(omega_z):
            rospy.logwarn_throttle(2.0, f'[teacher] drone{i+1} odom contains NaN/Inf, skipping step')
            return None

        # 起飞保护
        if t_elapsed < self.takeoff_duration:
            v_h_cmd = np.array([0.0, 0.0])
        else:
            e_h = self.p_des[i][:2] - p[:2]
            e_h_dot = self.v_des[i][:2] - v[:2]
            v_h_cmd = self.kp_h * e_h + self.kd_h * e_h_dot + self.compute_repulsion(i)

        e_z = self.p_des[i][2] - p[2]
        e_z_dot = self.v_des[i][2] - v[2]
        v_z_cmd = self.kp_z * e_z + self.kd_z * e_z_dot

        omega_yaw_cmd = -self.kyaw * omega_z

        vx = float(np.clip(v_h_cmd[0], -self.umax_xy, self.umax_xy))
        vy = float(np.clip(v_h_cmd[1], -self.umax_xy, self.umax_xy))
        vz = float(np.clip(v_z_cmd, -self.umax_z, self.umax_z))
        wz = float(np.clip(omega_yaw_cmd, -self.umax_yaw, self.umax_yaw))

        # 输出 NaN/Inf 保护：计算结果异常时停发
        if not all(np.isfinite([vx, vy, vz, wz])):
            rospy.logwarn_throttle(2.0, f'[teacher] drone{i+1} cmd_vel NaN/Inf, skipping')
            return None

        return (vx, vy, vz, wz)

    def run(self):
        r = rospy.Rate(self.rate_hz)
        log_period_ticks = max(1, int(self.rate_hz * 3))  # 每 3 秒打印
        tick = 0
        while not rospy.is_shutdown():
            t_elapsed = rospy.Time.now().to_sec() - self.t0
            tick += 1

            n_published = 0
            for i in range(self.num_drones):
                result = self.control_step(i, t_elapsed)
                if result is None:
                    continue
                v_x, v_y, v_z, w_z = result
                # Teacher PD is derived in world coordinates from world-frame errors.
                # Hector's /command/twist consumes world-frame TwistStamped directly,
                # while /cmd_vel would be interpreted in stabilized frame and rotated again.
                msg_world = TwistStamped()
                msg_world.header.stamp = rospy.Time.now()
                msg_world.header.frame_id = 'world'
                msg_world.twist.linear.x = v_x
                msg_world.twist.linear.y = v_y
                msg_world.twist.linear.z = v_z
                msg_world.twist.angular.z = w_z

                msg_teacher = Twist()
                msg_teacher.linear.x = v_x
                msg_teacher.linear.y = v_y
                msg_teacher.linear.z = v_z
                msg_teacher.angular.z = w_z

                if self.pub_cmd_world[i] is not None:
                    self.pub_cmd_world[i].publish(msg_world)
                self.pub_teacher[i].publish(msg_teacher)
                n_published += 1

            # 周期性 debug 日志
            if tick % log_period_ticks == 0:
                stats = []
                for i in range(self.num_drones):
                    o = 'o' if self.odoms[i] is not None else '-'
                    p = 'p' if self.p_des[i] is not None else '-'
                    v = 'v' if self.v_des[i] is not None else '-'
                    stats.append(f"d{i+1}:{o}{p}{v}")
                mode = 'world+label' if self.publish_world_cmd else 'label-only'
                rospy.loginfo(f'[teacher t={t_elapsed:.1f}s] data status: {" ".join(stats)}, '
                              f'outputs={mode}, published: {n_published}/{self.num_drones}')

            try:
                r.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                pass  # sim time 跳变时不要 crash


if __name__ == '__main__':
    try:
        TeacherController().run()
    except rospy.ROSInterruptException:
        pass
