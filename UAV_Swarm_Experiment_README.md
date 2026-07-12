# 无人机群编队端到端控制研究
## 完整实验复现 README
> — 含论文修改要点、仿真设计、训练流水线、审稿应对 —

版本：NaN-Bug待解决阶段  ·  2026-06

## 第 0 章  紧急状态与当前 P0 阻塞

> ⚠️  所有动态场景的 bag 数据均已损坏，需先解决 NaN Bug 才能继续复现

### 0.1 问题描述

hector_quadrotor 在持续动态飞行约 150–200 秒后触发 NaN 数值爆炸：

```text
propulsion model input contains **!?* Nan values!
drag model input contains **!?* Nan values!
```

NaN 一旦进入 Gazebo 物理积分器，飞机坐标每帧累积，最终到达 km 量级。已录制的 s2_circle bag 第一帧 drone1 = (3224, −7036, +2253) m，远超任何合理值。

### 0.2 受影响状态

| 场景 | 录制 bag | bag 状态 | 说明 |
| --- | --- | --- | --- |
| S1 悬停 | scene01_hover_4drones_seed42.bag  (16 MB) | ✅ 正常 | 静态悬停，z=1.80 m，xy ±0.5 m，可用 |
| S2 圆形 | s2_circle_4drones_seed42.bag      (24 MB) | ❌ 损坏 | 第一帧已在 3000+ m 外，NaN 已触发 |
| S2' 八字 | scene02p_lemni_4drones_seed42.bag (24 MB) | ❌ 损坏 | 位置爆炸至 x:-100~25 m 量级 |
| S3 重构 | scene03_reconfig_4drones_seed42.bag(32 MB) | ❌ 损坏 | drone3 飞到 (-50, -90) |
| S4 风扰 | scene04_wind_4drones_seed42.bag   (24 MB) | ❌ 损坏 | drone2 飞到 (1349, -1737, 95) |
| S5 长航时 | scene05_longtime_4drones_seed42.bag(48 MB) | ❌ 损坏 | drone4 飞到 (832, -2035, 6917) |

### 0.3 待决策：两条修复路径

| 路径 | 方案 | 预估时间 | 优点 | 风险 |
| --- | --- | --- | --- | --- |
| 路径 1 | 修复 hector_quadrotor NaN<br>• omega 降至 0.2 rad/s<br>• teacher 加 NaN/超界保护<br>• launch 后 10–15 s 立即录 bag | 1–2 天 | 改动小，代码结构不变 | NaN 只被延后，未根治，长航时仍可能复发 |
| 路径 2 | 切换到 RotorS（ETH Zurich）<br>• 更稳定，IEEE TRO/TAC 大量引用<br>• 原生支持多机 namespace | 约 1 周 | 从根本上解决数值不稳定 | 需重写 spawn_4drones、重新适配所有 Python 节点 |

决策原则： 如果论文计划投 IEEE TRO / T-AC 这类顶刊，强烈建议路径 2（RotorS 被审稿人更认可）。如果投会议或时间紧，路径 1 先跑出数据再说。

### 0.4 路径 1 具体操作（若选此路径）

修改 teacher_controller.py，在 control_step 开头加 NaN 保护：

```text
# 在 odom 回调里加检查
def is_valid_odom(self, odom):
    p = odom.pose.pose.position
    if any(np.isnan([p.x, p.y, p.z])): return False
    if any(np.abs([p.x, p.y, p.z]) > 500): return False  # 超 500m 视为爆炸
    return True
```

同时修改 scene02_circle_4drones.launch 里 omega 参数：

```text
<param name="omega" value="0.2"/>  <!-- 原为 0.4 -->
```

录 bag 时序：

1. roslaunch drone_sim scene02_circle_4drones.launch gui:=false
2. 等待约 10 秒（看 teacher 日志确认 cmd_vel published: 4/4）
3. 立即开始录制，不要等太久（NaN 在 150 s 左右触发）

```text
rosbag record -O ~/drone-formation-e2e/data/raw_bags/s2_circle_seed42.bag \
  --duration=90 \
  /drone{1..4}/p_des /drone{1..4}/v_des \
  /drone{1..4}/ground_truth/state \
  /drone{1..4}/cmd_vel /drone{1..4}/cmd_vel_teacher
```

## 第 1 章  论文核心理论与架构决策

### 1.1 方法定位：GCIL 而非纯 BC

本文方法的正确定位是 Goal-Conditioned Imitation Learning（目标条件模仿学习），而非传统 Behavioral Cloning（行为克隆）。这个区别至关重要：

- 纯 BC：Student 模仿 Teacher 的 (状态→动作) 映射，理论上无法超越 Teacher。
- GCIL：Student 输入比 Teacher 多出 p_des 和 v_des（期望位置/速度），即携带目标信息的额外 6 维。Teacher 做 PD 控制时看不到 p_des/v_des——这是设计上的非对称信息结构，理论上正当化了 Student 超越 Teacher。

引用依据：Codevilla et al., ICRA 2018；Ding et al., NeurIPS 2019。§4.1 必须重写强调 GCIL 框架。

### 1.2 网络架构

| 模块 | 规格 | 设计理由 |
| --- | --- | --- |
| 输入层 | 15D × W=20 时间窗口<br>= [p_des(3), v_des(3), p_actual(3), v_actual(3), e_p(3)] | GCIL 目标引导 + 实际状态 + 误差，三者缺一不可 |
| FA 特征注意力 | softmax 对 15 维加权 | 自适应放大 p_des 偏差等关键特征，压制传感器噪声 |
| BiLSTM × 2层 | 每层 64 隐层单元，tanh 激活 | 正向(过去→当前) + 反向(未来→当前)，捕获转向阶段信息 |
| TA 时序注意力 | softmax 对 W=20 时间步加权 | 聚焦轨迹转向关键时刻，减少历史误差干扰 |
| 全连接层 | 32 单元，relu | 特征融合 |
| 输出层 | 4D cmd_vel = [vx, vy, vz, ωz] | 速度级端到端，匹配 hector twist 接口 |

### 1.3 Teacher PD 控制器设计原则

Teacher 保持纯 PD（不加前馈、不加积分、不引入学习），原因：

- 保留 PD 稳态误差的理论可证性（Appendix A 需推导圆周轨迹下 PD 的稳态误差表达式）。
- 有了可证的 PD 稳态误差，BiLSTM-DA 通过隐式前馈超越 Teacher 才有意义。
- 审稿人常问"Teacher 够不够强？" → 回答：纯 PD 天然存在稳态误差，这是我们的理论瓶颈，BiLSTM-DA 用 GCIL 信息关闭这个缺口。

已确认的 PD 增益参数：

| 参数 | 值 | 物理含义 |
| --- | --- | --- |
| kp_h | 2.0 | 水平位置误差增益 |
| kd_h | 1.0 | 水平速度阻尼 |
| kp_z | 2.5 | 垂直位置误差增益 |
| kd_z | 1.2 | 垂直速度阻尼 |
| kyaw | 0.5 | 偏航角速度阻尼（自旋抑制） |
| k_rep | 1.0 | 碰撞排斥力系数 |
| d_min | 0.5 m | 碰撞排斥激活阈值 |
| umax_xy | 2.5 m/s | 水平速度指令饱和 |
| umax_z | 1.5 m/s | 垂直速度指令饱和 |

### 1.4 对比矩阵（7 个方法）

重要更正： 12D-LSTM 已从对比矩阵中删除。原始文章里"12D"实为 MPC 内部的小波+FFT 频域特征，属于 MPC 的内部设计，不是 LSTM 变体。错误保留会被审稿人立刻质疑。

| 方法 | 类别 | 核心特点 |
| --- | --- | --- |
| PID | 传统控制 | 基线，无学习 |
| 12D 优化 MPC | 模型预测控制（基线） | 内部用小波+FFT 频域特征（非 LSTM 变体），3D 误差 0.48 m（无风） |
| Teacher PD | 传统控制（数据生成器） | 本文 Teacher，PD+碰撞排斥，稳态误差 ~0.08 m |
| 9D-LSTM | 消融 A（特征） | 无 p_des/v_des 目标引导，易漂移 |
| 15D-LSTM（单向） | 消融 B（结构） | 有目标引导，但无 BiLSTM |
| 15D-BiLSTM（无DA） | 消融 C（注意力） | 有 BiLSTM，无双注意力 |
| 15D-BiLSTM-DA | 本文方法（Proposed） | 完整 GCIL + BiLSTM + 双注意力，RMSE 0.03 m（占位值） |

### 1.5 消融设计（3 轴）

| 消融轴 | 对比组 A | 对比组 B | 证明什么 |
| --- | --- | --- | --- |
| 特征消融 | 9D-LSTM（无 p_des/v_des） | 15D-LSTM | 目标条件输入的必要性 |
| 结构消融 | 15D-LSTM（单向） | 15D-BiLSTM（无 DA） | BiLSTM 对双向时序的价值 |
| 注意力消融 | 15D-BiLSTM（无 DA） | 15D-BiLSTM-DA（完整） | 双注意力的增益 |

每个方法 5 个随机种子；主要评估指标：3D RMSE 位置误差。

## 第 2 章  仿真环境与已验证的状态

### 2.1 环境栈

| 层次 | 软件 | 版本/来源 | 状态 |
| --- | --- | --- | --- |
| 操作系统 | WSL2 Ubuntu | 20.04 | ✅ 运行正常 |
| ROS | Noetic | apt 安装 | ✅ 运行正常 |
| 物理引擎 | Gazebo | 11.15.1 | ✅ 运行正常 |
| 四旋翼仿真包 | hector_quadrotor | RAFALAMAO/hector-quadrotor-noetic fork，源码编译于 ~/catkin_ws | ⚠️ 存在 NaN 数值不稳定问题 |
| 主机硬件 | Intel Iris Xe | Win11 + WSL2，纯 CPU，无 CUDA | ✅ 训练须用 CPU-only 模式 |

### 2.2 工作区结构

```text
~/catkin_ws/                        # hector_quadrotor 源码编译工作区
~/drone-formation-e2e/
  ros_ws/src/drone_sim/             # ROS 包主目录
    launch/                         # 所有场景 launch 文件
      spawn_4drones.launch          # 4 机 spawn
      scene01_hover_4drones.launch  # S1 悬停
      scene02_circle_4drones.launch # S2 圆形（主要测试场景）
      scene02p_lemni_4drones.launch # S2' 八字
      scene03_reconfig_4drones.launch # S3 重构
      scene04_wind_4drones.launch   # S4 风扰
      scene05_longtime_4drones.launch # S5 长航时
    scripts/
      teacher_controller.py         # 4 机 PD Teacher
      scene_driver.py               # 场景轨迹发布器
      enable_motors_delayed.py      # 延时激活电机
      wind_driver.py                # S4 风场扰动
      collect_seed42.sh             # 批量录制脚本
      analyze_bags.py               # 离线 bag 分析和可视化
    config/teacher_params.yaml
  data/raw_bags/                    # rosbag 数据
  results/figures/                  # 生成图片
```

.bashrc source 顺序（严格遵守）：

```text
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
source ~/drone-formation-e2e/ros_ws/devel/setup.bash
```

### 2.3 已验证完成的里程碑

| 里程碑 | 验证方式 | 结果 |
| --- | --- | --- |
| 4 机 spawn + namespace 隔离 | rostopic list | ✅ 每机 ~37 个 topic，完全独立 |
| 电机激活 | rosservice call /droneN/engage | ✅ 服务名为 engage（std_srvs/Empty） |
| cmd_vel 频率 | rostopic hz /drone1/cmd_vel | ✅ 稳定 10.000 Hz |
| S1 悬停精度 | rostopic echo ground_truth/state | ✅ z=1.799 m，xy ±0.1 m |
| S2 圆形编队（同步采样） | sync_check.py | ✅ 编队中心距原点 2.956 m（期望 3.0），各机到重心 0.707 m |
| 5 个场景 launch | roslaunch 各场景 | ✅ 均成功启动，teacher 稳定输出 cmd_vel 4/4 |
| teacher 状态日志 | launch 日志 | ✅ 持续 500+ s 输出 d1:opv d2:opv d3:opv d4:opv |

### 2.4 关键 ROS Topic 说明

| Topic | 类型 | 发布者 | 作用 |
| --- | --- | --- | --- |
| /droneN/p_des | geometry_msgs/PointStamped | scene_driver | 期望位置（构成15D的 xd,yd,zd） |
| /droneN/v_des | geometry_msgs/Vector3Stamped | scene_driver | 期望速度（构成15D的 vxd,vyd,vzd） |
| /droneN/ground_truth/state | nav_msgs/Odometry | Gazebo 插件 | 实际状态（位置+速度），83 Hz |
| /droneN/cmd_vel | geometry_msgs/Twist | teacher_controller | 速度指令，发给 hector twist controller |
| /droneN/cmd_vel_teacher | geometry_msgs/Twist | teacher_controller | 示教标签，供 LSTM 监督学习 |

## 第 3 章  5 个实验场景设计

### 3.1 编队几何（所有场景共用）

4 机矩形编队，以编队中心 c(t) 为基准，各机相对偏移固定：

| 无人机 | x 偏移 | y 偏移 | spawn 初始位置 |
| --- | --- | --- | --- |
| drone1 | +0.5 m | +0.5 m | (+0.5, +0.5, 0.3) |
| drone2 | −0.5 m | +0.5 m | (−0.5, +0.5, 0.3) |
| drone3 | −0.5 m | −0.5 m | (−0.5, −0.5, 0.3) |
| drone4 | +0.5 m | −0.5 m | (+0.5, −0.5, 0.3) |

编队间距 1.0 m（相邻机），对角线 1.414 m，均远超 d_min=0.5 m，碰撞排斥力不会触发。

### 3.2 场景列表

| 场景 | Launch 文件 | 时长 | 中心轨迹 c(t) | 论文作用 |
| --- | --- | --- | --- | --- |
| S1 悬停 | scene01_hover | 60 s | c=(0,0)，固定 | 基础悬停稳定性、编队保持基线 |
| S2 圆形 | scene02_circle | 90 s | c=(R·cosωt, R·sinωt), R=3 m, ω=0.4 rad/s | Teacher vs LSTM 主要对比场景 |
| S2' 八字 | scene02p_lemni | 90 s | c=(R·sinωt, R/2·sin2ωt) | 验证复杂曲率轨迹跟踪 |
| S3 重构 | scene03_reconfig | 120 s | c 仍走圆周；编队偏移在 t=40s/80s 切换 | 验证编队动态重构能力 |
| S4 风扰 | scene04_wind | 90 s | c 走圆周 + 叠加风扰 F=k·v_wind | 验证抗干扰鲁棒性 |
| S5 长航时 | scene05_longtime | 180 s | = S2 圆形，延长 | 验证无漂移长期稳定 |

### 3.3 S3 队形切换时序

| 时间段 | 队形 | 各机偏移（相对中心） |
| --- | --- | --- |
| 0–40 s | 矩形 | (±0.5, ±0.5) |
| 40–80 s | 菱形 | (±0.707, 0) 和 (0, ±0.707) |
| 80–120 s | 三角形+中心机 | drone1=(0,0.6), drone2=(-0.6,-0.36), drone3=(0.6,-0.36), drone4=(0,0) |

注意： 切换是硬切换，可能引起短暂扰动。如果飞机过于靠近（<d_min），碰撞排斥力会介入辅助安全分离。

### 3.4 S4 风扰模型参数

风场方程：v_wind_x(t) = v_mean + A·sin(2πft) + n_x(t)，v_wind_y 同式加相位差 φ=π/2，力 F = k_wind · v_wind。

| 参数 | 值 | 来源 |
| --- | --- | --- |
| v_mean_x | 2.0 m/s | 原始文章 §5.1.2 明确给出 |
| A（阵风振幅） | 1.0 m/s | 原始文章 §5.1.2 明确给出 |
| f（阵风频率） | 0.5 Hz | 顶刊默认（2 s 周期，符合自然低频风） |
| σ（噪声标准差） | 0.3 m/s | 顶刊默认（v_mean 的 15%） |
| k_wind | 0.3 | 顶刊默认 |
| φ（x/y 相位差） | π/2 | 模拟多向湍流 |

## 第 4 章  数据采集流水线

### 4.1 目标：5 场景 × 5 种子 = 25 个 Bag（待完成）

| 种子 | 状态 | 说明 |
| --- | --- | --- |
| seed=42 | ❌ 动态场景全损坏 | 只有 S1 hover 可用，其余 NaN 污染 |
| seed=123,256,512,1024 | ❌ 未录制 | 等待 NaN Bug 修复后批量录制 |

### 4.2 录制 Bag 的关键 Topic 清单

每次录制必须包含以下 20 个 topic（4 机 × 5 种）：

| Topic 模式 | 类型 | 频率 | 用途 |
| --- | --- | --- | --- |
| /droneN/ground_truth/state | Odometry | 83 Hz | 实际位置/速度 → 15D 特征中 p_actual, v_actual, e_p |
| /droneN/p_des | PointStamped | 10 Hz | 期望位置 → 15D 特征中 p_des |
| /droneN/v_des | Vector3Stamped | 10 Hz | 期望速度 → 15D 特征中 v_des |
| /droneN/cmd_vel | Twist | 10 Hz | 实际控制指令（验证用） |
| /droneN/cmd_vel_teacher | Twist | 10 Hz | 示教标签 u_T → LSTM 训练目标 |

### 4.3 批量录制脚本逻辑

collect_seed42.sh 已实现，路径：~/drone-formation-e2e/scripts/collect_seed42.sh。

关键时序（每个场景）：

1. killall 残留进程 → 等 4 s
2. roslaunch 场景 launch（后台运行）→ 等 25 s（等待 4 机 spawn + teacher 稳定）
3. rosbag record 开始（见上方 topic 清单），录制指定时长
4. 检查 bag 文件大小，确认正常后继续下一场景

⚠️ NaN 修复前： 收 wait 时长改为 5 s，总录制窗口需在 NaN 触发（约 150 s）之前完成。S2 圆形（90 s）最安全，S5 长航时（180 s）最危险。

### 4.4 Bag 质量验证脚本

使用 diagnose_bags.py 打印每架机的前 3 帧和后 3 帧位置：

```text
python3 ~/drone-formation-e2e/scripts/diagnose_bags.py
```

判断标准：

- 正常：所有坐标在 ±20 m 范围内，z 在 1.5–2.1 m
- 损坏：任何坐标超过 100 m，或 z 超过 10 m → NaN 已触发，此 bag 作废

## 第 5 章  MATLAB 数据预处理 Pipeline

### 5.1 15D 特征向量构造

从 bag 的 4 个 topic 提取 15 维状态向量（对齐到 10 Hz 时间基准）：

| 维度 | 来源 Topic | 字段 | 含义 |
| --- | --- | --- | --- |
| [1-3] p_des | /droneN/p_des | point.x/y/z | 期望位置（GCIL 目标信息） |
| [4-6] v_des | /droneN/v_des | vector.x/y/z | 期望速度（GCIL 目标信息） |
| [7-9] p_actual | /droneN/ground_truth/state | pose.pose.position.x/y/z | 实际位置 |
| [10-12] v_actual | /droneN/ground_truth/state | twist.twist.linear.x/y/z | 实际速度 |
| [13-15] e_p | 计算：p_des − p_actual | — | 位置误差（隐式前馈信息） |

输出标签（4D）：cmd_vel = [vx, vy, vz, ωz]，来自 /droneN/cmd_vel_teacher。

### 5.2 滑动窗口采样

参数：采样周期 Δt=0.1 s，窗口长度 W=20（对应 2 秒历史）。

对每个时刻 k，构造输入序列：

```text
X(k) = [s(k-W+1), s(k-W+2), ..., s(k)]  ∈ R^{15×W}
Y(k) = u_T(k) = [vx_T, vy_T, vz_T, ωz_T]  ∈ R^4
```

9D 实验时，输入向量截取 [7-15] 维（实际位置 + 速度 + 误差），去掉 [1-6] 的目标信息。

### 5.3 数据归一化

对每个维度单独做零均值单位方差归一化，在所有样本上统计：

```text
μ_X(i) = mean(X[:,i,:])   σ_X(i) = std(X[:,i,:]) + 1e-8
X_norm = (X - μ_X) / σ_X
U_norm = (U - μ_U) / σ_U
```

关键： 归一化参数 (μ_X, σ_X, μ_U, σ_U) 必须与模型权重一同保存到 .mat 文件。在线部署时，实时状态需要用训练阶段的同一组参数归一化，否则模型推理会有系统偏差。

### 5.4 数据集划分与保存

按 7:2:1 划分训练集、验证集、测试集，保证各场景样本均匀分布到三个集合。保存格式：

```text
save('samples_15d_seed42.mat', 'X_train', 'Y_train', 'X_val', 'Y_val',
     'X_test', 'Y_test', 'mu_X', 'sig_X', 'mu_U', 'sig_U');
```

每个场景至少录制 3–5 分钟数据（S1:60 s, S2:90 s, S3:120 s, S4:90 s, S5:180 s），全部合并再划分，以保证模型跨场景泛化。

## 第 6 章  LSTM 模型训练

### 6.1 训练配置

| 超参数 | 值 | 理由 |
| --- | --- | --- |
| 优化器 | Adam | 自适应学习率，BiLSTM 常用 |
| 初始学习率 | 1e-3 | 收敛快 |
| 学习率衰减 | 每 10 epoch × 0.9 | 避免过拟合 |
| Batch size | 64 | 内存与收敛折中 |
| 最大 epoch | 50 | 早停实际会更早终止 |
| Early stopping patience | 5 | 验证集 5 epoch 无下降即停 |
| Dropout | 0.2（LSTM 输出后） | 正则化 |

### 6.2 损失函数

4 通道加权 MSE（高度通道权重更大，保证 z 方向控制优先级）：

```text
L = α_xy·(‖vx_pred−vx_T‖² + ‖vy_pred−vy_T‖²)
  + α_z ·‖vz_pred−vz_T‖²
  + α_r ·‖ωz_pred−ωz_T‖²
其中 α_z > α_xy ≈ α_r
```

### 6.3 4 个 LSTM 变体的训练顺序

建议从简单到复杂，每步验证 RMSE 后再往下一步：

| 顺序 | 模型 | 输入维度 | 估计训练时间（CPU-only） |
| --- | --- | --- | --- |
| 1 | 9D-LSTM（单向） | 9D × W=20 | ~1 h / seed |
| 2 | 15D-LSTM（单向） | 15D × W=20 | ~1.5 h / seed |
| 3 | 15D-BiLSTM（无 DA） | 15D × W=20 | ~2 h / seed |
| 4 | 15D-BiLSTM-DA（完整） | 15D × W=20 | ~2.5 h / seed |

全部 4 模型 × 5 seeds = 20 次训练，总计约 140–160 小时 CPU 时间。建议挂后台跑（screen 或 tmux）。

### 6.4 模型打包保存

```text
save('model_package_15d_biDA_seed42.mat',
     'net',      % BiLSTM-DA 网络对象
     'mu_X', 'sig_X', 'mu_U', 'sig_U',  % 归一化参数（在线部署必须）
     'W',        % 窗口长度 = 20
     'D');       % 特征维度 = 15
```

## 第 7 章  评估指标与图表生成

### 7.1 主要评估指标

| 指标 | 公式 | 含义 |
| --- | --- | --- |
| 3D RMSE | √(Σ‖p_actual−p_des‖²/N) | 主要排名指标，越小越好 |
| 高度 RMSE | √(Σ(z−z_des)²/N) | 垂直通道单独评估 |
| 60s 后漂移率 | (RMSE_after60 − RMSE_before60)/RMSE_before60 | 衡量长期稳定性 |
| 最大 3D 误差 | max‖p_actual−p_des‖₂ | 最差情况 |
| cmd_vel 波动 | std(u_pred) | 衡量控制平滑性 |

### 7.2 论文所需图表清单（IEEE 顶刊标准）

| 图序 | 内容 | 对应论文章节 |
| --- | --- | --- |
| 图 5.1 | S2 圆形：XY 轨迹对比（期望/Teacher/9D-LSTM/15D-BiLSTM-DA） | §5.2.1 |
| 图 5.2 | 3D 位置误差时序（包含 60s 漂移检查线） | §5.2.2 |
| 图 5.3 | 高度 z(t) 对比（4 个控制器） | §5.2.3 |
| 图 5.4 | cmd_vel 4 通道波形对比 | §5.2.5 |
| 图 5.5 | 误差统计柱状图（RMSE/MAE/max，各方法对比） | §5.2.6 |
| 图 5.6 | S3 重构：编队切换时的 3D 误差时序 | §5.2 重构小节 |
| 图 5.7 | S4 风扰：XY 轨迹 + 误差对比 | §5.2 风扰小节 |
| 图 5.8 | S5 长航时：180 s 完整误差曲线（无漂移验证） | §5.2 长航时小节 |
| 图 5.9 | 综合 6 场景拼图（overview） | §5.3 综合对比 |

图表样式已在 analyze_bags.py 实现（Times New Roman，4 边框，内向刻度，色盲友好配色，300 dpi PNG + PDF）。

## 第 8 章  论文修改清单（相对原始文章）

### 8.1 必须修改的内容

| 位置 | 问题 | 修改方案 | 优先级 |
| --- | --- | --- | --- |
| §4.1 | 定位为纯 BC，无 GCIL 理论支撑 | 重写为 GCIL 框架，引用 Codevilla 2018、Ding 2019 | P0 |
| 对比矩阵 | 包含 12D-LSTM（不应存在） | 删除 12D-LSTM，保留 12D-MPC 作为独立基线 | P0 |
| 全文 X.XX | 所有数值为占位符 | 等复现完成后填入真实实验数据 | P0 |
| Appendix A | 缺少 PD 稳态误差推导 | 新增：圆周轨迹下纯 PD 控制器的稳态误差理论推导 | P1 |
| §5.2.6 | Teacher RMSE=9.134m 与全文数据不一致 | 重新确认：此数值是 Teacher 相对期望轨迹（非相对 LSTM）的误差，需注释清楚 | P1 |

### 8.2 审稿人常见问题应对

| 审稿人问题 | 正确回答 |
| --- | --- |
| 为什么 Student 能超越 Teacher？（最常见！） | GCIL 设计：Student 输入包含 p_des/v_des（任务目标），Teacher 做 PD 控制时不知道目标。信息非对称是架构设计，不是数据泄露。 |
| Teacher 够强吗？PD 是不是太弱的 Baseline？ | 纯 PD 对于圆周轨迹存在理论可证的稳态误差（见 Appendix A）。这个理论瓶颈是我们方法要克服的目标，不是缺陷。 |
| 为什么不用 12D-LSTM 做消融？ | 原始文章里"12D"指 MPC 的内部频域特征（小波+FFT），专属于 MPC 优化框架，不是 LSTM 的特征变体，两个技术体系不可混用。 |
| 5 个种子够吗？ | 5 seeds × 5 场景 = 25 个独立实验，统计意义充分。主要评估指标用均值±方差报告。 |

## 第 9 章  完整复现步骤（按顺序执行）

> 按顺序执行，每步验证通过再进行下一步，禁止跳步！

### STEP 1：解决 NaN Bug（路径 1 或 2）（P0 阻塞）

- 路径 1：修改 teacher_controller.py 加 NaN 保护 + 降低 omega → 重新编译 → 测试
- 路径 2：安装 RotorS → 重写 spawn_4drones.launch → 适配 Python 节点 → 测试
- 验证：启动 S2 圆形，运行 120 s，sync_check.py 显示编队误差 <1%

### STEP 2：重新录制 5 场景 × 1 种子（seed=42）（约 20 分钟）

- 确认每个 bag 的消息数：ground_truth/state = 采样时长(s) × 83 条
- 运行 diagnose_bags.py，确认所有坐标在 ±20 m 内
- 用 analyze_bags.py 生成轨迹图，目视验证轨迹合理

### STEP 3：扩展到 5 个种子（约 1.5 小时）

- 修改 collect_seed42.sh 生成 collect_all_seeds.sh
- 5 个种子：42, 123, 256, 512, 1024
- 共 25 个 bag，验证方法同 STEP 2

### STEP 4：MATLAB bag → samples.mat 预处理（约 30 分钟）

- 实现 bag_to_mat.m（或 Python 版），提取 15D 特征 + 4D 标签
- 构造滑动窗口（W=20），归一化，7:2:1 划分
- 验证：打印训练集大小，检查有无 NaN/Inf

### STEP 5：训练 4 个 LSTM 变体 × 5 种子（约 1 周（CPU））

- 按顺序：9D → 15D-LSTM → 15D-BiLSTM → 15D-BiLSTM-DA
- 每次训练记录 val_loss 曲线，验证早停工作正常
- 保存 model_package_*.mat（含归一化参数）

### STEP 6：离线评估 + 生成图表（约 2 小时）

- 在测试集上计算各模型的 3D RMSE / 高度 RMSE / 漂移率
- 生成第 7 章所有图表（用 analyze_bags.py 或 MATLAB）
- 对比原始文章基线数值（Teacher 9.134m，LSTM 2.154m），记录差异和原因

### STEP 7：填写论文数值 + 完善修改（约 1 天）

- 用真实实验数据替换所有 X.XX 占位符
- 完成 §4.1 GCIL 重写、Appendix A PD 误差推导
- 最终送审前检查：12D-LSTM 已删除？数据一致性？图表标注？

## 第 10 章  原始文章关键数据（复现参考值）

以下数值来自原始文章，用于与复现结果对比。所有复现数值均待替换。

| 来源章节 | 数据 | 用途 |
| --- | --- | --- |
| §5.2.6 | LSTM RMSE = 2.154 m（相对期望轨迹）<br>Teacher RMSE = 9.134 m（相对期望轨迹） | 主要对比指标基线（GCIL 解释"Student 超 Teacher"的核心证据） |
| §5.2.2 | Teacher 相对 LSTM 的 RMSE ≈ 6.56 m（0–60 s） | 注意：这是两个控制器之间的相对误差，不是相对期望轨迹 |
| Table 5.4 | 12D 优化 MPC 3D 误差 = 0.48 m（无风） | MPC 最佳基线，LSTM 需超越此值才有说服力 |
| §3.5 | 12D 增强模型 = 小波+FFT 融合 → MPC 内部，非 LSTM | 重要澄清：删除 12D-LSTM 的理由 |
| §5.3.4 | 9D 普通 LSTM 整体 RMSE = 1.57 m | 消融 A 的参考值 |
| §5.3.4 | 15D 防漂移 LSTM RMSE = 0.30 m | 消融 C 完整模型的参考值 |

> 注意：X.XX 表示论文初稿占位符，等复现完成后填入真实数值。0.08m/2.5m/0.03m 均为预期目标，非已验证结果。

## 附录  常用命令速查

### A. 环境启动

```text
# source 顺序不可颠倒
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
source ~/drone-formation-e2e/ros_ws/devel/setup.bash
```

```text
# 启动 S2 圆形场景（最常用）
roslaunch drone_sim scene02_circle_4drones.launch gui:=false
```

### B. 实时验证

```text
# cmd_vel 频率
rostopic hz /drone1/cmd_vel

# 4 机位置（注意：for 循环各条之间有 0.5–1 s 延迟，结果不同步）
for i in 1 2 3 4; do echo "drone$i:"; rostopic echo /drone$i/ground_truth/state/pose/pose/position -n 1; done

# 同步采样（推荐）
python3 /tmp/sync_check.py   # 见之前准备好的脚本
```

### C. Bag 操作

```text
# 录制 S2 圆形 90 s
rosbag record -O ~/drone-formation-e2e/data/raw_bags/s2_circle.bag \
  --duration=90 \
  /drone{1..4}/p_des /drone{1..4}/v_des \
  /drone{1..4}/ground_truth/state \
  /drone{1..4}/cmd_vel /drone{1..4}/cmd_vel_teacher

# 查看 bag 信息
rosbag info ~/drone-formation-e2e/data/raw_bags/s2_circle.bag

# 诊断 bag 数据
python3 ~/drone-formation-e2e/scripts/diagnose_bags.py

# 生成论文图表
python3 ~/drone-formation-e2e/scripts/analyze_bags.py
```

### D. 编译 ROS 工作区

```text
cd ~/drone-formation-e2e/ros_ws && catkin_make && source devel/setup.bash

# 仅修改 Python 脚本时不需要重新编译，直接 chmod +x 即可
chmod +x ~/drone-formation-e2e/ros_ws/src/drone_sim/scripts/*.py
```

### E. 进程清理

```text
killall -9 gzserver gzclient rosmaster roscore rosout 2>/dev/null
pkill -9 -f "scene_driver|teacher_controller|enable_motors|wind_driver|rosbag" 2>/dev/null
sleep 4  # 等 Gazebo 完全退出
```
