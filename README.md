# 无人机群编队端到端控制实验复现 README

本仓库用于归档、修改和复现实验相关材料，主题为多无人机编队端到端控制。当前重点是围绕 `hector_quadrotor` 仿真、Teacher PD 控制器、目标条件模仿学习（GCIL）以及 LSTM/BiLSTM/双注意力模型完成数据采集、训练、评估和论文图表生成。

> 当前状态：所有动态场景的历史 bag 数据存在数值爆炸问题，必须优先解决 NaN Bug 后再继续完整复现。

## 1. 当前 P0 阻塞：NaN Bug

### 1.1 问题描述

`hector_quadrotor` 在持续动态飞行约 150-200 秒后会触发数值异常：

```text
propulsion model input contains **!?* Nan values!
drag model input contains **!?* Nan values!
```

NaN 一旦进入 Gazebo 物理积分器，飞机坐标会逐帧累积并最终达到 km 量级。已录制的 `s2_circle` bag 第一帧中，`drone1 = (3224, -7036, +2253) m`，远超合理范围。

### 1.2 受影响数据

| 场景 | 录制 bag | 状态 | 说明 |
| --- | --- | --- | --- |
| S1 悬停 | `scene01_hover_4drones_seed42.bag` | 正常 | 静态悬停，`z=1.80 m`，xy 约 `±0.5 m` |
| S2 圆形 | `s2_circle_4drones_seed42.bag` | 损坏 | 第一帧已在 3000 m 外，NaN 已触发 |
| S2' 八字 | `scene02p_lemni_4drones_seed42.bag` | 损坏 | 位置爆炸到 `x:-100~25 m` 量级 |
| S3 重构 | `scene03_reconfig_4drones_seed42.bag` | 损坏 | `drone3` 飞到 `(-50, -90)` |
| S4 风扰 | `scene04_wind_4drones_seed42.bag` | 损坏 | `drone2` 飞到 `(1349, -1737, 95)` |
| S5 长航时 | `scene05_longtime_4drones_seed42.bag` | 损坏 | `drone4` 飞到 `(832, -2035, 6917)` |

### 1.3 两条修复路径

| 路径 | 方案 | 优点 | 风险 |
| --- | --- | --- | --- |
| 路径 1 | 修复 `hector_quadrotor` NaN：降低 `omega`、给 Teacher 加 NaN/越界保护、启动后 10-15 s 立即录 bag | 改动小，代码结构基本不变 | 可能只是延后 NaN，而不是根治 |
| 路径 2 | 切换到 RotorS（ETH Zurich），重新适配多机 namespace 和 Python 节点 | 从根上解决数值稳定性问题，审稿认可度更高 | 需要重写 spawn、launch 和节点适配 |

建议：如果目标是 IEEE TRO/T-AC 等顶刊，优先考虑路径 2；如果时间紧，先走路径 1 快速产出可用数据。

### 1.4 路径 1 操作要点

在 `teacher_controller.py` 的 odom 回调或 `control_step` 前增加 NaN/越界保护：

```python
def is_valid_odom(self, odom):
    p = odom.pose.pose.position
    if any(np.isnan([p.x, p.y, p.z])):
        return False
    if any(np.abs([p.x, p.y, p.z]) > 500):
        return False
    return True
```

同时将 `scene02_circle_4drones.launch` 中的角速度降低：

```xml
<param name="omega" value="0.2"/>
```

录制 bag 时不要长时间等待，启动稳定后尽快录制：

```bash
roslaunch drone_sim scene02_circle_4drones.launch gui:=false

rosbag record -O ~/drone-formation-e2e/data/raw_bags/s2_circle_seed42.bag \
  --duration=90 \
  /drone{1..4}/p_des /drone{1..4}/v_des \
  /drone{1..4}/ground_truth/state \
  /drone{1..4}/cmd_vel /drone{1..4}/cmd_vel_teacher
```

## 2. 方法定位与论文核心决策

### 2.1 方法定位：GCIL 而非纯 BC

本文方法应定位为 Goal-Conditioned Imitation Learning（目标条件模仿学习，GCIL），不是传统 Behavioral Cloning（BC）。

| 方法 | 输入信息 | 理论含义 |
| --- | --- | --- |
| 纯 BC | 只模仿 Teacher 的状态到动作映射 | 理论上难以超过 Teacher |
| GCIL | Student 额外输入 `p_des` 和 `v_des` | Student 拥有 Teacher 不直接使用的目标信息，具备超过 Teacher 的合理性 |

关键解释：Student 输入比 Teacher 多出期望位置和期望速度，形成非对称信息结构。这不是数据泄露，而是目标条件控制设计。

### 2.2 网络架构

| 模块 | 规格 | 设计理由 |
| --- | --- | --- |
| 输入层 | `15D x W=20`，包含 `p_des`、`v_des`、`p_actual`、`v_actual`、`e_p` | 同时提供目标、实际状态和误差 |
| FA 特征注意力 | 对 15 维特征做 softmax 加权 | 放大关键误差特征，抑制噪声 |
| BiLSTM x 2 | 每层 64 隐藏单元，`tanh` 激活 | 捕获双向时序信息 |
| TA 时序注意力 | 对 `W=20` 时间步加权 | 聚焦轨迹转向等关键时刻 |
| 全连接层 | 32 单元，`relu` | 融合高级时序特征 |
| 输出层 | 4D `cmd_vel = [vx, vy, vz, wz]` | 匹配 hector twist 控制接口 |

### 2.3 Teacher PD 控制器

Teacher 保持纯 PD，不加入前馈、积分或学习项。这样可以保留可证明的稳态误差，并让 BiLSTM-DA 通过 GCIL 信息结构补足 Teacher 的不足。

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `kp_h` | 2.0 | 水平位置误差增益 |
| `kd_h` | 1.0 | 水平速度阻尼 |
| `kp_z` | 2.5 | 垂直位置误差增益 |
| `kd_z` | 1.2 | 垂直速度阻尼 |
| `kyaw` | 0.5 | 偏航角速度阻尼 |
| `k_rep` | 1.0 | 碰撞排斥力系数 |
| `d_min` | 0.5 m | 碰撞排斥激活阈值 |
| `umax_xy` | 2.5 m/s | 水平速度饱和 |
| `umax_z` | 1.5 m/s | 垂直速度饱和 |

### 2.4 对比矩阵

| 方法 | 类别 | 说明 |
| --- | --- | --- |
| PID | 传统控制 | 无学习基线 |
| 12D 优化 MPC | 模型预测控制 | 内部使用小波/FFT 频域特征，不是 LSTM 变体 |
| Teacher PD | 数据生成器 | PD + 碰撞排斥 |
| 9D-LSTM | 特征消融 | 去掉 `p_des/v_des` 目标引导 |
| 15D-LSTM | 结构消融 | 有目标信息，但无双向结构 |
| 15D-BiLSTM | 注意力消融 | 有 BiLSTM，无双注意力 |
| 15D-BiLSTM-DA | Proposed | 完整 GCIL + BiLSTM + 双注意力 |

重要修正：不要保留 “12D-LSTM”。原始文章中的 12D 是 MPC 内部特征设计，不是 LSTM 变体。

## 3. 仿真环境与工作区

### 3.1 环境栈

| 层次 | 软件 | 版本/来源 | 状态 |
| --- | --- | --- | --- |
| 操作系统 | WSL2 Ubuntu | 20.04 | 正常 |
| ROS | Noetic | apt 安装 | 正常 |
| 物理引擎 | Gazebo | 11.15.1 | 正常 |
| 四旋翼仿真包 | hector_quadrotor | `RAFALAMAO/hector-quadrotor-noetic` fork | 存在 NaN 不稳定 |
| 主机硬件 | Intel Iris Xe | Win11 + WSL2，CPU-only | 训练需 CPU 模式 |

### 3.2 工作区结构

```text
~/catkin_ws/
  hector_quadrotor 源码编译工作区

~/drone-formation-e2e/
  ros_ws/src/drone_sim/
    launch/
      spawn_4drones.launch
      scene01_hover_4drones.launch
      scene02_circle_4drones.launch
      scene02p_lemni_4drones.launch
      scene03_reconfig_4drones.launch
      scene04_wind_4drones.launch
      scene05_longtime_4drones.launch
    scripts/
      teacher_controller.py
      scene_driver.py
      wind_driver.py
  data/raw_bags/
  scripts/
    collect_seed42.sh
    diagnose_bags.py
    analyze_bags.py
```

### 3.3 关键 Topic

| Topic | 类型 | 用途 |
| --- | --- | --- |
| `/droneN/p_des` | 期望位置 | GCIL 目标输入 |
| `/droneN/v_des` | 期望速度 | GCIL 目标输入 |
| `/droneN/ground_truth/state` | 实际状态 | 位置、速度、误差计算 |
| `/droneN/cmd_vel` | Student/控制输出 | 在线控制接口 |
| `/droneN/cmd_vel_teacher` | Teacher 标签 | 训练监督信号 |

## 4. 实验场景设计

### 4.1 编队几何

4 机矩形编队以编队中心 `c(t)` 为基准，各机固定偏移：

| 无人机 | 相对偏移 |
| --- | --- |
| `drone1` | `(+0.5, +0.5, 0)` |
| `drone2` | `(+0.5, -0.5, 0)` |
| `drone3` | `(-0.5, +0.5, 0)` |
| `drone4` | `(-0.5, -0.5, 0)` |

相邻机间距 1.0 m，对角线 1.414 m，均大于 `d_min=0.5 m`。

### 4.2 场景列表

| 场景 | 内容 | 时长建议 | 目标 |
| --- | --- | --- | --- |
| S1 | 悬停 | 60 s | 验证静态稳定性 |
| S2 | 圆形轨迹 | 90 s | 主测试场景 |
| S2' | 八字轨迹 | 90 s | 曲率变化和泛化 |
| S3 | 编队重构 | 120 s | 队形切换稳定性 |
| S4 | 风扰 | 90 s | 抗扰能力 |
| S5 | 长航时 | 180 s | 漂移和长期稳定性 |

### 4.3 风扰模型

风场可按如下形式建模：

```text
v_wind_x(t) = v_mean + A * sin(2πft) + n_x(t)
v_wind_y(t) = v_mean + A * sin(2πft + φ) + n_y(t), φ = π/2
F = k_wind * v_wind
```

## 5. 数据采集流水线

目标是完成 `5 场景 x 5 种子 = 25 个 bag`。

推荐种子：

```text
42, 123, 256, 512, 1024
```

每次录制必须包含 20 个 topic：

```bash
/drone{1..4}/p_des
/drone{1..4}/v_des
/drone{1..4}/ground_truth/state
/drone{1..4}/cmd_vel
/drone{1..4}/cmd_vel_teacher
```

批量采集逻辑：

1. 清理 Gazebo、ROS、rosbag 残留进程。
2. 启动对应场景 launch。
3. 等待 10-15 s，确认 4 架无人机均有 `cmd_vel`。
4. 按场景时长录制 bag。
5. 运行 `diagnose_bags.py` 验证坐标范围和消息数量。

质量判断标准：

| 检查项 | 合格标准 |
| --- | --- |
| 坐标范围 | 所有坐标在 `±20 m` 内 |
| NaN/Inf | 不允许出现 |
| 消息频率 | `ground_truth/state` 约等于时长 x 83 条 |
| 轨迹图 | 与场景设计一致，无瞬移和爆炸 |

## 6. MATLAB 数据预处理

### 6.1 15D 特征向量

| 维度 | 内容 |
| --- | --- |
| 1-3 | `p_des = [x_des, y_des, z_des]` |
| 4-6 | `v_des = [vx_des, vy_des, vz_des]` |
| 7-9 | `p_actual = [x, y, z]` |
| 10-12 | `v_actual = [vx, vy, vz]` |
| 13-15 | `e_p = p_des - p_actual` |

输出标签为 4D：

```text
cmd_vel = [vx, vy, vz, wz]
```

标签来自 `/droneN/cmd_vel_teacher`。

### 6.2 滑动窗口采样

采样周期 `Δt=0.1 s`，窗口长度 `W=20`，对应 2 秒历史：

```text
X_k = [x_{k-W+1}, ..., x_k]  ->  u_k
```

9D 消融实验只保留第 7-15 维，即实际位置、速度和误差，去掉目标信息。

### 6.3 归一化与划分

对每个维度单独做零均值单位方差归一化：

```text
x_norm = (x - mu_X) / sigma_X
u_norm = (u - mu_U) / sigma_U
```

训练集、验证集、测试集按 `7:2:1` 划分，并保证各场景样本均匀分布。

保存文件建议：

```text
samples_15d_W20_all_scenes.mat
  X_train, U_train
  X_val, U_val
  X_test, U_test
  mu_X, sig_X, mu_U, sig_U
  meta
```

## 7. LSTM 模型训练

### 7.1 训练配置

| 超参数 | 值 |
| --- | --- |
| 优化器 | Adam |
| 初始学习率 | `1e-3` |
| 学习率衰减 | 每 10 epoch x 0.9 |
| Batch size | 64 |
| 最大 epoch | 50 |
| Early stopping patience | 5 |
| Dropout | 0.2 |

### 7.2 损失函数

采用 4 通道加权 MSE，高度通道权重更大：

```text
L = alpha_xy * (||vx_pred - vx_T||^2 + ||vy_pred - vy_T||^2)
  + alpha_z  * ||vz_pred - vz_T||^2
  + alpha_r  * ||wz_pred - wz_T||^2

alpha_z > alpha_xy >= alpha_r
```

### 7.3 训练顺序

| 顺序 | 模型 | 输入维度 | CPU-only 估计时间 |
| --- | --- | --- | --- |
| 1 | 9D-LSTM | `9D x W=20` | 约 1 h / seed |
| 2 | 15D-LSTM | `15D x W=20` | 约 1.5 h / seed |
| 3 | 15D-BiLSTM | `15D x W=20` | 约 2 h / seed |
| 4 | 15D-BiLSTM-DA | `15D x W=20` | 约 2.5 h / seed |

全部 `4 模型 x 5 seeds = 20` 次训练，CPU 总时间约 140-160 小时，建议使用 `screen` 或 `tmux` 后台运行。

模型包需要保存：

```matlab
save('model_package_15d_biDA_seed42.mat', ...
     'net', ...
     'mu_X', 'sig_X', 'mu_U', 'sig_U', ...
     'W', ...
     'D');
```

## 8. 评估指标与图表

### 8.1 主要指标

| 指标 | 含义 |
| --- | --- |
| 3D RMSE | 主排序指标，越小越好 |
| 高度 RMSE | 单独评估 z 方向控制 |
| 60 s 后漂移率 | 衡量长期稳定性 |
| 最大 3D 误差 | 衡量最差情况 |
| `cmd_vel` 波动 | 衡量控制平滑性 |

### 8.2 论文图表清单

| 图序 | 内容 |
| --- | --- |
| 图 5.1 | S2 圆形 XY 轨迹对比 |
| 图 5.2 | 3D 位置误差时序，包含 60 s 漂移检查线 |
| 图 5.3 | 高度 `z(t)` 对比 |
| 图 5.4 | `cmd_vel` 四通道波形 |
| 图 5.5 | RMSE/MAE/max 误差统计柱状图 |
| 图 5.6 | S3 重构时的 3D 误差时序 |
| 图 5.7 | S4 风扰下的 XY 轨迹和误差对比 |
| 图 5.8 | S5 长航时完整误差曲线 |
| 图 5.9 | 6 场景综合 overview |

图表样式建议：Times New Roman、四边框、内向刻度、色盲友好配色、300 dpi PNG + PDF。

## 9. 论文修改清单

| 位置 | 问题 | 修改方案 | 优先级 |
| --- | --- | --- | --- |
| §4.1 | 定位为纯 BC，缺少 GCIL 理论支撑 | 重写为 GCIL 框架，引用 Codevilla 2018、Ding 2019 | P0 |
| 对比矩阵 | 包含不存在的 12D-LSTM | 删除 12D-LSTM，保留 12D-MPC 作为独立基线 | P0 |
| 全文数值 | 存在 `X.XX` 占位符 | 复现后填入真实实验数据 | P0 |
| Appendix A | 缺少 PD 稳态误差推导 | 新增圆周轨迹下 PD 控制器误差推导 | P1 |
| §5.2.6 | Teacher RMSE 与上下文不一致 | 明确该值是相对期望轨迹还是相对 LSTM | P1 |

审稿常见问题准备：

| 问题 | 回答要点 |
| --- | --- |
| Student 为什么能超过 Teacher？ | 因为 GCIL 中 Student 输入包含 `p_des/v_des`，Teacher PD 不直接使用该目标条件，信息结构不同 |
| Teacher 是否太弱？ | 纯 PD 对圆周轨迹存在可证明稳态误差，这是本文要克服的理论瓶颈 |
| 为什么不用 12D-LSTM？ | 12D 是 MPC 内部频域特征，不是 LSTM 特征变体 |
| 5 个种子是否足够？ | 5 seeds x 5 场景 = 25 个独立实验，可报告均值和标准差 |

## 10. 完整复现步骤

### STEP 1：解决 NaN Bug（P0 阻塞）

优先完成以下任一路径：

- 路径 1：修改 `teacher_controller.py` 增加 NaN 保护，降低 `omega`，重新编译并测试。
- 路径 2：切换 RotorS，重写 `spawn_4drones.launch` 并适配 Python 节点。

验证标准：S2 圆形运行 120 s，所有坐标合理，`sync_check.py` 显示编队误差稳定。

### STEP 2：重新录制 seed=42 的 5 个场景

录制后运行：

```bash
python3 ~/drone-formation-e2e/scripts/diagnose_bags.py
python3 ~/drone-formation-e2e/scripts/analyze_bags.py
```

确认所有坐标在合理范围内，轨迹图与场景设计一致。

### STEP 3：扩展到 5 个种子

将 `collect_seed42.sh` 扩展为 `collect_all_seeds.sh`，完成 25 个 bag 录制。

### STEP 4：预处理为训练样本

实现 `bag_to_mat.m` 或 Python 版本，提取 15D 特征、4D 标签、滑动窗口和归一化参数。

### STEP 5：训练 4 个模型变体

按顺序训练：

```text
9D-LSTM -> 15D-LSTM -> 15D-BiLSTM -> 15D-BiLSTM-DA
```

每次训练记录验证集 loss、早停 epoch 和测试 RMSE。

### STEP 6：离线评估并生成图表

计算所有模型在测试集上的 3D RMSE、高度 RMSE、漂移率、最大误差和控制波动，生成论文图表。

### STEP 7：填写论文数值

用真实复现实验结果替换所有 `X.XX`，检查 12D-LSTM 是否已删除、图表是否一致、结论是否有数据支撑。

## 11. 原始文章关键参考值

以下数值仅用于对比，复现完成后应替换为真实结果。

| 来源 | 数据 | 用途 |
| --- | --- | --- |
| §5.2.6 | LSTM RMSE = 2.154 m；Teacher RMSE = 9.134 m | 对比 Student 与 Teacher |
| §5.2.2 | Teacher 相对 LSTM 的 RMSE 约 6.56 m | 注意这是控制器之间误差，不是相对期望轨迹 |
| Table 5.4 | 12D 优化 MPC 3D 误差 = 0.48 m | MPC 强基线 |
| §3.5 | 12D 增强模型 = 小波 + FFT 融合 | 澄清 12D 属于 MPC 内部设计 |
| §5.3.4 | 9D 普通 LSTM 整体 RMSE = 1.57 m | 特征消融参考 |
| §5.3.4 | 15D 防漂移 LSTM RMSE = 0.30 m | 完整模型参考 |

注意：`0.08 m`、`2.5 m`、`0.03 m` 等应视为预期目标或占位值，不能作为已验证结论直接写入论文。

## 12. 常用命令速查

### 12.1 环境启动

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
source ~/drone-formation-e2e/ros_ws/devel/setup.bash

roslaunch drone_sim scene02_circle_4drones.launch gui:=false
```

### 12.2 实时验证

```bash
rostopic hz /drone1/cmd_vel

for i in 1 2 3 4; do
  echo "drone$i:"
  rostopic echo /drone$i/ground_truth/state/pose/pose/position -n 1
done

python3 /tmp/sync_check.py
```

### 12.3 Bag 操作

```bash
rosbag record -O ~/drone-formation-e2e/data/raw_bags/s2_circle.bag \
  --duration=90 \
  /drone{1..4}/p_des /drone{1..4}/v_des \
  /drone{1..4}/ground_truth/state \
  /drone{1..4}/cmd_vel /drone{1..4}/cmd_vel_teacher

rosbag info ~/drone-formation-e2e/data/raw_bags/s2_circle.bag

python3 ~/drone-formation-e2e/scripts/diagnose_bags.py
python3 ~/drone-formation-e2e/scripts/analyze_bags.py
```

### 12.4 编译 ROS 工作区

```bash
cd ~/drone-formation-e2e/ros_ws
catkin_make
source devel/setup.bash

chmod +x ~/drone-formation-e2e/ros_ws/src/drone_sim/scripts/*.py
```

### 12.5 进程清理

```bash
killall -9 gzserver gzclient rosmaster roscore rosout 2>/dev/null
pkill -9 -f "scene_driver|teacher_controller|enable_motors|wind_driver|rosbag" 2>/dev/null
sleep 4
```

## 13. Word/Git 工作流

本仓库仍保留 Word 文档编辑和 Git 归档流程。

### 13.1 推荐结构

```text
docs/       Word 主文档
snapshots/ 里程碑 PDF 或截图
```

### 13.2 VS Code 编辑 Word

安装 SuperDoc：

```powershell
code --install-extension superdoc-dev.superdoc-vscode-ext
```

### 13.3 Git LFS

`.docx` 文件建议通过 Git LFS 管理：

```powershell
git lfs install --local
```

### 13.4 常用 Git 命令

```powershell
git status
git add .
git commit -m "update experiment readme and documents"
git push
```
