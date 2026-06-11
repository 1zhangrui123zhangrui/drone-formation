# PROJECT_README — BiLSTM-DA 无人机编队端到端控制复现

> **作用**：给新开窗口的 Claude/Kiro 提供完整上下文。每次完成重要阶段后请更新此文件。

---

## 论文摘要

**题目**：基于双向LSTM与双重注意力机制的子母无人机编队端到端控制方法 (BiLSTM-DA)

**论文文件**：`docs/初版.docx`（已提取为 `docs/Paper.md`）

**核心叙事**：
```
PD Teacher (0.08m RMSE) → 9D-LSTM (6.56m RMSE) → 15D-BiLSTM-DA (0.30m RMSE，突破PD理论瓶颈)
```

**GCIL理论依据**：Student 能超越 Teacher 是因为 Student 看到 `p_des/v_des`（目标条件输入），Teacher 看不到（纯状态反馈 PD），这一信息差是学习前馈的理论根据（Codevilla ICRA2018 Conditional Imitation Learning）。

---

## 方法细节（§4）

### 15维输入向量
```
x(t) = [p_des(3), v_des(3), p(3), v(3), u_teacher_xyz(3)]
         期望位置   期望速度   实际位置  实际速度   Teacher参考速度前3维
```
注意：第15组不是 `e_p` 而是 **Teacher参考速度指令前三分量** `u_teacher_xyz`（论文§4.2原文）。

### 滑动窗口
- 采样周期 Δt = 0.1s（10Hz），窗口长度 W = 20 步（2s历史）
- 归一化：训练集统计量，每维独立 z-score（均值0，方差1）
- 数据集划分：**7:2:1**（训练/验证/测试），各场景均匀采样

### BiLSTM 主干（§4.3）
- 正反向各 **2层堆叠**，每层 **64个隐藏单元**，tanh激活
- 双向融合：可学习线性组合 `h_t = W_f * h_t^f + W_b * h_t^b + bias`
- 部署时：对固定长度历史窗口做双向重编码（非在线流式，而是每步对20帧窗口跑一遍BiLSTM）

### 双重注意力（§4.4）
**特征注意力（FA，BiLSTM前）**：
```
e = W_fa * x_t + b_fa      # (15,) → scalar per dim
α_fa = softmax(e)           # (15,)
x̃_t = α_fa ⊙ x_t           # 加权输入
```

**时序注意力（TA，BiLSTM后）**：
```
e_t = v^T * tanh(W_ta * h_t + b_ta)   # per timestep scalar
α_ta = softmax(e)                       # (W,)
c = Σ α_ta_t * h_t                     # context vector (128-dim)
```

**输出**：
```
y_hidden = ReLU(W1 * c + b1)    # 32 units
u_S = W2 * y_hidden + b2         # 4D cmd_vel (linear)
```

### 训练（§4.5）
| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 学习率 | 初始（文中未明确，约1e-3），每10 epoch × 0.9衰减 |
| Batch size | 64 |
| Max epochs | 50 |
| Early stopping | patience=5（验证集） |
| Dropout | 保留率0.9（BiLSTM后） |
| 损失函数 | 分通道加权 MSE（垂直通道权重更大） |
| 训练策略 | 两阶段：①全场景监督预训练 ②漂移增强微调 |

### Teacher-Student 控制权分配（§4.6）
- **漂移检测**：`d_i(t) = ||p_i(t) - p_des_i(t)||_2`
- **软切换**：`d < d_threshold` 时，`α = f(d)` 平滑插值，`u = α*u_S + (1-α)*u_T`，`α_min=0.6`
- **硬切换**：`d ≥ d_threshold` 时，切换至 Teacher 接管 τ 秒
- **滑窗重置**：每 **30s** 用期望状态替换历史窗口（防积累误差）
- **轻量化**：结构化剪枝，单步推理 ≤ 8ms（满足100Hz实时需求）

---

## 实验设计（§5）

### 仿真平台
- ROS Noetic + Gazebo 11 + hector_quadrotor（RAFALAMAO Noetic fork）
- 4架子机，矩形编队 (±0.5, ±0.5)m，间距1m

### 5个场景
| 场景 | 类型 | 时长 | 参数 |
|------|------|------|------|
| S1 | 静态悬停 | 60s | 固定点，无运动 |
| S2 | 圆形巡航 | 90s | R=3m, ω=0.4 rad/s（v=1.2m/s）|
| S2' | 八字（双纽线）| 90s | R=3m, ω=0.4 rad/s |
| S3 | 编队重构 | 120s | 矩形→菱形(40s)→三角+中心(80s)，中心走圆 |
| S4 | 风扰 | 90s | v_mean=2.0, A=1.0, f=0.5Hz, σ=0.3, k_wind=0.3 |
| S5 | 长航时 | 180s | 圆形巡航180s |

### 7个对比方法
1. PID（经典）
2. MPC（12D优化，内部小波+FFT特征，论文原文含义）
3. Teacher PD（本文教师控制器）
4. 9D-LSTM（9维输入：p+v+ω）
5. 15D-LSTM（15维单向LSTM）
6. 15D-BiLSTM（无注意力）
7. **15D-BiLSTM-DA**（本文方法）

### 消融3轴
| 消融轴 | 对比 |
|--------|------|
| 特征维度 | 9D vs 15D |
| 时序方向 | 单向 vs 双向 |
| 注意力 | 无DA vs DA |

### 关键定量结果（§5/§6.1论文声称）
| 方法 | S2圆形巡航3D RMSE |
|------|------------------|
| Teacher PD | ~1.64m（0.30的5.47倍，从降低81.7%推算）|
| 9D-LSTM | 6.56m |
| 15D-LSTM | ~2.154m（从67.2%降幅推算）|
| 15D-BiLSTM | ~1.57m |
| 15D-BiLSTM（无DA）| ~1.57m（等同，从80.9%降幅推算约0.30/0.191）|
| **BiLSTM-DA（本文）** | **0.30m** |

注：消融结果 9D→15D: 6.56→2.154m (-67.2%)，单向→双向: 2.154→1.57m (-27.1%)

**长航时S5**：硬切换0次，残差权重 0.6~1.0区间稳定，推理≤8ms

### Teacher PD增益
```
kp_h=2.0, kd_h=1.0, kp_z=2.5, kd_z=1.2, kyaw=0.5
```

---

## 当前进度与状态（最后更新：2026-06-11）

### ✅ 已完成

#### 仿真与数据录制
- 4机编队仿真框架（ROS Noetic + Gazebo11 + hector_quadrotor，teacher_controller, scene_driver, wind_driver）
- 5个场景 launch 文件（omega 已降至 0.2 rad/s，kp_h 降至 1.5）
- teacher_controller.py 加 NaN/Inf 输出保护
- **所有6个 bag 已录制完毕**（data/raw_bags/scene0*.bag）

#### 数据质量（NaN 修复后）
| Bag | 可用样本 | 说明 |
|-----|---------|------|
| scene01_hover_4drones_seed42.bag | 2396（4机均干净）| S1 悬停 |
| s2_circle_4drones_seed42.bag | 1798（drone1/2 NaN 已过滤）| S2 圆形 |
| scene02p_lemni_4drones_seed42.bag | 3596（4机均干净）| S2' 双纽线 |
| scene03_reconfig_4drones_seed42.bag | 3068（drone1/4 部分过滤）| S3 编队重构 |
| scene04_wind_4drones_seed42.bag | 2269（drone2/3 部分过滤）| S4 风扰 |
| scene05_longtime_4drones_seed42.bag | 4055（drone3/4 部分过滤）| S5 长航时 |
| **合计** | **17182 干净样本** | 全部零 NaN |

#### Python 数据管道
- `scripts/bag_to_mat.py` ✅ — 读 rosbag，同步对齐4话题，构造15D特征，NaN 过滤，保存 .mat
- `data/processed/raw/*.mat` ✅ — 6个 .mat 已生成

#### MATLAB 数据管道（等待在 Windows MATLAB 中运行）
- `matlab/data_pipeline/normalize.m` ✅ — z-score + 持久化 stats
- `matlab/data_pipeline/bag_to_samples.m` ✅ — 加载 Python .mat，格式验证
- `matlab/data_pipeline/prepare_all_datasets.m` ✅ — 合并→归一化→滑窗(W=20)→7:2:1分割
- 预期输出：~12013 训练窗口 / 3432 验证 / 1716 测试

#### MATLAB 训练脚本（等待在 Windows MATLAB 中运行）
- `matlab/train/attention_layer_feature.m` ✅ — 特征注意力自定义层（FA，论文§4.4.1）
- `matlab/train/attention_layer_temporal.m` ✅ — 时序注意力自定义层（TA，论文§4.4.2）
- `matlab/train/train_lstm_9d.m` ✅ — C1: 9D-LSTM（128 hidden）
- `matlab/train/train_lstm_15d.m` ✅ — C2: 15D-LSTM（192 hidden）
- `matlab/train/train_bilstm.m` ✅ — C3a: 15D-BiLSTM（无注意力，消融）
- `matlab/train/train_lstm_bidir_attn.m` ✅ — C3: BiLSTM-DA（本文方法）
- `matlab/train/train_all_models.m` ✅ — 批量训练入口

#### MATLAB 评估脚本
- `matlab/evaluation/compute_metrics.m` ✅ — RMSE/MAE/最大误差
- `matlab/evaluation/build_comparison_table.m` ✅ — 7方法×5场景 RMSE 表
- `matlab/evaluation/ablation_table.m` ✅ — 消融3轴结果表
- `matlab/evaluation/plot_all_figures.m` ✅ — IEEE 风格图表
- `matlab/run_full_experiment.m` ✅ — 一键运行顶层脚本

### ⏳ 待完成

1. **在 Windows MATLAB 中运行训练流程**
   ```matlab
   cd('\\wsl.localhost\Ubuntu-20.04\home\jiuyao\drone-formation-e2e')
   addpath('matlab/data_pipeline','matlab/train','matlab/evaluation')
   run matlab/run_full_experiment.m
   ```
   MATLAB 路径：`D:\MATLAB R2024\MATLAB\bin\matlab.exe`

2. **训练结果验证**：检查各模型验证损失曲线，确认无 NaN/发散

3. **闭环评估**：将训练好的模型部署到 ROS `online_lstm_controller.py`，跑仿真采集 p_actual vs p_des

4. **生成论文图表**：RMSE 对比表（Table 1）+ 消融表（Table 2）+ IEEE 轨迹图

---

## 代码框架

```
drone-formation-e2e/
├── ros_ws/src/
│   ├── drone_sim/
│   │   ├── scripts/
│   │   │   ├── teacher_controller.py     # ✅ PD Teacher（含NaN保护）
│   │   │   ├── scene_driver.py           # ✅ 轨迹生成
│   │   │   ├── wind_driver.py            # ✅ 风场
│   │   │   └── enable_motors_delayed.py  # ✅ 电机使能
│   │   └── launch/                       # ✅ 5个场景（omega=0.2 rad/s）
│   └── drone_control/
│       └── scripts/
│           └── online_lstm_controller.py # ⏳ 仅模板，待训练后填入推理
├── matlab/
│   ├── data_pipeline/
│   │   ├── bag_to_samples.m              # ✅ 加载Python .mat，验证格式
│   │   ├── normalize.m                   # ✅ z-score + 持久化 stats
│   │   ├── sliding_window.m              # ✅ 滑动窗口
│   │   └── prepare_all_datasets.m        # ✅ 全流程数据准备（新建）
│   ├── train/
│   │   ├── attention_layer_feature.m     # ✅ FA 自定义层（新建）
│   │   ├── attention_layer_temporal.m    # ✅ TA 自定义层（新建）
│   │   ├── train_lstm_9d.m               # ✅ C1: 9D-LSTM
│   │   ├── train_lstm_15d.m              # ✅ C2: 15D-LSTM
│   │   ├── train_bilstm.m                # ✅ C3a: BiLSTM（新建）
│   │   ├── train_lstm_bidir_attn.m       # ✅ C3: BiLSTM-DA
│   │   └── train_all_models.m            # ✅ 批量训练入口（新建）
│   ├── evaluation/
│   │   ├── compute_metrics.m             # ✅ RMSE/MAE
│   │   ├── build_comparison_table.m      # ✅ 7方法对比表（新建）
│   │   ├── ablation_table.m              # ✅ 消融表（新建）
│   │   └── plot_all_figures.m            # ✅ IEEE 图表
│   └── run_full_experiment.m             # ✅ 一键运行（新建）
├── scripts/
│   ├── bag_to_mat.py                     # ✅ rosbag→15D特征→.mat（新建）
│   ├── analyze_bags.py                   # ✅ 可视化工具
│   ├── diagnose_bags.py                  # ✅ 诊断工具
│   └── record_bags_v2.sh                 # ✅ 录包脚本（omega=0.2版）
├── data/
│   ├── raw_bags/                         # ✅ 6个 bag（scene0*.bag）
│   ├── processed/raw/                    # ✅ 6个 .mat（Python 已转换）
│   ├── processed/                        # ⏳ 等待 MATLAB prepare_all_datasets
│   └── trained_models/                   # ⏳ 等待 MATLAB 训练
└── PROJECT_README.md                     # 本文件（上下文锚点）
```

---

## 下一步执行计划

### 阶段3（当前）：在 Windows MATLAB 中训练模型

**前置条件已全部满足**：
- `data/processed/raw/*.mat` 已存在（6个场景，17182样本）
- 所有 MATLAB 脚本已实现

**操作步骤**：
```matlab
% 1. 打开 Windows MATLAB（D:\MATLAB R2024\MATLAB\bin\matlab.exe）
% 2. 设置工作目录（WSL2 共享路径）
cd('\\wsl.localhost\Ubuntu-20.04\home\jiuyao\drone-formation-e2e')
addpath('matlab/data_pipeline', 'matlab/train', 'matlab/evaluation')

% 3. 一键运行（数据准备 + 训练 + 验证）
run matlab/run_full_experiment.m
```

预期训练时长（CPU）：
- 9D-LSTM：~5 min
- 15D-LSTM：~8 min
- BiLSTM（无注意力）：~10 min
- BiLSTM-DA：~15 min

### 阶段4：闭环评估

1. 将模型部署到 `ros_ws/src/drone_control/scripts/online_lstm_controller.py`
2. 跑仿真，采集各场景 `p_actual` vs `p_des`
3. 调用 `compute_metrics(p_actual, p_des)` 计算每场景每方法 RMSE
4. 调用 `build_comparison_table(results)` 生成 Table 1
5. 调用 `ablation_table(r9, r15, rbi, rda)` 生成 Table 2

### 阶段5：IEEE 图表与论文写作

1. `plot_all_figures('data/eval_results', 'figures')` 生成柱状图、消融图、轨迹图
2. 补充 PID/MPC 基线指标（需单独实现或填入参考值）
3. 整合进论文

---

## 环境信息

- WSL2 Ubuntu20.04, ROS Noetic, Gazebo11
- hector_quadrotor（RAFALAMAO Noetic fork）
- 无CUDA（Intel Iris Xe），训练在 Windows MATLAB R2024（Deep Learning Toolbox）
- Python 3.8，scipy 1.10.1（用于 bag→.mat 转换）
- MATLAB 路径：`D:\MATLAB R2024\MATLAB\bin\matlab.exe`
- WSL2↔Windows 共享：WSL2 `~/drone-formation-e2e/` = Windows `\\wsl.localhost\Ubuntu-20.04\home\jiuyao\drone-formation-e2e\`

## 重要笔记

- **NaN 根因**：hector_quadrotor 推进/阻力模型数值不稳定（高速/长时间），非控制器 bug。修复方案：omega=0.2，kp_h=1.5，teacher NaN 保护。
- **15维第15组子向量**：是 `u_teacher_xyz`（Teacher速度指令前3维），**不是** `e_p`（论文§4.2原文确认）。
- **双向LSTM部署方式**：每步对20帧历史窗口做完整双向重编码，**不是**在线流式（每步都跑一遍 BiLSTM）。
- **自定义注意力层**：`attention_layer_feature.m` / `attention_layer_temporal.m` 继承 `nnet.layer.Layer`，MATLAB R2019b+ 兼容，通过 `predict()` 实现前向传播，MATLAB 自动微分处理反传。
- **数据分割方式**：先合并所有场景样本，再在窗口级别做 shuffle + 7:2:1 分割（rng seed=42）。
