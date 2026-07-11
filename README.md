# drone-formation-e2e

子母无人机编队端到端控制复现实验仓库。当前论文正文分别位于 [docs/initial_part1.docx](docs/initial_part1.docx) 和 [docs/experiment.docx](docs/experiment.docx)，代码实现覆盖 ROS/Gazebo 仿真、Teacher 数据生成、MATLAB 数据处理与训练脚本，以及部分结果可视化。

本 README 已整合并取代此前根目录中分散的实验说明、阶段记录和临时上下文文件。以下状态以 **2026-07-11** 的仓库实际文件为准，而不是早期计划稿。

## 论文方法概要

本文方法是一个 Teacher-Guided 的端到端编队控制框架 `BiLSTM-DA`，核心结构是：

1. `15D` 输入特征：`[p_des(3), v_des(3), p_actual(3), v_actual(3), u_teacher_xyz(3)]`
2. `W=20` 的滑动时间窗，采样周期 `0.1 s`
3. 双向 `BiLSTM` 主干
4. 特征注意力 `FA` + 时序注意力 `TA`
5. Student/Teacher 软硬切换与长航时部署增强

当前代码中，`bag_to_mat.py` 和 MATLAB 数据管线使用的 15 维定义已经和论文一致，最后 3 维是 `u_teacher_xyz`，不是 `e_p`。

## 当前完成到哪一步

### 已完成

| 模块 | 实际状态 | 证据 |
| --- | --- | --- |
| 论文方法与实验设计文稿 | 已更新到 Word 原稿版本 | `docs/initial_part1.docx`, `docs/experiment.docx` |
| 4 机仿真场景/Teacher 控制器 | 已实现并可生成 rosbag | `ros_ws/src/drone_sim/scripts/` |
| rosbag 转 `.mat` | 已完成 | `scripts/bag_to_mat.py` |
| MATLAB 数据集构建 | 已完成并已有产物 | `data/processed/dataset_*`, `norm_stats_*` |
| 9D-LSTM 训练 | 已产出模型 | `data/trained_models/c1_lstm9d.mat` |
| 15D-LSTM 训练 | 已产出模型 | `data/trained_models/c2_lstm15d.mat` |
| 15D-BiLSTM 无注意力训练 | 已产出模型 | `data/trained_models/c3a_bilstm.mat` |
| 15D-BiLSTM-DA 训练 | 已产出模型 | `data/trained_models/c3_bidir_attn.mat` |
| canonical test set 离线评测 | 已完成 | `data/eval_results/offline/offline_summary.csv` |
| 训练留痕审计 | 已完成 | `data/eval_results/training_audit/training_audit.csv` |
| MATLAB 最小在线推理链路 | 已实现 `c1/c3a` 专用入口，且 `localhost` 连接流程已打通 | `matlab/deployment/run_online_model_controller.m` |
| bag 轨迹/误差可视化 | 已完成 | `results/figures/*.png`, `*.pdf` |

### 尚未完成

| 模块 | 当前问题 |
| --- | --- |
| `BiLSTM-DA` 最终模型 | checkpoint 已存在，但当前离线效果明显落后于 `c1/c3a`，不能直接当作论文主方法结果 |
| 在线部署控制器 | Python 占位节点仍未打通；当前最小可用链路改为 MATLAB + ROS Toolbox，仅覆盖 `c1/c3a`。ROS 连接已打通，但闭环稳定性仍在排查 |
| 论文主实验闭环评估 | 尚未形成 8 方法 x 多场景的真实结果矩阵 |
| 统计显著性 | 没有 5 次独立运行均值/方差与 `p-value` |
| 论文图表 | 当前大多是 bag 诊断图，不是论文最终对比图 |

## 数据现状审计

下面的统计现在分成两层理解：

- `seed42` 旧 bag 与早期失败 pilot 仍保留，用于诊断历史问题
- 当前真正应作为主数据入口的，是 `data/raw_bags/v2/` 下通过正式审计的 `v2` bag，以及由它们重建出的 `data/processed/raw/*.mat`

当前 canonical `v2` 原始数据统计如下：

| 场景 | bag 文件 | 干净样本数 | 现状判断 |
| --- | --- | ---: | --- |
| S1 Hover | `scene01_hover_4drones_v2.bag` | 2386 | 4 机全部通过审计 |
| S2 Circle | `scene02_circle_4drones_v2_worldfix_entryfix.bag` | 3591 | 4 机全部通过审计 |
| S2' Lemni | `scene02p_lemni_4drones_v2.bag` | 3582 | 4 机全部通过审计 |
| S3 Reconfig | `scene03_reconfig_4drones_v2.bag` | 4774 | 4 机全部通过审计 |
| S4 Wind | `scene04_wind_4drones_v2.bag` | 3581 | 4 机全部通过审计 |
| S5 Longtime | `scene05_longtime_4drones_v2.bag` | 7188 | 4 机全部通过审计 |

`data/processed/` 已基于上述 6 个 `v2` bag 重新生成。需要注意的是，`2026-07-10` 起正式采用了更严格的数据集构建协议：按 `scene × drone × 连续时间段` 切窗、split 之间保留 `W-1` guard rows、并且只用 train split 计算归一化统计。当前构建结果是：

- 原始样本总数：`25102`
- `15D` 窗口总数：`22812`
- 划分结果：`train=16471`, `val=4381`, `test=1960`
- 构建留痕文件：`data/processed/dataset_build_manifest.json`

## 与论文目标的比对

### 已对齐的部分

| 论文要求 | 当前仓库情况 | 是否对齐 |
| --- | --- | --- |
| 15D 输入使用 `u_teacher_xyz` | 代码已这样实现 | 是 |
| `W=20`、`10 Hz`、`7:2:1` 划分 | 数据管线已这样实现 | 是 |
| Teacher PD 数据生成链路 | 已实现 | 是 |
| 9D / 15D / BiLSTM 无注意力训练脚本 | 已实现并已有模型 | 是 |

### 未对齐的部分

| 论文要求 | 当前仓库情况 | 问题 |
| --- | --- | --- |
| 最终 `BiLSTM-DA` 模型 | `c3_bidir_attn.mat` 已训练完成，但当前 canonical test set 上显著落后于 `c1/c3a` | 不能直接写成论文主结果 |
| 在线 Student 推理与软硬切换 | 已有 `c1/c3a` 的最小 MATLAB 闭环推理链路，但完整 Teacher-Student 软/硬切换仍未实现 | 只能用于 `S2` student pilot，不能当作论文最终部署结果 |
| 完整 8 方法主对比 | 只有早期 `c1_lstm9d` bootstrap 记录 | 论文主结果尚未落地 |
| 5 次独立运行统计 | 没有 | 无法支撑论文表 5.8 及显著性检验 |
| PID/MPC/Teacher/学习方法统一评估 | 没有统一评估产物 | 对比矩阵不完整 |
| 长航时 0 次硬切换、`alpha` 曲线 | 没有部署级日志 | 论文部署结论未验证 |

### 当前 canonical 数据上的离线结论

当前基于 `data/processed/` canonical test split 的统一离线评测结果为：

| 模型 | `rmse_mean` | 结论 |
| --- | ---: | --- |
| `c3a_bilstm` | `0.012627` | 当前离线最优 |
| `c1_lstm9d` | `0.013841` | 次优，作为简洁基线仍值得进入闭环 pilot |
| `c2_lstm15d` | `0.015581` | 当前不作为第一优先级闭环对象 |
| `c3_bidir_attn` | `0.037035` | 明显退化，暂不进入下一步闭环主线 |

训练审计也给出了同样方向：`c3` 的 `best_val_loss=0.1469`，明显高于 `c1/c2/c3a`。因此，当前最合理的闭环入口不是直接上 `c3`，而是先用 `Teacher / c1 / c3a` 做一个范围收敛、留痕完整的 `S2` 小闭环 pilot，验证离线排序与在线行为是否一致。

## 当前实验最主要的问题

### 1. 旧的实验说明已经过期

此前根目录存在多个相互重叠的阶段文档，其中最核心的问题是它们把当前状态描述成“动态 bag 全坏，训练还没开始”。这已经不准确。现在的真实情况是：

- 动态场景确实存在严重数值爆炸和轨迹漂移
- 但仓库已经基于样本过滤提取出了可训练数据
- 数据不是“完全不可用”，而是“**部分无人机、部分时段可用**”

这意味着当前阶段不是“完全卡死”，而是“已经进入离线训练中段，但数据质量仍不足以支撑论文最终结论”。

### 2. 旧文档之间互相矛盾

此前根目录旧文件里同时存在这些互相冲突的说法：

- 有的文档说 15D 最后 3 维是 `e_p`
- 有的文档说 15D 最后 3 维是 `u_teacher_xyz`
- 有的文档说 Teacher 只在数据生成阶段使用
- 论文正文现在明确写的是部署阶段 Teacher 仍在线参与融合
- 有的文档说 MATLAB 训练还没开始
- 但实际仓库里已经有 `c1/c2/c3a` 三个模型产物

如果不清掉这些冲突，后续实验会继续沿着错误目标推进。

### 3. 数据质量不足以直接宣称论文结果成立

当前 bag 的问题不是只有 NaN：

- `S2` 圆形场景中，最关键的主实验场景只有 2 架机保留了样本
- `S4/S5` 里有明显飞离、坠落或归零现象
- 样本是事后过滤出来的，不是完整闭环稳定运行得到的

这意味着现有数据能支撑“原型训练”和“数据管线验证”，但还不能严肃支撑论文中关于多场景稳定性、四机全编队一致性、长航时零接管和风扰鲁棒性的结论。

### 4. 部署链路基本还没开始

当前最关键的缺口不是再写一个训练脚本，而是部署没有打通：

- `online_lstm_controller.py` 是模板
- 配置文件仍和论文参数不一致
- 没有真实的 normalization / model loading / rolling window inference
- 没有 Teacher-Student 软硬切换实现

所以现在的“实验做到什么程度”更准确地说是：

`Teacher 数据生成 + 数据集构建 + 前 3 个离线模型训练完成`

而不是：

`论文实验已完成，只差画图`

### 5. 配置文件仍有明显占位值

当前 `configs/*.yaml` 里至少有这些和论文/代码不一致的问题：

- `sample_time: 0.02`，而论文与数据管线使用的是 `0.1 s`
- `c3_bidir_attn.yaml` 里 `sequence_length: 30`，而训练数据用的是 `20`
- `stats_file` 写成 `data/processed/normalization_stats.mat`，但实际文件是 `norm_stats_9d.mat` / `norm_stats_15d.mat`

这些问题说明部署配置还停留在模板阶段。

## 结果与论文现象的比对

### 当前已经观察到的现象

- S1 悬停数据稳定，说明基础仿真链路没问题
- 旧 `seed42` 动态 bag 与修复前的 `S2` pilot 中，部分无人机会大范围漂移或爆炸，说明动态场景稳定性曾是核心问题
- 经过“世界系控制接口 + 平滑入轨 + 按仿真时间正式录制”修复后，`S1-S5` 现在都已经拿到新的 `v2` 审计通过 bag
- 基于这套 `v2` bag 重建后的 canonical 原始样本总量为 `25102`
- 现有 `results/figures/` 主要是 bag 的实际轨迹图和误差图，证明场景确实跑过，但不是论文最终模型对比图

### 和论文目标相比，哪些对，哪些不对

对的部分：

- 论文的理论主线已经体现在代码里：Teacher 数据、15D 目标引导输入、BiLSTM/注意力训练脚本都已存在
- 论文实验矩阵的雏形已经搭好：场景、数据、训练、评估脚本目录都齐

不对的部分：

- 论文主方法 `BiLSTM-DA` 的最终闭环结果还没有实际产出
- 论文中的很多数值仍是文稿中的目标值或占位值，不是当前仓库真实跑出来的结果
- 虽然 `v2` 原始数据已经通过审计，但现有模型 checkpoint 还没有基于这套新数据重训

## 下一步应该怎么完成

建议按下面顺序推进，而不是继续堆新的临时说明文件。

### 阶段 1：先把“真实可用数据”问题收口

1. 重新定义数据合格标准：主实验场景至少保证 4 机全时段可用，不能靠大量事后过滤支撑结论。
2. `S2` 已经拿到一份新的正式 `PASS` bag，接下来重点转为评估 `S2' / S3 / S4 / S5` 是否需要按同样流程重录。
3. 如果继续使用 `hector_quadrotor`，要先验证：
   - 4 机都不爆炸
   - 90 s / 180 s 内不出现大范围漂移
   - 主动态场景都至少拿到一份通过正式审计的四机数据
4. 如果始终做不到，尽快转向更稳定的仿真底座，而不是继续在坏数据上训练。

### 阶段 1 的正式验收标准

从现在开始，任何 bag 只有在通过 `scripts/audit_bag_quality.py` 后，才能进入论文主实验或训练主数据集。

当前采用的高标准验收口径是：

1. 四架无人机都必须有 odom 数据。
2. 每架无人机都必须达到该场景的最小时长。
3. 每架无人机的轨迹坐标都必须处于合理物理范围内。
4. 每架无人机都必须有非零的干净同步样本。
5. 每架无人机的 `clean_ratio` 必须不低于 `95%`。

场景阈值当前定义为：

| 场景 | 最小时长 | 最大允许 `max |pos|` |
| --- | ---: | ---: |
| `scene01_hover` | 55 s | 20 m |
| `scene02_circle` | 85 s | 20 m |
| `scene02p_lemni` | 85 s | 20 m |
| `scene03_reconfig` | 110 s | 25 m |
| `scene04_wind` | 85 s | 30 m |
| `scene05_longtime` | 170 s | 35 m |

注意：这套标准是按“论文最终数据”设计的，不是按“能凑出一些训练样本”设计的。任何不满足标准的数据都只能作为诊断材料，不能当作正式结果使用。

### 阶段 1 的当前准备结论

在正式重录 `S2` 之前，当前工程里还有一个必须明确的现实前提：

- [scene02_circle_4drones.launch](/home/jiuyao/drone-formation-e2e/ros_ws/src/drone_sim/launch/scene02_circle_4drones.launch:1) 当前使用的是 `omega=0.2`
- [teacher_params.yaml](/home/jiuyao/drone-formation-e2e/ros_ws/src/drone_sim/config/teacher_params.yaml:1) 当前使用的是 `kp_h=1.5`

这两个值都是“为了降低数值爆炸概率而做的稳定化配置”，并不等同于论文文稿中最理想的目标参数。为了保证学术诚信，后续必须遵守两条原则：

1. 如果正式实验最终使用的是这组稳定化参数，论文中必须如实说明，不得写成原始目标参数跑出的结果。
2. 如果想回到更激进的目标参数，必须先做逐步稳定性验证，不能直接跳到论文目标值后只保留好看的结果。

按论文要求推进时，必须进一步区分两种口径：

1. **论文目标参数口径**
   - 指论文文稿希望验证的原始目标参数组合
   - 只有在这些参数下完成稳定性验证并通过正式审计，才可以写成“按论文目标参数复现”
2. **稳定化复现实验口径**
   - 指当前为了让 `hector_quadrotor` 不发生数值爆炸而采用的降阶配置
   - 如果最终论文实验沿用 `omega=0.2`、`kp_h=1.5`，正文必须明确写成“稳定化复现配置”或同等含义，不能与论文目标参数混写

因此，当前 `S2` 的第一轮正式任务不是“直接追求最好看数值”，而是：

- 先用当前稳定化配置录出 **四机全程 PASS** 的 `S2`
- 在此基础上再决定是否逐步恢复更接近论文目标的参数

### 2026-07-09 最新诊断、修复与 S2 正式通过结果

今天围绕 `S2` 已经完成了一条完整闭环：先复盘旧失败，再做诊断 pilot，随后修复控制链路和场景进入方式，最后重新正式录制并通过审计。需要把这条因果链明确记录下来。

1. `data/raw_bags/v2/scene02_circle_4drones_v2.bag` 的失败首先暴露了**录制流程错误**。
   - 该 bag 的消息时间戳范围只有约 `69 s` 仿真时间，不是目标的 `90 s`
   - 根因是旧版 `record_bags_v2.sh` 按**墙钟时间** `sleep 90` 后停止录包
   - 在 `use_sim_time=true` 的 Gazebo 下，墙钟 `90 s` 并不等于仿真 `90 s`

2. 随后的 `S2` pilot 证明旧问题不止是录包流程，确实还存在**真实的动力学/控制失稳**。
   - `wait_scene_ready.py` 能正常判定 4 机 ready，说明场景起飞链路已经打通
   - 但旧控制链路下，`scene02_circle` 仍会在早期触发 `flip over`、`NaN` 和轨迹发散
   - 这说明仅修复录制流程并不足以得到论文级原始数据

3. 这次真正找到的核心根因有两个，而且都已经修复。
   - `teacher_controller.py` 的水平 PD 是按**世界坐标系**位置误差生成 `vx, vy`
   - 但 `hector` 的 `/cmd_vel` 会把 `geometry_msgs/Twist` 解释为 **stabilized frame** 指令，再按 yaw 旋转到世界系
   - 结果就是 Teacher 的世界系控制被**重复旋转**，动态场景下方向会系统性偏掉
   - 正确修复不是改数据标签，而是把真实执行指令改发到 `/droneN/command/twist`，使用 `TwistStamped` 的世界系接口
   - 同时保留 `/droneN/cmd_vel_teacher` 作为世界系 Teacher 标签，这样论文的 `15D` 特征/标签语义不变
   - 第二个根因是动态轨迹启动过于突兀：圆轨迹一开始就在运动，而无人机刚 spawn 时还在原点附近
   - 修复方式是在 `scene_driver.py` 中加入 `prehover_duration` 和 `transition_duration`，让动态场景先悬停，再平滑进入目标轨迹

4. 本次仓库内的正式修复包括：
   - `scripts/record_bags_v2.sh` 改为先等待**4 机服务、关键话题、起飞高度和最低仿真时间**，再开始录包
   - 录包现在包含 `/clock`，并改为按 **sim time** 等待目标时长
   - `enable_motors_delayed.py` 增加 `engage` 重试
   - `teacher_controller.py` 真实控制改发 `/droneN/command/twist`
   - `scene_driver.py` 新增平滑入轨逻辑
   - `scene02_circle_4drones.launch`、`scene02p_lemni_4drones.launch`、`scene03_reconfig_4drones.launch`、`scene04_wind_4drones.launch` 已统一接入 `prehover_duration=4.0` 与 `transition_duration=8.0`

5. 诊断历史仍然保留，便于追溯。
   - 旧正式失败 bag：`data/raw_bags/v2/scene02_circle_4drones_v2.bag`
   - 旧诊断 pilot bag：`data/raw_bags/v2/scene02_circle_4drones_v2_pilot_2026-07-09_interrupt.bag`
   - 新正式通过 bag：`data/raw_bags/v2/scene02_circle_4drones_v2_worldfix_entryfix.bag`

### 2026-07-09 夜间 S2 正式重录结果

修复完成后，已经实际执行：

- `RECORD_BAG_SUFFIX='_worldfix_entryfix' bash scripts/record_bags_v2.sh s2`

本次结果已经满足当前 README 中定义的论文级数据审计门槛：

- `rosbag info` 显示 bag 时长约 `90 s`，时间范围 `14.28 s -> 104.42 s`
- `scripts/audit_bag_quality.py` 对 `scene02_circle` 的审计结果为 `PASS`
- 四架无人机全部满足 `duration >= 85 s`
- 四架无人机全部满足 `clean_ratio = 100.0%`
- 四架无人机全部满足 `max|pos| = 3.45 m`，远低于 `20 m` 上限

审计结果如下：

- `drone1`: `duration=90.1s`, `clean=902`, `clean_ratio=100.0%`, `max|pos|=3.45m`
- `drone2`: `duration=90.0s`, `clean=898`, `clean_ratio=100.0%`, `max|pos|=3.45m`
- `drone3`: `duration=89.8s`, `clean=896`, `clean_ratio=100.0%`, `max|pos|=3.45m`
- `drone4`: `duration=89.7s`, `clean=895`, `clean_ratio=100.0%`, `max|pos|=3.45m`

这说明本次修复的意义不是“把失稳延后一点”，而是已经把旧 `S2` 失败中的核心控制执行错误和入轨冲击问题同时消掉了。当前可以正式确认：

- `S2` 已经拿到一份**四机全程 PASS** 的正式原始 bag
- 这份结果目前属于**稳定化复现实验配置**，因为它仍然建立在 `omega=0.2`、`kp_h=1.5` 的前提上
- 因此它已经满足“论文级原始数据”的质量门槛，但还不能自动写成“原始论文目标参数下的复现结果”

### 下一步执行顺序

在当前事实下，实验部分最合理的下一步已经不是继续修 `S2`，而是把这份通过结果作为稳定基线向后推进：

1. **先固化 `S2` 稳定基线**
   - 把 `scene02_circle_4drones_v2_worldfix_entryfix.bag` 视为当前第一份合格的 `S2` 正式数据
   - 后续所有 README、论文草稿和实验表述都要明确它对应的是“稳定化复现实验配置”

2. **立即做 `S2` 的可重复性验证**
   - 用完全相同的参数和流程再独立重录 `2-3` 次
   - 目标不是追求更好看的数值，而是确认这次 `PASS` 不是偶然样本
   - 每次都要保留 bag 名称、`rosbag info`、`audit_bag_quality.py` 输出

3. **把同一套修复思路推到其余动态场景**
   - 按优先级建议先做 `S2'`，再做 `S4`，然后 `S5`
   - 因为这些场景同样依赖动态目标轨迹，也最可能受益于“世界系控制接口 + 平滑入轨”
   - 只有动态场景整体稳定后，训练集和主结果表才有真正可靠的基础

4. **把“回到论文目标参数”单列成后续研究任务**
   - 现在不应该回头破坏已经通过的 `S2` 基线
   - 如果后面要恢复更接近论文目标的参数，必须从这份稳定基线出发，做单变量递进验证
   - 每次调整都只能作为新的诊断实验，不能与当前稳定化正式结果混写

5. **训练与论文主表的准入条件**
   - 至少先确保主动态场景中，正式纳入的数据都通过 `audit_bag_quality.py`
   - 再基于这些通过审计的 bag 重新生成 `processed/raw` 和训练集
   - 在此之前，现有很多处理产物仍只能看作“阶段性诊断产物”，不能直接当最终论文结果

### 2026-07-09 夜间六场景 v2 数据集重建结果

在 `S2 / S2' / S3 / S4 / S5` 全部通过审计后，又补录了 `S1`，因此当前已经形成一套完整的六场景 `v2` canonical bag 集：

- `scene01_hover_4drones_v2.bag`
- `scene02_circle_4drones_v2_worldfix_entryfix.bag`
- `scene02p_lemni_4drones_v2.bag`
- `scene03_reconfig_4drones_v2.bag`
- `scene04_wind_4drones_v2.bag`
- `scene05_longtime_4drones_v2.bag`

随后已完成两步正式重建：

1. 用 `scripts/bag_to_mat.py` 重写 `data/processed/raw/*.mat`
2. 用新的 `scripts/prepare_datasets.py` 重写 `data/processed/` 下的归一化统计与训练/验证/测试集

当前重建结果为：

- `15D`: `train=16471`, `val=4381`, `test=1960`
- `9D`: `train=16471`, `val=4381`, `test=1960`
- 归一化统计：`norm_stats_15d.mat`, `norm_stats_9d.mat`
- 构建留痕：`dataset_build_manifest.json`

必须明确的一点是：

- 当前 `data/trained_models/` 中已有 checkpoint 已经不再对应最新的 boundary-safe canonical 数据集
- 这次完成的是“更严格协议下的新 canonical 数据与训练集重建”，需要基于它们重新训练模型
- 任何基于旧 `processed/` 数据集得到的离线结果或模型排序，都不应再直接作为论文正式证据

### 论文统计口径规则

从现在开始，论文实验结果必须遵守下面的统计规则，不能再按单次最好结果组织结论：

1. 论文主表、主图、鲁棒性对比结论必须来自**多组独立运行的均值统计**，不能只取单次最好 bag。
2. 当前已经完成的 `S2` `repeat01/repeat02` 属于**稳定性复验**，它的作用是证明 `PASS` 不是偶然；这一步可以先做 `2-3` 次。
3. 真正进入论文结果汇总时，主动态场景和主要对比方法应至少做 **5 次独立运行**，统一报告 `mean/std/n`，如果做显著性检验，也必须基于同一批独立重复。
4. 训练用 canonical 数据集与论文评测用重复运行 bag 必须分开管理，不能把“训练集构建用 bag”和“评测均值统计用 bag”混成同一口径。
5. 使用 `omega=0.2`、`kp_h=1.5` 的结果，仍然只能归入**稳定化复现实验配置**；如果后续回到论文目标参数，必须单独成组统计，不能混算平均值。

### 当前最优先的下一步

截至 `2026-07-10`，canonical `v2` 数据上的一轮正式 MATLAB 重训、离线评测与训练审计已经完成。当前最优先的下一步不再是重复训练，而是把工作收敛到一个可审计的闭环入口：

1. 先打通最小在线部署链路。
   - 目标只包含 `Teacher / c1 / c3a`
   - 不把 `c2` 和当前明显退化的 `c3` 混入第一轮闭环
2. 先做 `S2` 小闭环 pilot，而不是直接扩展到全场景与 8 方法。
3. 只有在这个 pilot 证明在线链路可信、且闭环排序与离线结果大体一致后，才进入多次独立运行统计。

### 数据集构建修正

从 `2026-07-10` 开始，`processed/` 数据集的正式构建协议应满足以下约束：

1. 以 `scene × drone × 连续时间段` 为单位切窗，窗口不得跨场景、跨无人机或跨时间缺口。
2. 先进行 train/val/test 切分，再只用 train split 计算归一化统计。
3. train/val/test 之间保留 `W-1` 的 guard rows，避免 `stride=1` 造成近邻窗口泄漏。

因此，任何旧的“全局拼接后统一滑窗、再随机切分”的数据集产物，都不应再作为论文正式训练与评测依据。

如果离线评测显示 `c3_bidir_attn` 没有优于更简单的基线，不要立即进入闭环主实验。应先运行：

- `run matlab/run_training_audit.m`

这一步会审查四个 checkpoint 的 `info/metadata`，汇总验证损失、最佳验证位置和训练留痕，帮助判断问题是在训练收敛、早停、实现细节还是模型复杂度本身。

### 阶段 3：打通在线部署

1. 先修正 `configs/*.yaml` 和论文/数据管线的参数不一致。
2. 当前已落地一条**最小 MATLAB 在线链路**：
   - [run_online_model_controller.m](/home/jiuyao/drone-formation-e2e/matlab/deployment/run_online_model_controller.m:1)
   - [run_online_c1_controller.m](/home/jiuyao/drone-formation-e2e/matlab/run_online_c1_controller.m:1)
   - [run_online_c3a_controller.m](/home/jiuyao/drone-formation-e2e/matlab/run_online_c3a_controller.m:1)
   - 它只支持 `c1/c3a`，并通过 MATLAB 直接加载 `.mat` 网络做 ROS 在线推理
   - 当前默认安全策略是：`sim_time < 12s` 时保持 Teacher 执行，`sim_time >= 12s` 后 Student 才接管；同时 Student 输出按 Teacher 的物理限幅裁剪
3. Student pilot 的仿真侧入口是：
   - [scene02_circle_4drones_student.launch](/home/jiuyao/drone-formation-e2e/ros_ws/src/drone_sim/launch/scene02_circle_4drones_student.launch:1)
   - 该入口会让 Teacher 保持 `/cmd_vel_teacher` 在线，但关闭真实执行输出，避免与 Student 抢控制权
4. 当前这台机器上**唯一经过本仓库实际打通验证**的 **Windows MATLAB + WSL ROS** 连接方式是：
   - **WSL ROS master 保持本地：**
     - `ROS_MASTER_URI=http://localhost:11311`
     - `ROS_IP=127.0.0.1`
     - `ROS_HOSTNAME=localhost`
   - **Windows MATLAB 也使用本地地址连接：**
     - `rosinit('http://localhost:11311','NodeHost','localhost')`
     - 或直接运行 [run_connect_ros_wsl_windows.m](/home/jiuyao/drone-formation-e2e/matlab/run_connect_ros_wsl_windows.m:1)
   - 当前 helper [connect_ros_wsl_windows.m](/home/jiuyao/drone-formation-e2e/matlab/deployment/connect_ros_wsl_windows.m:1) 与 [run_online_c1_controller.m](/home/jiuyao/drone-formation-e2e/matlab/run_online_c1_controller.m:1) / [run_online_c3a_controller.m](/home/jiuyao/drone-formation-e2e/matlab/run_online_c3a_controller.m:1) 已按这套 `localhost` 基线收敛
   - **不要再把当前机器的常规连接流程写成 `NodeHost=10.16.33.80`。**
     - 这条数值 IPv4 路线在本项目历史联调中曾出现“`rostopic info` 注册看似正常，但 publisher 端口无法真正监听或 WSL 侧回连失败”的问题
     - 也不要依赖 Windows 主机名自动注册，因为这台机器的主机名 `灵犀` 在 ROS URI 中会变成 punycode `xn--5nxo7b`，WSL 侧若不能解析该名字，就会出现“MATLAB 日志正常、Gazebo 收不到 `/droneN/command/twist`”的假连通
   - 只有在**主动更改了网络拓扑**并且重新做过端到端验证时，才允许偏离 `localhost` 基线
5. 当前推荐的**标准连接流程**如下：
   - **步骤 1：在 WSL 启动场景**
     - `cd /home/jiuyao/drone-formation-e2e`
     - `source ros_ws/devel/setup.bash`
     - `export ROS_MASTER_URI=http://localhost:11311`
     - `export ROS_IP=127.0.0.1`
     - `export ROS_HOSTNAME=localhost`
     - `roslaunch drone_sim scene02_circle_4drones_student.launch gui:=true`
   - **步骤 2：在 Windows MATLAB 启动连接**
     - `cd('\\wsl.localhost\ubuntu-20.04\home\jiuyao\drone-formation-e2e')`
     - `run matlab/run_connect_ros_wsl_windows.m`
   - **步骤 3：在 Windows MATLAB 启动在线控制**
     - `run matlab/run_online_c1_controller.m`
     - 或 `run matlab/run_online_c3a_controller.m`
   - **步骤 4：在 WSL 做最小验证**
     - `rostopic info /drone1/command/twist`
     - `rostopic echo -n 5 /drone1/command/twist`
     - `rostopic hz /drone1/command/twist`
   - **当前正确现象应当是：**
     - publisher URI 显示为 `http://localhost:<port>/`
     - `rostopic echo` 能收到 `TwistStamped`
     - `rostopic hz` 不应退化到接近 `1 Hz`
6. 当前推荐的**最小判因顺序**如下，不要再跳步：
   - **先看连接是否真的打通**
     - 如果 `rostopic echo /drone1/command/twist` 完全收不到消息，先检查 MATLAB 是否按 `localhost` 基线连接，而不是先改模型或先怀疑场景
   - **再看 publisher URI 是否异常**
     - 如果 URI 里出现 `xn--5nxo7b` 或其他 Windows 主机名，说明又回到了主机名自动注册路径
     - 如果 URI 里出现数值 IPv4，也不要直接当成正确，应优先回到 `localhost` 基线复核
   - **连接通了再看频率**
     - 如果 `echo` 有消息，但 `rostopic hz /drone1/command/twist` 明显低于 `10 Hz`，优先怀疑 MATLAB 在线推理链路吞吐，而不是先下结论说模型本身错误
   - **最后才看闭环稳定性**
     - 只有当 topic 接线、消息内容和发布频率都正常后，GUI 里的“乱飞”才有资格被归因到控制/模型/动力学不稳定
7. 再实现 Teacher-Student：
   - 软切换
   - 硬切换
   - 漂移检测
   - 长航时滑窗重置

### 当前收敛出的 `S2` 小闭环 pilot

这个 pilot 的目标不是直接产出论文表 5.8，而是回答两个更基础的问题：

1. 当前 canonical 数据上离线更优的 `c3a`，在闭环里是否也至少优于或不差于 `c1`。
2. 现有在线部署链路是否已经足够可信，可以进入后续多次独立运行统计。

pilot 的固定前提如下：

1. 场景固定为 [scene02_circle_4drones.launch](/home/jiuyao/drone-formation-e2e/ros_ws/src/drone_sim/launch/scene02_circle_4drones.launch:1)。
2. 参数固定为当前**稳定化复现实验配置**：`omega=0.2`、`kp_h=1.5`、`rate=10 Hz`、`W=20`。
3. 对象只包含 `Teacher`、`c1_lstm9d`、`c3a_bilstm`，不包含 `c2` 与 `c3`。
4. 所有 run 都必须独立留痕，且与训练集构建 bag 分开保存。

建议执行顺序如下：

1. **在线推理预检**
   - 先不争论精度，先验证 `c1/c3a` 的在线输入构造、归一化、窗口维护、模型加载、4 机独立推理和 ROS topic 接线全部正确。
   - 当前推荐做法是启动 `scene02_circle_4drones_student.launch`，然后在 Windows MATLAB 中运行：
     - `run matlab/run_online_c1_controller.m`
     - 或 `run matlab/run_online_c3a_controller.m`
   - 这一步若失败，说明问题在部署链路，不在论文方法本身，必须先停下修链路。
   - 预检时必须同时看三项：
     - `rostopic echo -n 5 /drone1/command/twist`
     - `rostopic hz /drone1/cmd_vel_teacher`
     - `rostopic hz /drone1/command/twist`
   - 如果 `cmd_vel_teacher` 接近 `10 Hz`，但 `command/twist` 明显掉到低频，说明当前问题在 Student 在线推理/发布链路，不在场景 driver 或 Teacher 标签链路。
2. **Teacher 基线 run**
   - 用当前 `S2` 稳定化配置跑一条 `Teacher` 正式 bag，确认场景、录制与审计流程本身仍然稳定。
3. **`c1` 单模型闭环 run**
   - 只启用 `c1`，固定其他条件不变，做一条完整 `90 s` 的 `S2` run。
4. **`c3a` 单模型闭环 run**
   - 只启用 `c3a`，固定其他条件不变，做一条完整 `90 s` 的 `S2` run。
5. **通过后再做重复**
   - 对首次通过的学习方法再独立重录 `2` 次，确认 `PASS` 不是偶然。

pilot 的验收口径分两层：

1. **工程准入**
   - 在线节点全程无异常退出。
   - 4 架子机都能持续收到有限值控制指令，不出现 `NaN/Inf`。
   - 控制循环与窗口预热行为和 `10 Hz / W=20` 一致。
2. **场景通过**
   - `scripts/audit_bag_quality.py` 对 `scene02_circle` 给出 `PASS`。
   - 4 架子机都满足 `min_duration >= 85 s`、`clean_ratio >= 95%`、`max |pos| <= 20 m`。
   - 不出现 flip-over、持续漂移、坠落或大面积样本失效。
3. **研究判断**
   - `c3a` 若闭环方向上不优于 `c1`，或两者都明显劣化于 `Teacher`，则不能继续扩展到 `S2'/S4/S5`，应先回到在线部署实现与特征/归一化一致性排查。
   - 只有当 `Teacher / c1 / c3a` 的闭环表现和当前离线排序大体一致时，才进入多次独立运行与后续场景扩展。

### 阶段 4：重新定义“论文实验完成”的标准

只有以下 4 类产物都齐了，才能说实验部分真正完成：

1. **主结果**
   - `S2` 下 8 方法统一评估表
   - 至少 5 次独立运行统计
2. **消融结果**
   - 特征、双向、注意力、部署机制四组消融
3. **鲁棒性/长航时**
   - `S4` 风扰
   - `S5` 长航时
   - `alpha` 和切换日志
4. **论文图表**
   - 真正来自评估结果的表和图，而不是 bag 诊断图

## 目录说明

- `configs/`: 控制器配置，当前仍有部分模板值待修正
- `data/`: raw bags、processed datasets、trained models
- `docs/`: 论文正文和实验设计 Word 原稿
- `matlab/`: 数据处理、训练和评估脚本
- `results/`: 当前 bag 诊断图和少量早期 run 记录
- `ros_ws/`: ROS 工作空间
- `scripts/`: rosbag 转换、诊断、可视化与复现实验入口

## 一句话结论

这个仓库现在已经完成了论文实验的“离线前半程”，但还没有进入“论文结果可交付”的状态。最关键的不是继续扩写计划，而是用稳定四机数据补齐 `BiLSTM-DA` 训练、在线部署和统一评估。
