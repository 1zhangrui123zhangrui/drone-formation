# drone-formation

端到端无人机编队控制研究仓库模板，覆盖 ROS/Gazebo 仿真、MATLAB 数据处理与训练、在线控制部署、实验复现与结果汇总。

## 目标

- 统一管理教师控制器、端到端网络控制器与对比基线。
- 固化实验配置，支持场景切换、消融实验与论文图表生成。
- 将原始 rosbag、处理样本、训练模型与结果分层管理，方便后续接入 DVC 或 Git LFS。

## 一键复现建议流程

1. 创建 Conda 环境：
   `conda env create -f environment.yml`
2. 构建容器环境：
   `docker compose -f docker/docker-compose.yml build`
3. 启动核心仿真与控制：
   `bash scripts/reproduce_main_results.sh`
4. 运行消融实验：
   `bash scripts/reproduce_ablation.sh`
5. 生成论文图表：
   `bash scripts/generate_paper_figures.sh`

## 目录说明

- `configs/`: 控制器与实验超参数配置。
- `ros_ws/`: ROS 工作空间，负责 Gazebo 场景、教师命令和在线控制节点。
- `matlab/`: 数据处理、训练、基线控制器和结果评估。
- `data/`: 原始数据、处理样本和模型权重，建议接入 DVC 或 Git LFS。
- `results/`: 每次实验独立存放日志、图像、指标和配置快照。
- `scripts/`: 一键复现实验入口脚本。
- `docs/`: 复现实验的补充说明。
- `tests/`: MATLAB 侧单元测试。

## 当前状态

当前仓库已完成基础骨架初始化，所有脚本与函数均为可扩展模板。将真实模型、rosbag、MATLAB 训练逻辑和 ROS topic 映射填入后即可进入联调。
