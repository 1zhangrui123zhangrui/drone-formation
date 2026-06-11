%RUN_FULL_EXPERIMENT  一键运行完整实验流程
%
% 前置条件（WSL2 端已完成）：
%   python3 scripts/bag_to_mat.py <each_bag> data/processed/raw/<scene>.mat
%   (已为所有 scene 生成 .mat 文件)
%
% 运行方式（Windows MATLAB 命令行）：
%   cd('\\wsl.localhost\Ubuntu-20.04\home\jiuyao\drone-formation-e2e')
%   run matlab/run_full_experiment.m
%
% 步骤：
%   1. 数据准备：合并所有 raw .mat → normalize → sliding window → train/val/test split
%   2. 训练四个模型：C1(9D-LSTM) C2(15D-LSTM) C3a(BiLSTM) C3(BiLSTM-DA)
%   3. 快速推理验证：用测试集第一个样本检查输出是否合理

clear; clc;
addpath('matlab/data_pipeline', 'matlab/train', 'matlab/evaluation');

ROOT = fileparts(mfilename('fullpath'));  % matlab/ dir
% 若从项目根目录运行则 ROOT 为空，dataDir 直接用相对路径
DATA_DIR  = 'data/processed';
MODEL_DIR = 'data/trained_models';
RAW_DIR   = fullfile(DATA_DIR, 'raw');

%% Step 1: 数据准备
if ~isfile(fullfile(DATA_DIR, 'dataset_15d_train.mat'))
    fprintf('=== Step 1: prepare_all_datasets ===\n');
    prepare_all_datasets(RAW_DIR, DATA_DIR);
else
    fprintf('=== Step 1: datasets already exist, skipping ===\n');
    tr = load(fullfile(DATA_DIR, 'dataset_15d_train.mat')); tr = tr.tr;
    fprintf('  train windows: %d  (W=%d, D=%d)\n', size(tr.X,3), size(tr.X,1), size(tr.X,2));
end

%% Step 2: 训练所有模型
fprintf('\n=== Step 2: train_all_models ===\n');
train_all_models(DATA_DIR, MODEL_DIR);

%% Step 3: 推理验证
fprintf('\n=== Step 3: inference check ===\n');
teRaw = load(fullfile(DATA_DIR, 'dataset_15d_test.mat')); te = teRaw.te;
x_test = squeeze(te.X(:,:,1))';  % [D=15, W=20]

% C1: 9D
if isfile(fullfile(MODEL_DIR, 'c1_lstm9d.mat'))
    m = load(fullfile(MODEL_DIR, 'c1_lstm9d.mat'));
    x9 = x_test(7:15, :);  % 9D: cols 7-15
    y_pred_c1 = predict(m.net, {x9});
    fprintf('C1 (9D-LSTM) pred:   vx=%.3f vy=%.3f vz=%.3f wz=%.3f\n', y_pred_c1);
end

% C3: BiLSTM-DA
if isfile(fullfile(MODEL_DIR, 'c3_bidir_attn.mat'))
    m = load(fullfile(MODEL_DIR, 'c3_bidir_attn.mat'));
    y_pred_c3 = predict(m.net, {x_test});
    fprintf('C3 (BiLSTM-DA) pred: vx=%.3f vy=%.3f vz=%.3f wz=%.3f\n', y_pred_c3);
end

fprintf('Ground truth:        vx=%.3f vy=%.3f vz=%.3f wz=%.3f\n', te.Y(1,:));

fprintf('\n=== run_full_experiment DONE ===\n');
fprintf('Next steps:\n');
fprintf('  1. Evaluate models on test sets: compute_metrics(p_actual, p_des)\n');
fprintf('  2. Run build_comparison_table(results) to generate Table 1\n');
fprintf('  3. Run ablation_table(r9,r15,rbi,rda) to generate Table 2\n');
fprintf('  4. Run plot_all_figures to generate IEEE figures\n');
