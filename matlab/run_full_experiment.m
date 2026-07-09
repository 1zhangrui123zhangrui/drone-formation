%RUN_FULL_EXPERIMENT Run dataset prep, training, and a quick inference check.
%
% Example from Windows MATLAB:
%   cd('\\wsl.localhost\Ubuntu-20.04\home\jiuyao\drone-formation-e2e')
%   run matlab/run_full_experiment.m

clear; clc;

scriptDir = fileparts(mfilename('fullpath'));  % .../matlab
projectRoot = fileparts(scriptDir);

ensure_path(fullfile(scriptDir, 'data_pipeline'));
ensure_path(fullfile(scriptDir, 'train'));
ensure_path(fullfile(scriptDir, 'evaluation'));

DATA_DIR = fullfile(projectRoot, 'data', 'processed');
MODEL_DIR = fullfile(projectRoot, 'data', 'trained_models');
RAW_DIR = fullfile(DATA_DIR, 'raw');

%% Step 1: dataset preparation
if ~isfile(fullfile(DATA_DIR, 'dataset_15d_train.mat'))
    fprintf('=== Step 1: prepare_all_datasets ===\n');
    prepare_all_datasets(RAW_DIR, DATA_DIR);
else
    fprintf('=== Step 1: datasets already exist, skipping ===\n');
    tr = load(fullfile(DATA_DIR, 'dataset_15d_train.mat')); tr = tr.tr;
    fprintf('  train windows: %d  (W=%d, D=%d)\n', size(tr.X,3), size(tr.X,1), size(tr.X,2));
end

%% Step 2: training
fprintf('\n=== Step 2: train_all_models ===\n');
train_all_models(DATA_DIR, MODEL_DIR);

%% Step 3: inference check
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

function ensure_path(p)
if ~contains([path pathsep], [char(p) pathsep])
    addpath(p);
end
end
