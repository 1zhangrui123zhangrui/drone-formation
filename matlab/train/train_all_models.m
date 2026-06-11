function train_all_models(dataDir, modelDir)
%TRAIN_ALL_MODELS  批量训练所有 LSTM 对比方法
%
% 用法（Windows MATLAB）：
%   cd('\\wsl.localhost\Ubuntu-20.04\home\jiuyao\drone-formation-e2e')
%   addpath('matlab/data_pipeline','matlab/train')
%   train_all_models('data/processed','data/trained_models')
%
% 训练顺序：C1(9D-LSTM) → C2(15D-LSTM) → C3a(BiLSTM) → C3(BiLSTM-DA)

if nargin < 1; dataDir  = 'data/processed'; end
if nargin < 2; modelDir = 'data/trained_models'; end

addpath('matlab/data_pipeline');
addpath('matlab/train');

if ~isfolder(modelDir); mkdir(modelDir); end

t0 = tic;
fprintf('\n========== C1: 9D-LSTM ==========\n');
train_lstm_9d(dataDir, fullfile(modelDir, 'c1_lstm9d.mat'));

fprintf('\n========== C2: 15D-LSTM ==========\n');
train_lstm_15d(dataDir, fullfile(modelDir, 'c2_lstm15d.mat'));

fprintf('\n========== C3a: 15D-BiLSTM (no attention) ==========\n');
train_bilstm(dataDir, fullfile(modelDir, 'c3a_bilstm.mat'));

fprintf('\n========== C3: BiLSTM-DA (proposed) ==========\n');
train_lstm_bidir_attn(dataDir, fullfile(modelDir, 'c3_bidir_attn.mat'));

fprintf('\n[train_all_models] DONE. Total elapsed: %.1fs\n', toc(t0));
fprintf('Models saved in: %s\n', modelDir);
end
