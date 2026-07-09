%RUN_C3_TRAINING_ONLY Retrain only the proposed BiLSTM-DA model.
%
% Recommended from Windows MATLAB:
%   cd('\\wsl.localhost\Ubuntu-20.04\home\jiuyao\drone-formation-e2e')
%   run matlab/run_c3_training_only.m

clear; clc;
clear attention_layer_feature attention_layer_temporal;
rehash;

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);

ensure_path(fullfile(scriptDir, 'train'));

DATA_DIR = fullfile(projectRoot, 'data', 'processed');
MODEL_DIR = fullfile(projectRoot, 'data', 'trained_models');
LOG_DIR = fullfile(projectRoot, 'results', 'training');

if ~isfolder(LOG_DIR); mkdir(LOG_DIR); end

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
logPath = fullfile(LOG_DIR, ['matlab_train_c3_only_' timestamp '.log']);

diary(logPath);
cleanupObj = onCleanup(@() diary('off'));

fprintf('=== run_c3_training_only ===\n');
fprintf('Project root: %s\n', projectRoot);
fprintf('Data dir:     %s\n', DATA_DIR);
fprintf('Model dir:    %s\n', MODEL_DIR);
fprintf('Log file:     %s\n', logPath);

modelPath = fullfile(MODEL_DIR, 'c3_bidir_attn.mat');
train_lstm_bidir_attn(DATA_DIR, modelPath);

fprintf('\n=== verify C3 artifact ===\n');
s = load(modelPath);
required = {'net', 'info', 'metadata'};
for i = 1:numel(required)
    if ~isfield(s, required{i})
        error('[verify] c3_bidir_attn missing field "%s"', required{i});
    end
end

fprintf('c3_bidir_attn saved successfully with net/info/metadata.\n');
fprintf('=== run_c3_training_only DONE ===\n');

function ensure_path(p)
if ~contains([path pathsep], [char(p) pathsep])
    addpath(p);
end
end
