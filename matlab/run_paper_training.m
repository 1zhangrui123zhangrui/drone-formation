%RUN_PAPER_TRAINING Canonical MATLAB training entry for the paper pipeline.
%
% Recommended from Windows MATLAB:
%   cd('\\wsl.localhost\Ubuntu-20.04\home\jiuyao\drone-formation-e2e')
%   run matlab/run_paper_training.m

clear; clc;
clear attention_layer_feature attention_layer_temporal;
rehash;

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);

ensure_path(fullfile(scriptDir, 'train'));

DATA_DIR = fullfile(projectRoot, 'data', 'processed');
MODEL_DIR = fullfile(projectRoot, 'data', 'trained_models');
LOG_DIR = fullfile(projectRoot, 'results', 'training');
MANIFEST_PATH = fullfile(DATA_DIR, 'dataset_build_manifest.json');

if ~isfolder(LOG_DIR); mkdir(LOG_DIR); end

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
logPath = fullfile(LOG_DIR, ['matlab_train_' timestamp '.log']);

diary(logPath);
cleanupObj = onCleanup(@() diary('off'));

fprintf('=== run_paper_training ===\n');
fprintf('Project root: %s\n', projectRoot);
fprintf('Data dir:     %s\n', DATA_DIR);
fprintf('Model dir:    %s\n', MODEL_DIR);
fprintf('Log file:     %s\n', logPath);

if isfile(MANIFEST_PATH)
    fprintf('Dataset manifest found: %s\n', MANIFEST_PATH);
else
    warning('Dataset manifest not found: %s', MANIFEST_PATH);
end

fprintf('\n=== Step 1: train_all_models ===\n');
train_all_models(DATA_DIR, MODEL_DIR);

fprintf('\n=== Step 2: verify saved artifacts ===\n');
verify_artifact(fullfile(MODEL_DIR, 'c1_lstm9d.mat'), 'c1_lstm9d');
verify_artifact(fullfile(MODEL_DIR, 'c2_lstm15d.mat'), 'c2_lstm15d');
verify_artifact(fullfile(MODEL_DIR, 'c3a_bilstm.mat'), 'c3a_bilstm');
verify_artifact(fullfile(MODEL_DIR, 'c3_bidir_attn.mat'), 'c3_bidir_attn');

fprintf('\n=== run_paper_training DONE ===\n');
fprintf('Next recommended step: inspect validation curves and then proceed to repeated evaluation runs.\n');

function verify_artifact(modelPath, modelName)
fprintf('[verify] %s\n', modelName);
if ~isfile(modelPath)
    error('[verify] missing model file: %s', modelPath);
end

s = load(modelPath);
required = {'net', 'info', 'metadata'};
for i = 1:numel(required)
    if ~isfield(s, required{i})
        error('[verify] %s missing field "%s"', modelName, required{i});
    end
end

fprintf('  file exists: yes\n');
fprintf('  has net/info/metadata: yes\n');

if isstruct(s.metadata)
    if isfield(s.metadata, 'train_windows')
        fprintf('  train_windows: %d\n', s.metadata.train_windows);
    end
    if isfield(s.metadata, 'val_windows')
        fprintf('  val_windows: %d\n', s.metadata.val_windows);
    end
    if isfield(s.metadata, 'input_dim')
        fprintf('  input_dim: %d\n', s.metadata.input_dim);
    end
end

if isstruct(s.info)
    if isfield(s.info, 'FinalValidationLoss')
        fprintf('  FinalValidationLoss: %.6f\n', s.info.FinalValidationLoss);
    end
    if isfield(s.info, 'TrainingLoss')
        fprintf('  epochs logged: %d\n', numel(s.info.TrainingLoss));
    end
end
end

function ensure_path(p)
if ~contains([path pathsep], [char(p) pathsep])
    addpath(p);
end
end
