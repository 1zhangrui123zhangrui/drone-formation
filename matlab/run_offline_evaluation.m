%RUN_OFFLINE_EVALUATION Evaluate all trained models on canonical test sets.
%
% Recommended from Windows MATLAB:
%   cd('\\wsl.localhost\Ubuntu-20.04\home\jiuyao\drone-formation-e2e')
%   run matlab/run_offline_evaluation.m

clear; clc;
clear attention_layer_feature attention_layer_temporal;
rehash;

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);

ensure_path(fullfile(scriptDir, 'evaluation'));
ensure_path(fullfile(scriptDir, 'train'));

DATA_DIR = fullfile(projectRoot, 'data', 'processed');
MODEL_DIR = fullfile(projectRoot, 'data', 'trained_models');
OUT_DIR = fullfile(projectRoot, 'data', 'eval_results', 'offline');

summary = evaluate_offline_models(DATA_DIR, MODEL_DIR, OUT_DIR); %#ok<NASGU>

fprintf('\n=== run_offline_evaluation DONE ===\n');
fprintf('Saved offline evaluation results to: %s\n', OUT_DIR);

function ensure_path(p)
if ~contains([path pathsep], [char(p) pathsep])
    addpath(p);
end
end
