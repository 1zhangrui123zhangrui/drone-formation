function train_all_models(dataDir, modelDir)
%TRAIN_ALL_MODELS Train all model variants used in the MATLAB experiments.

thisDir = fileparts(mfilename('fullpath'));
matlabDir = fileparts(thisDir);
projectRoot = fileparts(matlabDir);

if nargin < 1; dataDir = fullfile(projectRoot, 'data', 'processed'); end
if nargin < 2; modelDir = fullfile(projectRoot, 'data', 'trained_models'); end

dataDir = resolve_project_path(projectRoot, dataDir);
modelDir = resolve_project_path(projectRoot, modelDir);

ensure_path(fullfile(matlabDir, 'data_pipeline'));
ensure_path(fullfile(matlabDir, 'train'));

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

function p = resolve_project_path(projectRoot, p)
if ~(ischar(p) || isstring(p))
    return;
end

p = char(p);
if startsWith(p, '\\') || startsWith(p, '/') || ~isempty(regexp(p, '^[A-Za-z]:[\\/]', 'once'))
    return;
end

p = fullfile(projectRoot, p);
end

function ensure_path(p)
if ~contains([path pathsep], [char(p) pathsep])
    addpath(p);
end
end
