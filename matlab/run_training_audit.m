%RUN_TRAINING_AUDIT Audit the newly trained checkpoints before closed-loop use.
%
% Recommended from Windows MATLAB:
%   cd('\\wsl.localhost\Ubuntu-20.04\home\jiuyao\drone-formation-e2e')
%   run matlab/run_training_audit.m

clear; clc;
clear attention_layer_feature attention_layer_temporal;
rehash;

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);

ensure_path(fullfile(scriptDir, 'evaluation'));
ensure_path(fullfile(scriptDir, 'train'));

MODEL_DIR = fullfile(projectRoot, 'data', 'trained_models');
OUT_DIR = fullfile(projectRoot, 'data', 'eval_results', 'training_audit');

audit = run_training_audit_local(MODEL_DIR, OUT_DIR); %#ok<NASGU>

fprintf('\n=== run_training_audit DONE ===\n');
fprintf('Saved training audit results to: %s\n', OUT_DIR);

function audit = run_training_audit_local(modelDir, outDir)
if nargin < 1; modelDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'trained_models'); end
if nargin < 2; outDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'eval_results', 'training_audit'); end

projectRoot = fileparts(mfilename('fullpath'));
projectRoot = fileparts(projectRoot);

modelDir = resolve_project_path(projectRoot, modelDir);
outDir = resolve_project_path(projectRoot, outDir);

if ~isfolder(outDir); mkdir(outDir); end

specs = {
    'c1',  'c1_lstm9d.mat';
    'c2',  'c2_lstm15d.mat';
    'c3a', 'c3a_bilstm.mat';
    'c3',  'c3_bidir_attn.mat';
};

audit = struct();
audit.created_at = datestr(now, 'yyyy-mm-dd HH:MM:SS');
audit.model_dir = modelDir;

rows = [];
for i = 1:size(specs, 1)
    key = specs{i, 1};
    fileName = specs{i, 2};
    modelPath = fullfile(modelDir, fileName);
    row = inspect_one(modelPath, key);
    if isempty(rows)
        rows = row;
    else
        rows(end+1, 1) = row; %#ok<AGROW>
    end
    audit.(key) = row;
end

T = struct2table(rows, 'AsArray', true);
save(fullfile(outDir, 'training_audit.mat'), 'audit', 'T');
writetable(T, fullfile(outDir, 'training_audit.csv'));

fprintf('\n===== Training Audit Summary =====\n');
disp(T);
fprintf('[run_training_audit] saved to %s\n', outDir);
end

function row = inspect_one(modelPath, key)
if ~isfile(modelPath)
    error('[training audit] missing model file: %s', modelPath);
end

S = load(modelPath);

row = struct();
row.key = key;
row.file = string(modelPath);
row.has_net = isfield(S, 'net');
row.has_info = isfield(S, 'info');
row.has_metadata = isfield(S, 'metadata');

row.train_windows = NaN;
row.val_windows = NaN;
row.input_dim = NaN;
row.output_dim = NaN;
row.best_val_loss = NaN;
row.final_val_loss = NaN;
row.final_train_loss = NaN;
row.epochs_logged = NaN;
row.best_val_epoch = NaN;

if isfield(S, 'metadata')
    md = S.metadata;
    row.train_windows = get_field_or_nan(md, 'train_windows');
    row.val_windows = get_field_or_nan(md, 'val_windows');
    row.input_dim = get_field_or_nan(md, 'input_dim');
    row.output_dim = get_field_or_nan(md, 'output_dim');
end

if isfield(S, 'info')
    info = S.info;
    if isfield(info, 'ValidationLoss') && ~isempty(info.ValidationLoss)
        valLoss = double(info.ValidationLoss(:));
        validMask = isfinite(valLoss);
        if any(validMask)
            validIdx = find(validMask);
            validLoss = valLoss(validMask);
            [row.best_val_loss, localIdx] = min(validLoss);
            row.best_val_epoch = validIdx(localIdx);
            row.final_val_loss = validLoss(end);
        end
    end
    if isfield(info, 'TrainingLoss') && ~isempty(info.TrainingLoss)
        trLoss = double(info.TrainingLoss(:));
        validMask = isfinite(trLoss);
        if any(validMask)
            validLoss = trLoss(validMask);
            row.final_train_loss = validLoss(end);
            row.epochs_logged = numel(trLoss);
        end
    end
end
end

function value = get_field_or_nan(s, fieldName)
if isfield(s, fieldName)
    value = double(s.(fieldName));
else
    value = NaN;
end
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
