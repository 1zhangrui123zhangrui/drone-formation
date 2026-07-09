function summary = evaluate_offline_models(dataDir, modelDir, outDir)
%EVALUATE_OFFLINE_MODELS Evaluate trained models on canonical test datasets.
%
% This is an offline sanity-check stage after training and before closed-loop
% paper evaluation. It measures command regression error on the held-out test
% sets and saves a reproducible summary.

thisDir = fileparts(mfilename('fullpath'));
matlabDir = fileparts(thisDir);
projectRoot = fileparts(matlabDir);

if nargin < 1; dataDir = fullfile(projectRoot, 'data', 'processed'); end
if nargin < 2; modelDir = fullfile(projectRoot, 'data', 'trained_models'); end
if nargin < 3; outDir = fullfile(projectRoot, 'data', 'eval_results', 'offline'); end

dataDir = resolve_project_path(projectRoot, dataDir);
modelDir = resolve_project_path(projectRoot, modelDir);
outDir = resolve_project_path(projectRoot, outDir);

if ~isfolder(outDir); mkdir(outDir); end

summary = struct();
summary.created_at = datestr(now, 'yyyy-mm-dd HH:MM:SS');
summary.data_dir = dataDir;
summary.model_dir = modelDir;

summary.c1 = evaluate_one( ...
    fullfile(modelDir, 'c1_lstm9d.mat'), ...
    fullfile(dataDir, 'dataset_9d_test.mat'), ...
    'te', 'c1_lstm9d');
summary.c2 = evaluate_one( ...
    fullfile(modelDir, 'c2_lstm15d.mat'), ...
    fullfile(dataDir, 'dataset_15d_test.mat'), ...
    'te', 'c2_lstm15d');
summary.c3a = evaluate_one( ...
    fullfile(modelDir, 'c3a_bilstm.mat'), ...
    fullfile(dataDir, 'dataset_15d_test.mat'), ...
    'te', 'c3a_bilstm');
summary.c3 = evaluate_one( ...
    fullfile(modelDir, 'c3_bidir_attn.mat'), ...
    fullfile(dataDir, 'dataset_15d_test.mat'), ...
    'te', 'c3_bidir_attn');

save(fullfile(outDir, 'offline_summary.mat'), 'summary');
write_summary_csv(summary, fullfile(outDir, 'offline_summary.csv'));

fprintf('\n===== Offline Test-Set Summary =====\n');
disp(struct2table(flatten_summary(summary), 'AsArray', true));
fprintf('[evaluate_offline_models] saved to %s\n', outDir);
end

function result = evaluate_one(modelPath, dataPath, splitName, modelName)
if ~isfile(modelPath)
    error('[offline eval] missing model: %s', modelPath);
end
if ~isfile(dataPath)
    error('[offline eval] missing dataset: %s', dataPath);
end

S = load(modelPath);
D = load(dataPath);

if ~isfield(S, 'net')
    error('[offline eval] %s missing net field', modelName);
end
if ~isfield(D, splitName)
    error('[offline eval] dataset %s missing split "%s"', dataPath, splitName);
end

split = D.(splitName);
X = seq2cell(split.X);
Y = double(split.Y);
Y_pred = predict(S.net, X, 'MiniBatchSize', 64);
Y_pred = normalize_prediction_shape(Y_pred, size(Y, 1));

err = Y_pred - Y;
rmse_vec = sqrt(mean(err.^2, 1));
mae_vec = mean(abs(err), 1);

result = struct();
result.model_name = modelName;
result.samples = size(Y, 1);
result.output_dim = size(Y, 2);
result.rmse_mean = sqrt(mean(err(:).^2));
result.mae_mean = mean(abs(err(:)));
result.rmse_vx = rmse_vec(1);
result.rmse_vy = rmse_vec(2);
result.rmse_vz = rmse_vec(3);
result.rmse_wz = rmse_vec(4);
result.mae_vx = mae_vec(1);
result.mae_vy = mae_vec(2);
result.mae_vz = mae_vec(3);
result.mae_wz = mae_vec(4);
end

function Y_pred = normalize_prediction_shape(Y_pred, expectedN)
if iscell(Y_pred)
    error('[offline eval] expected numeric prediction array, got cell output.');
end

sz = size(Y_pred);
if ismatrix(Y_pred) && sz(1) == expectedN
    return;
end
if ismatrix(Y_pred) && sz(2) == expectedN
    Y_pred = Y_pred';
    return;
end

error('[offline eval] unexpected prediction shape: [%s]', num2str(sz));
end

function cells = seq2cell(W_array)
N = size(W_array, 3);
cells = cell(1, N);
for i = 1:N
    cells{i} = squeeze(W_array(:,:,i))';
end
end

function rows = flatten_summary(summary)
rows = struct([]);
names = {'c1','c2','c3a','c3'};
for i = 1:numel(names)
    s = summary.(names{i});
    rows(i).model = s.model_name;
    rows(i).samples = s.samples;
    rows(i).rmse_mean = s.rmse_mean;
    rows(i).mae_mean = s.mae_mean;
    rows(i).rmse_vx = s.rmse_vx;
    rows(i).rmse_vy = s.rmse_vy;
    rows(i).rmse_vz = s.rmse_vz;
    rows(i).rmse_wz = s.rmse_wz;
end
end

function write_summary_csv(summary, outPath)
rows = flatten_summary(summary);
T = struct2table(rows, 'AsArray', true);
writetable(T, outPath);
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
