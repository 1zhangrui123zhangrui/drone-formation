function prepare_all_datasets(rawMatDir, outDir)
%PREPARE_ALL_DATASETS Build normalized train/val/test datasets from raw MAT files.
%
% Example:
%   cd('\\wsl.localhost\Ubuntu-20.04\home\jiuyao\drone-formation-e2e')
%   addpath('matlab/data_pipeline')
%   prepare_all_datasets('data/processed/raw', 'data/processed')

thisDir = fileparts(mfilename('fullpath'));
matlabDir = fileparts(thisDir);
projectRoot = fileparts(matlabDir);

if nargin < 1; rawMatDir = fullfile(projectRoot, 'data', 'processed', 'raw'); end
if nargin < 2; outDir = fullfile(projectRoot, 'data', 'processed'); end

rawMatDir = resolve_project_path(projectRoot, rawMatDir);
outDir = resolve_project_path(projectRoot, outDir);

W = 20;
STRIDE = 1;
SPLIT = [0.7, 0.2, 0.1];

files = dir(fullfile(rawMatDir, '*.mat'));
assert(~isempty(files), 'No .mat files found in %s', rawMatDir);

all_X15 = [];
all_Y4 = [];

for fi = 1:numel(files)
    fp = fullfile(rawMatDir, files(fi).name);
    fprintf('[prepare] loading %s\n', files(fi).name);
    s = bag_to_samples(fp);
    all_X15 = [all_X15; s.X]; %#ok<AGROW>
    all_Y4 = [all_Y4; s.Y]; %#ok<AGROW>
end

fprintf('[prepare] total raw samples: %d\n', size(all_X15, 1));

[X15_norm, stats15] = normalize(all_X15);

% 9D input = [p_actual(3), v_actual(3), u_teacher_xyz(3)]
X9_raw = all_X15(:, 7:15);
[X9_norm, stats9] = normalize(X9_raw);

W15 = sliding_window(X15_norm, W, STRIDE);  % [W, D, N]
W9 = sliding_window(X9_norm, W, STRIDE);    % [W, D, N]

N = size(W15, 3);
Y_w = all_Y4(W : W + N - 1, :);  % [N, 4]

fprintf('[prepare] windows: %d (W=%d, stride=%d)\n', N, W, STRIDE);

rng(42);
idx = randperm(N);
n_train = round(SPLIT(1) * N);
n_val = round(SPLIT(2) * N);

i_tr = idx(1:n_train);
i_va = idx(n_train + 1 : n_train + n_val);
i_te = idx(n_train + n_val + 1 : end);

save_split(outDir, 'dataset_15d', W15, Y_w, i_tr, i_va, i_te);
save(fullfile(outDir, 'norm_stats_15d.mat'), 'stats15');

save_split(outDir, 'dataset_9d', W9, Y_w, i_tr, i_va, i_te);
save(fullfile(outDir, 'norm_stats_9d.mat'), 'stats9');

fprintf('[prepare] done. Files in: %s\n', outDir);
end

function save_split(outDir, prefix, W_all, Y_all, i_tr, i_va, i_te)
if ~isfolder(outDir); mkdir(outDir); end

tr.X = W_all(:,:,i_tr); tr.Y = Y_all(i_tr,:);
va.X = W_all(:,:,i_va); va.Y = Y_all(i_va,:);
te.X = W_all(:,:,i_te); te.Y = Y_all(i_te,:);

save(fullfile(outDir, [prefix '_train.mat']), 'tr', '-v7.3');
save(fullfile(outDir, [prefix '_val.mat']), 'va', '-v7.3');
save(fullfile(outDir, [prefix '_test.mat']), 'te', '-v7.3');
fprintf('[prepare] %s: train=%d  val=%d  test=%d\n', ...
    prefix, numel(i_tr), numel(i_va), numel(i_te));
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
