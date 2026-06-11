function prepare_all_datasets(rawMatDir, outDir)
%PREPARE_ALL_DATASETS  合并所有 raw .mat → normalize → sliding_window → split → 保存
%
% 用法（Windows MATLAB）：
%   cd('\\wsl.localhost\Ubuntu-20.04\home\jiuyao\drone-formation-e2e')
%   addpath('matlab/data_pipeline')
%   prepare_all_datasets('data/processed/raw', 'data/processed')
%
% rawMatDir: Python 转出的 .mat 目录, 默认 "data/processed/raw"
% outDir:    输出目录, 默认 "data/processed"
%
% 输出文件:
%   <outDir>/dataset_9d_{train,val,test}.mat
%   <outDir>/dataset_15d_{train,val,test}.mat
%   <outDir>/norm_stats_9d.mat
%   <outDir>/norm_stats_15d.mat

if nargin < 1; rawMatDir = "data/processed/raw"; end
if nargin < 2; outDir    = "data/processed"; end

W      = 20;             % 滑窗长度（论文§4.3）
STRIDE = 1;              % 步长
SPLIT  = [0.7, 0.2, 0.1]; % train/val/test

% ---- 收集所有 raw .mat ----
files = dir(fullfile(rawMatDir, "*.mat"));
assert(~isempty(files), 'No .mat files found in %s', rawMatDir);

all_X15 = [];
all_Y4  = [];

for fi = 1:numel(files)
    fp = fullfile(rawMatDir, files(fi).name);
    fprintf('[prepare] loading %s\n', files(fi).name);
    s  = bag_to_samples(fp);
    all_X15 = [all_X15; s.X];  %#ok<AGROW>
    all_Y4  = [all_Y4;  s.Y];  %#ok<AGROW>
end

fprintf('[prepare] total raw samples: %d\n', size(all_X15,1));

% ---- 归一化（在滑窗前计算统计量）----
[X15_norm, stats15] = normalize(all_X15);

% 9D 输入: [p_actual(3), v_actual(3), u_teacher_xyz(3)] = cols 7-15
X9_raw = all_X15(:, 7:15);
[X9_norm, stats9] = normalize(X9_raw);

% ---- 滑窗 ----
% sliding_window 返回 [W, D, N_windows]
W15 = sliding_window(X15_norm, W, STRIDE);  % [20, 15, N]
W9  = sliding_window(X9_norm,  W, STRIDE);  % [20, 9,  N]

% 标签取窗口最后一帧（窗口起始索引为1..N，对应原始样本 W..W+N-1）
N   = size(W15, 3);
Y_w = all_Y4(W : W + N - 1, :);  % [N, 4]

fprintf('[prepare] windows: %d (W=%d, stride=%d)\n', N, W, STRIDE);

% ---- 7:2:1 shuffle 分割 ----
rng(42);
idx = randperm(N);
n_train = round(SPLIT(1) * N);
n_val   = round(SPLIT(2) * N);

i_tr = idx(1:n_train);
i_va = idx(n_train+1 : n_train+n_val);
i_te = idx(n_train+n_val+1 : end);

% ---- 保存 15D ----
save_split(outDir, 'dataset_15d', W15, Y_w, i_tr, i_va, i_te);
save(fullfile(outDir, 'norm_stats_15d.mat'), 'stats15');

% ---- 保存 9D ----
save_split(outDir, 'dataset_9d', W9, Y_w, i_tr, i_va, i_te);
save(fullfile(outDir, 'norm_stats_9d.mat'), 'stats9');

fprintf('[prepare] done. Files in: %s\n', outDir);
end

% ----------------------------------------------------------------
function save_split(outDir, prefix, W_all, Y_all, i_tr, i_va, i_te)
% W_all: [W, D, N];  Y_all: [N, 4]
if ~isfolder(outDir); mkdir(outDir); end

tr.X = W_all(:,:,i_tr); tr.Y = Y_all(i_tr,:);
va.X = W_all(:,:,i_va); va.Y = Y_all(i_va,:);
te.X = W_all(:,:,i_te); te.Y = Y_all(i_te,:);

save(fullfile(outDir, [prefix '_train.mat']), 'tr', '-v7.3');
save(fullfile(outDir, [prefix '_val.mat']),   'va', '-v7.3');
save(fullfile(outDir, [prefix '_test.mat']),  'te', '-v7.3');
fprintf('[prepare] %s: train=%d  val=%d  test=%d\n', ...
    prefix, numel(i_tr), numel(i_va), numel(i_te));
end
