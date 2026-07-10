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
GAP_THRESHOLD = 0.15;

files = dir(fullfile(rawMatDir, '*.mat'));
assert(~isempty(files), 'No .mat files found in %s', rawMatDir);

segments = struct([]);
segCount = 0;
trainRows15 = {};
trainRows9 = {};

for fi = 1:numel(files)
    fp = fullfile(rawMatDir, files(fi).name);
    fprintf('[prepare] loading %s\n', files(fi).name);
    s = bag_to_samples(fp);
    fileSegs = split_samples_by_drone_and_gap(s, erase(files(fi).name, '.mat'), GAP_THRESHOLD);
    for si = 1:numel(fileSegs)
        segCount = segCount + 1;
        segments(segCount) = fileSegs(si); %#ok<AGROW>
    end
end

totalRaw = 0;
for si = 1:numel(segments)
    totalRaw = totalRaw + size(segments(si).X15, 1);
    segments(si).split_info = split_raw_ranges(size(segments(si).X15, 1), W, SPLIT);
    [tr0, tr1] = deal(segments(si).split_info.train(1), segments(si).split_info.train(2));
    if tr1 > tr0
        trainRows15{end+1} = segments(si).X15(tr0:tr1, :); %#ok<AGROW>
        trainRows9{end+1} = segments(si).X15(tr0:tr1, 7:15); %#ok<AGROW>
    end
end

fprintf('[prepare] total raw samples across contiguous segments: %d\n', totalRaw);
assert(~isempty(trainRows15), 'No training rows available after boundary-safe splitting.');

allTrain15 = vertcat(trainRows15{:});
allTrain9 = vertcat(trainRows9{:});
[~, stats15] = normalize(allTrain15);
[~, stats9] = normalize(allTrain9);

tr15_w = {}; va15_w = {}; te15_w = {};
tr9_w = {};  va9_w = {};  te9_w = {};
trY = {};    vaY = {};    teY = {};

for si = 1:numel(segments)
    seg = segments(si);
    [tr0, tr1] = deal(seg.split_info.train(1), seg.split_info.train(2));
    [va0, va1] = deal(seg.split_info.val(1), seg.split_info.val(2));
    [te0, te1] = deal(seg.split_info.test(1), seg.split_info.test(2));

    [w15, y15] = chunk_to_windows(seg.X15(tr0:tr1, :), seg.Y4(tr0:tr1, :), stats15, W, STRIDE, 15);
    [w9, y9] = chunk_to_windows(seg.X15(tr0:tr1, 7:15), seg.Y4(tr0:tr1, :), stats9, W, STRIDE, 9);
    tr15_w{end+1} = w15; tr9_w{end+1} = w9; trY{end+1} = y15; %#ok<AGROW>
    assert(size(y15,1) == size(y9,1), 'Train label mismatch in segment %s', seg.segment_id);

    [w15, y15] = chunk_to_windows(seg.X15(va0:va1, :), seg.Y4(va0:va1, :), stats15, W, STRIDE, 15);
    [w9, y9] = chunk_to_windows(seg.X15(va0:va1, 7:15), seg.Y4(va0:va1, :), stats9, W, STRIDE, 9);
    va15_w{end+1} = w15; va9_w{end+1} = w9; vaY{end+1} = y15; %#ok<AGROW>
    assert(size(y15,1) == size(y9,1), 'Val label mismatch in segment %s', seg.segment_id);

    [w15, y15] = chunk_to_windows(seg.X15(te0:te1, :), seg.Y4(te0:te1, :), stats15, W, STRIDE, 15);
    [w9, y9] = chunk_to_windows(seg.X15(te0:te1, 7:15), seg.Y4(te0:te1, :), stats9, W, STRIDE, 9);
    te15_w{end+1} = w15; te9_w{end+1} = w9; teY{end+1} = y15; %#ok<AGROW>
    assert(size(y15,1) == size(y9,1), 'Test label mismatch in segment %s', seg.segment_id);
end

tr.X = concat_windows(tr15_w, W, 15); tr.Y = concat_labels(trY);
va.X = concat_windows(va15_w, W, 15); va.Y = concat_labels(vaY);
te.X = concat_windows(te15_w, W, 15); te.Y = concat_labels(teY);
save(fullfile(outDir, 'dataset_15d_train.mat'), 'tr', '-v7.3');
save(fullfile(outDir, 'dataset_15d_val.mat'), 'va', '-v7.3');
save(fullfile(outDir, 'dataset_15d_test.mat'), 'te', '-v7.3');
fprintf('[prepare] dataset_15d: train=%d  val=%d  test=%d\n', size(tr.X,3), size(va.X,3), size(te.X,3));
save(fullfile(outDir, 'norm_stats_15d.mat'), 'stats15');

tr.X = concat_windows(tr9_w, W, 9); tr.Y = concat_labels(trY);
va.X = concat_windows(va9_w, W, 9); va.Y = concat_labels(vaY);
te.X = concat_windows(te9_w, W, 9); te.Y = concat_labels(teY);
save(fullfile(outDir, 'dataset_9d_train.mat'), 'tr', '-v7.3');
save(fullfile(outDir, 'dataset_9d_val.mat'), 'va', '-v7.3');
save(fullfile(outDir, 'dataset_9d_test.mat'), 'te', '-v7.3');
fprintf('[prepare] dataset_9d: train=%d  val=%d  test=%d\n', size(tr.X,3), size(va.X,3), size(te.X,3));
save(fullfile(outDir, 'norm_stats_9d.mat'), 'stats9');

fprintf('[prepare] done. Files in: %s\n', outDir);
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

function segments = split_samples_by_drone_and_gap(samples, sceneName, gapThreshold)
segments = struct([]);
droneIds = unique(samples.drone_id(:))';
segCount = 0;

for did = droneIds
    mask = samples.drone_id(:) == did;
    t = samples.t_vec(mask);
    X = samples.X(mask, :);
    Y = samples.Y(mask, :);

    [t, order] = sort(t, 'ascend');
    X = X(order, :);
    Y = Y(order, :);

    dt = diff(t);
    boundaries = [1; find(~isfinite(dt) | dt <= 0 | dt > gapThreshold) + 1; numel(t) + 1];
    for bi = 1:numel(boundaries)-1
        i0 = boundaries(bi);
        i1 = boundaries(bi+1) - 1;
        if i1 < i0
            continue;
        end
        segCount = segCount + 1;
        segments(segCount).scene = sceneName; %#ok<AGROW>
        segments(segCount).drone_id = did;
        segments(segCount).segment_id = sprintf('%s_d%d_seg%02d', sceneName, did, bi - 1);
        segments(segCount).t_vec = t(i0:i1);
        segments(segCount).X15 = X(i0:i1, :);
        segments(segCount).Y4 = Y(i0:i1, :);
    end
end
end

function splitInfo = split_raw_ranges(numRows, windowSize, splitRatio)
guard = windowSize - 1;
effective = numRows - 2 * guard;

if effective <= 0
    splitInfo.train = [1, numRows];
    splitInfo.val = [1, 0];
    splitInfo.test = [1, 0];
    splitInfo.guard = guard;
    splitInfo.effective_rows = effective;
    return;
end

nTrain = round(splitRatio(1) * effective);
nVal = round(splitRatio(2) * effective);
if nTrain + nVal > effective
    nVal = max(0, effective - nTrain);
end

tr0 = 1;
tr1 = nTrain;
va0 = tr1 + guard + 1;
va1 = va0 + nVal - 1;
te0 = va1 + guard + 1;
te1 = numRows;

splitInfo.train = [tr0, tr1];
splitInfo.val = [va0, va1];
splitInfo.test = [te0, te1];
splitInfo.guard = guard;
splitInfo.effective_rows = effective;
end

function [windows, labels] = chunk_to_windows(Xraw, Yraw, stats, windowSize, stride, inputDim)
if isempty(Xraw) || size(Xraw, 1) < windowSize
    windows = zeros(windowSize, inputDim, 0, 'single');
    labels = zeros(0, size(Yraw, 2), 'single');
    return;
end

Xnorm = normalize(Xraw, stats);
starts = 1:stride:(size(Xnorm, 1) - windowSize + 1);
nw = numel(starts);
windows = zeros(windowSize, inputDim, nw, 'single');
labels = zeros(nw, size(Yraw, 2), 'single');
for wi = 1:nw
    s = starts(wi);
    windows(:,:,wi) = single(Xnorm(s:s+windowSize-1, :));
    labels(wi,:) = single(Yraw(s+windowSize-1, :));
end
end

function W = concat_windows(chunks, windowSize, inputDim)
valid = ~cellfun(@isempty, chunks);
chunks = chunks(valid);
if isempty(chunks)
    W = zeros(windowSize, inputDim, 0, 'single');
    return;
end
W = cat(3, chunks{:});
end

function Y = concat_labels(chunks)
valid = ~cellfun(@isempty, chunks);
chunks = chunks(valid);
if isempty(chunks)
    Y = zeros(0, 4, 'single');
    return;
end
Y = cat(1, chunks{:});
end
