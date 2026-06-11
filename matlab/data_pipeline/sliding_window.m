function windows = sliding_window(sequence, windowSize, stride)
%SLIDING_WINDOW Slice time-series samples into fixed-length windows.

if nargin < 2
    windowSize = 20;
end
if nargin < 3
    stride = 1;
end

numRows = size(sequence, 1);
if numRows < windowSize
    windows = zeros(windowSize, size(sequence, 2), 0);
    return;
end

startIndices = 1:stride:(numRows - windowSize + 1);
windows = zeros(windowSize, size(sequence, 2), numel(startIndices));
for idx = 1:numel(startIndices)
    startIdx = startIndices(idx);
    windows(:, :, idx) = sequence(startIdx:startIdx + windowSize - 1, :);
end
end
