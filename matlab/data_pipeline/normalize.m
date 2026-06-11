function [normalized, stats] = normalize(data, stats)
%NORMALIZE z-score 归一化，stats 可复用于测试集
% data: [N, D] float
% stats: (可选) struct with .mean [1,D] and .std [1,D]

if nargin < 2 || isempty(stats)
    stats.mean = mean(data, 1, 'omitnan');
    stats.std  = std(data, 0, 1, 'omitnan');
    stats.std(stats.std < 1e-8) = 1;  % 防零除
end

normalized = (data - stats.mean) ./ stats.std;
end
