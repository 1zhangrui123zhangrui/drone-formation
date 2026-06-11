function metrics = compute_metrics(p_actual, p_des, hard_switch_count)
%COMPUTE_METRICS  计算3D RMSE、MAE 和追踪误差统计（论文 Table 1 指标）
%
% 用法:
%   metrics = compute_metrics(p_actual, p_des)
%   metrics = compute_metrics(p_actual, p_des, hard_switch_count)
%
% 输入:
%   p_actual [N, 3]  实际位置序列 (x,y,z)
%   p_des    [N, 3]  期望位置序列 (x,y,z)
%   hard_switch_count  标量，编队切换次数（可选，S3专用）
%
% 输出 metrics struct:
%   .rmse              3D 欧氏距离 RMSE (m)
%   .mae               3D 欧氏距离 MAE  (m)
%   .max_err           最大误差 (m)
%   .std_err           误差标准差 (m)
%   .hard_switch_count 切换次数

if nargin < 3; hard_switch_count = NaN; end

% 3D 欧氏距离序列
err3d = sqrt(sum((p_actual - p_des).^2, 2));  % [N,1]

metrics.rmse              = sqrt(mean(err3d.^2));
metrics.mae               = mean(err3d);
metrics.max_err           = max(err3d);
metrics.std_err           = std(err3d);
metrics.hard_switch_count = hard_switch_count;
end
