function ablation_table(results_9d, results_15d, results_bilstm, results_da)
%ABLATION_TABLE  打印消融研究结果（论文 Table 2，S2 圆形场景）
%
% 每个参数为 struct with field .rmse (3D RMSE in meters, scalar)
%
% 用法:
%   r9  = struct('rmse', 2.154);
%   r15 = struct('rmse', 0.712);
%   rbi = struct('rmse', 0.519);
%   rda = struct('rmse', 0.300);
%   ablation_table(r9, r15, rbi, rda)

configs = {'9D-LSTM (baseline)', '15D-LSTM (+v_des/p_des)', ...
           '15D-BiLSTM (+bidirectional)', '15D-BiLSTM-DA (+dual attention)'};
rmse_vals = [results_9d.rmse, results_15d.rmse, results_bilstm.rmse, results_da.rmse];

fprintf('\n===== Ablation Study — S2 Circle Scene (3D RMSE m) =====\n');
fprintf('%-36s  RMSE(m)   vs prev   vs baseline\n', 'Configuration');
fprintf('%s\n', repmat('-', 70, 1));

for i = 1:4
    vs_prev     = '';
    vs_baseline = '';
    if i > 1
        delta_prev = (rmse_vals(i) - rmse_vals(i-1)) / rmse_vals(i-1) * 100;
        vs_prev = sprintf('%+.1f%%', delta_prev);
    end
    if i > 1
        delta_base = (rmse_vals(i) - rmse_vals(1)) / rmse_vals(1) * 100;
        vs_baseline = sprintf('%+.1f%%', delta_base);
    end
    fprintf('%-36s  %.4f    %-9s %s\n', configs{i}, rmse_vals(i), vs_prev, vs_baseline);
end
fprintf('%s\n', repmat('-', 70, 1));

% 关键消融数值（对应论文 §5.3）
fprintf('9D→15D improvement:    %.1f%%\n', (rmse_vals(1)-rmse_vals(2))/rmse_vals(1)*100);
fprintf('uni→bi improvement:    %.1f%%\n', (rmse_vals(2)-rmse_vals(3))/rmse_vals(2)*100);
fprintf('+DA improvement:       %.1f%%\n', (rmse_vals(3)-rmse_vals(4))/rmse_vals(3)*100);
fprintf('Total vs 9D-LSTM:      %.1f%%\n', (rmse_vals(1)-rmse_vals(4))/rmse_vals(1)*100);
end
