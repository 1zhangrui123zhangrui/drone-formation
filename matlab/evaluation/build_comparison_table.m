function T = build_comparison_table(resultsStruct)
%BUILD_COMPARISON_TABLE  生成 7方法 × 5场景 RMSE 对比表（论文 Table 1）
%
% resultsStruct: nested struct, fields = method names, sub-fields = scene names
%   resultsStruct.Teacher_PD.S1.rmse  = 0.082
%   resultsStruct.BiLSTM_DA.S2.rmse   = 0.30
%
% 用法示例（手动填入后调用）:
%   r.Teacher_PD.S1.rmse = 0.082; ...
%   T = build_comparison_table(r);
%
% 输出: MATLAB table，行=方法，列=场景，值=RMSE

methods = fieldnames(resultsStruct);
scenes  = fieldnames(resultsStruct.(methods{1}));

rmse_mat = nan(numel(methods), numel(scenes));
for mi = 1:numel(methods)
    for si = 1:numel(scenes)
        try
            m = resultsStruct.(methods{mi}).(scenes{si});
            if isfield(m, 'rmse')
                rmse_mat(mi, si) = m.rmse;
            end
        catch
        end
    end
end

T = array2table(rmse_mat, ...
    'RowNames',    strrep(methods, '_', ' '), ...
    'VariableNames', scenes);

fprintf('\n===== 3D Position RMSE Comparison Table (m) =====\n');
disp(T);
fprintf('Best per scene (excluding Teacher_PD):\n');
[best_vals, best_idx] = min(rmse_mat(2:end,:), [], 1);
for si = 1:numel(scenes)
    fprintf('  %s: %.4f (%s)\n', scenes{si}, best_vals(si), methods{best_idx(si)+1});
end
end
