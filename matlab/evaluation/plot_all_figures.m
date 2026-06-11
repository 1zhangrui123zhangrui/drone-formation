function plot_all_figures(resultsDir, outDir)
%PLOT_ALL_FIGURES  生成 IEEE 风格论文图表
%
% 用法（Windows MATLAB）：
%   addpath('matlab/evaluation')
%   plot_all_figures('data/eval_results','figures')
%
% 生成内容：
%   figures/fig_rmse_comparison.pdf  — 7方法对比柱状图
%   figures/fig_ablation.pdf         — 消融折线图
%   figures/fig_trajectory_s2.pdf    — S2 XY 轨迹图

if nargin < 1; resultsDir = 'data/eval_results'; end
if nargin < 2; outDir = 'figures'; end

if ~isfolder(outDir); mkdir(outDir); end

% IEEE 图表风格
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultAxesFontSize', 9);
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultTextFontSize', 9);
set(groot,'defaultLineLineWidth', 1.0);

%% 尝试加载评估结果
summaryFile = fullfile(resultsDir, 'summary.mat');
if ~isfile(summaryFile)
    warning('[plot] summary.mat not found in %s — run evaluate_all_models first', resultsDir);
    return
end
S = load(summaryFile);  % expects S.results (nested struct)

%% Fig 1: RMSE 对比柱状图（5场景 × 4方法）
try
    methods = fieldnames(S.results);
    scenes  = {'S1','S2','S2p','S3','S4','S5'};
    valid_scenes = fieldnames(S.results.(methods{1}));
    rmse_mat = nan(numel(methods), numel(valid_scenes));
    for mi = 1:numel(methods)
        for si = 1:numel(valid_scenes)
            try
                rmse_mat(mi,si) = S.results.(methods{mi}).(valid_scenes{si}).rmse;
            catch; end
        end
    end

    fig1 = figure('Position',[100 100 700 280],'PaperPositionMode','auto');
    bar(rmse_mat');
    set(gca,'XTickLabel', valid_scenes);
    ylabel('3D RMSE (m)');
    legend(strrep(methods,'_',' '), 'Location','northeast','FontSize',7);
    grid on; box on;
    title('Formation Control RMSE Comparison');
    saveas(fig1, fullfile(outDir,'fig_rmse_comparison.pdf'));
    fprintf('[plot] saved fig_rmse_comparison.pdf\n');
catch e
    warning('[plot] Fig 1 failed: %s', e.message);
end

%% Fig 2: 消融折线图
try
    if isfield(S, 'ablation')
        ab = S.ablation;
        configs = {'9D-LSTM','15D-LSTM','BiLSTM','BiLSTM-DA'};
        vals = [ab.c1.rmse, ab.c2.rmse, ab.c3a.rmse, ab.c3.rmse];

        fig2 = figure('Position',[100 100 350 250],'PaperPositionMode','auto');
        plot(1:4, vals, '-o','Color',[0.2 0.4 0.8],'MarkerSize',6,'MarkerFaceColor',[0.2 0.4 0.8]);
        set(gca,'XTick',1:4,'XTickLabel',configs,'XTickLabelRotation',15);
        ylabel('3D RMSE (m)');
        title('Ablation Study (S2 Circle)');
        grid on; box on;
        saveas(fig2, fullfile(outDir,'fig_ablation.pdf'));
        fprintf('[plot] saved fig_ablation.pdf\n');
    end
catch e
    warning('[plot] Fig 2 failed: %s', e.message);
end

fprintf('[plot_all_figures] done. Output in: %s\n', outDir);
end
