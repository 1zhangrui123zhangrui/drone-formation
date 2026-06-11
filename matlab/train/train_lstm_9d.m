function net = train_lstm_9d(dataDir, modelPath)
%TRAIN_LSTM_9D  训练 9D 单向 LSTM 控制器（论文对比方法 C1）
%
% 用法（Windows MATLAB）：
%   addpath('matlab/data_pipeline','matlab/train')
%   net = train_lstm_9d('data/processed','data/trained_models/c1_lstm9d.mat')
%
% 输入维度: 9 = [p_actual(3), v_actual(3), u_teacher_xyz(3)]
% 网络结构: LSTM(128) → FC(64,ReLU) → FC(4,linear)

if nargin < 1; dataDir   = 'data/processed'; end
if nargin < 2; modelPath = 'data/trained_models/c1_lstm9d.mat'; end

addpath('matlab/data_pipeline');

%% 加载数据
fprintf('[C1] loading 9D datasets from %s\n', dataDir);
trRaw = load(fullfile(dataDir, 'dataset_9d_train.mat')); tr = trRaw.tr;
vaRaw = load(fullfile(dataDir, 'dataset_9d_val.mat'));   va = vaRaw.va;

% 转换: [W, D, N] → cell array {[D, W]}_N  (MATLAB LSTM 期望 [D, T])
X_tr = seq2cell(tr.X);
X_va = seq2cell(va.X);
Y_tr = num2cell(tr.Y', 1);  % {[4, 1]}_N 每个样本输出一帧
Y_va = num2cell(va.Y', 1);

fprintf('[C1] train=%d  val=%d  (each window: 9x%d)\n', ...
    numel(X_tr), numel(X_va), size(tr.X, 1));

%% 网络定义
layers = [
    sequenceInputLayer(9,  'Name','input')
    lstmLayer(128, 'OutputMode','last', 'Name','lstm')
    fullyConnectedLayer(64, 'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(4,  'Name','output')
    regressionLayer('Name','reg')
];

opts = trainingOptions('adam', ...
    'InitialLearnRate',     1e-3, ...
    'LearnRateSchedule',    'piecewise', ...
    'LearnRateDropFactor',  0.9, ...
    'LearnRateDropPeriod',  10, ...
    'MaxEpochs',            80, ...
    'MiniBatchSize',        64, ...
    'Shuffle',              'every-epoch', ...
    'ValidationData',       {X_va, Y_va}, ...
    'ValidationFrequency',  50, ...
    'ValidationPatience',   8, ...
    'Plots',                'none', ...
    'Verbose',              true, ...
    'VerboseFrequency',     100);

%% 训练
fprintf('[C1] training 9D-LSTM...\n');
net = trainNetwork(X_tr, Y_tr, layers, opts);

%% 保存
mkdir_safe(fileparts(modelPath));
save(modelPath, 'net');
fprintf('[C1] saved → %s\n', modelPath);
end

function cells = seq2cell(W_array)
% W_array: [W, D, N] → cell {[D, W]}_N
W = size(W_array, 1);
D = size(W_array, 2);
N = size(W_array, 3);
cells = cell(1, N);
for i = 1:N
    % [W, D] → transpose → [D, W]
    cells{i} = squeeze(W_array(:,:,i))';
end
end

function mkdir_safe(d)
if ~isempty(d) && ~isfolder(d); mkdir(d); end
end
