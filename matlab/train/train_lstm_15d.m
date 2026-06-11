function net = train_lstm_15d(dataDir, modelPath)
%TRAIN_LSTM_15D  训练 15D 单向 LSTM 控制器（论文对比方法 C2）
%
% 用法（Windows MATLAB）：
%   addpath('matlab/data_pipeline','matlab/train')
%   net = train_lstm_15d('data/processed','data/trained_models/c2_lstm15d.mat')
%
% 输入维度: 15 = [p_des(3), v_des(3), p_actual(3), v_actual(3), u_teacher_xyz(3)]
% 网络结构: LSTM(192) → FC(64,ReLU) → FC(4,linear)

if nargin < 1; dataDir   = 'data/processed'; end
if nargin < 2; modelPath = 'data/trained_models/c2_lstm15d.mat'; end

addpath('matlab/data_pipeline');

%% 加载数据
fprintf('[C2] loading 15D datasets from %s\n', dataDir);
trRaw = load(fullfile(dataDir, 'dataset_15d_train.mat')); tr = trRaw.tr;
vaRaw = load(fullfile(dataDir, 'dataset_15d_val.mat'));   va = vaRaw.va;

X_tr = seq2cell(tr.X);
X_va = seq2cell(va.X);
Y_tr = num2cell(tr.Y', 1);
Y_va = num2cell(va.Y', 1);

fprintf('[C2] train=%d  val=%d  (each window: 15x%d)\n', ...
    numel(X_tr), numel(X_va), size(tr.X, 1));

%% 网络定义
layers = [
    sequenceInputLayer(15, 'Name','input')
    lstmLayer(192, 'OutputMode','last', 'Name','lstm')
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
fprintf('[C2] training 15D-LSTM...\n');
net = trainNetwork(X_tr, Y_tr, layers, opts);

%% 保存
mkdir_safe(fileparts(modelPath));
save(modelPath, 'net');
fprintf('[C2] saved → %s\n', modelPath);
end

function cells = seq2cell(W_array)
N = size(W_array, 3);
cells = cell(1, N);
for i = 1:N
    cells{i} = squeeze(W_array(:,:,i))';
end
end

function mkdir_safe(d)
if ~isempty(d) && ~isfolder(d); mkdir(d); end
end
