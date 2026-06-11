function net = train_lstm_bidir_attn(dataDir, modelPath)
%TRAIN_LSTM_BIDIR_ATTN  训练 BiLSTM-DA（论文核心方法§4）
%
% 用法（Windows MATLAB）：
%   addpath('matlab/data_pipeline','matlab/train')
%   net = train_lstm_bidir_attn('data/processed','data/trained_models/c3_bidir_attn.mat')
%
% 网络结构（论文§4.3-4.4）:
%   FA(15) → BiLSTM(64,seq) → Dropout(0.1) → BiLSTM(64,seq) → TA(128,64)
%           → FC(32,ReLU) → FC(4,linear)
%
% 注意：自定义层（attention_layer_feature, attention_layer_temporal）
% 使用 predict() 实现，不需要 backward()，由 trainNetwork 自动微分处理。

if nargin < 1; dataDir   = 'data/processed'; end
if nargin < 2; modelPath = 'data/trained_models/c3_bidir_attn.mat'; end

addpath('matlab/data_pipeline');
addpath('matlab/train');

%% 加载数据
fprintf('[C3] loading 15D datasets from %s\n', dataDir);
trRaw = load(fullfile(dataDir, 'dataset_15d_train.mat')); tr = trRaw.tr;
vaRaw = load(fullfile(dataDir, 'dataset_15d_val.mat'));   va = vaRaw.va;

X_tr = seq2cell(tr.X);
X_va = seq2cell(va.X);
Y_tr = num2cell(tr.Y', 1);
Y_va = num2cell(va.Y', 1);

fprintf('[C3] train=%d  val=%d\n', numel(X_tr), numel(X_va));

%% 网络参数
HIDDEN   = 64;
BILSTM_H = HIDDEN * 2;  % BiLSTM 双向输出维度 = 128
ATTN_DIM = 64;

%% 网络定义
layers = [
    sequenceInputLayer(15, 'Name','input')
    attention_layer_feature(15, 'feature_attn')
    bilstmLayer(HIDDEN, 'OutputMode','sequence', 'Name','bilstm1')
    dropoutLayer(0.1, 'Name','drop1')
    bilstmLayer(HIDDEN, 'OutputMode','sequence', 'Name','bilstm2')
    attention_layer_temporal(BILSTM_H, ATTN_DIM, 'temporal_attn')
    fullyConnectedLayer(32, 'Name','fc1')
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
fprintf('[C3] training BiLSTM-DA...\n');
net = trainNetwork(X_tr, Y_tr, layers, opts);

%% 保存
mkdir_safe(fileparts(modelPath));
save(modelPath, 'net');
fprintf('[C3] saved → %s\n', modelPath);
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
