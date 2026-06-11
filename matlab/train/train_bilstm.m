function net = train_bilstm(dataDir, modelPath)
%TRAIN_BILSTM  训练 15D BiLSTM（无注意力）控制器（消融对比 C3a）
%
% 用法（Windows MATLAB）：
%   addpath('matlab/data_pipeline','matlab/train')
%   net = train_bilstm('data/processed','data/trained_models/c3a_bilstm.mat')
%
% 输入维度: 15
% 网络结构: BiLSTM(64) → Dropout(0.1) → BiLSTM(64,last) → FC(32,ReLU) → FC(4)

if nargin < 1; dataDir   = 'data/processed'; end
if nargin < 2; modelPath = 'data/trained_models/c3a_bilstm.mat'; end

addpath('matlab/data_pipeline');

%% 加载数据
fprintf('[C3a] loading 15D datasets from %s\n', dataDir);
trRaw = load(fullfile(dataDir, 'dataset_15d_train.mat')); tr = trRaw.tr;
vaRaw = load(fullfile(dataDir, 'dataset_15d_val.mat'));   va = vaRaw.va;

X_tr = seq2cell(tr.X);
X_va = seq2cell(va.X);
Y_tr = num2cell(tr.Y', 1);
Y_va = num2cell(va.Y', 1);

fprintf('[C3a] train=%d  val=%d\n', numel(X_tr), numel(X_va));

%% 网络定义（双向，两层叠加）
layers = [
    sequenceInputLayer(15, 'Name','input')
    bilstmLayer(64, 'OutputMode','sequence', 'Name','bilstm1')
    dropoutLayer(0.1, 'Name','drop1')
    bilstmLayer(64, 'OutputMode','last', 'Name','bilstm2')
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
fprintf('[C3a] training 15D-BiLSTM (no attention)...\n');
net = trainNetwork(X_tr, Y_tr, layers, opts);

%% 保存
mkdir_safe(fileparts(modelPath));
save(modelPath, 'net');
fprintf('[C3a] saved → %s\n', modelPath);
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
