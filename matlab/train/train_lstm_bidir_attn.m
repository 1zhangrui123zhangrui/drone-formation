function net = train_lstm_bidir_attn(dataDir, modelPath)
%TRAIN_LSTM_BIDIR_ATTN Train the BiLSTM-DA model.

thisDir = fileparts(mfilename('fullpath'));
matlabDir = fileparts(thisDir);
projectRoot = fileparts(matlabDir);

if nargin < 1; dataDir = fullfile(projectRoot, 'data', 'processed'); end
if nargin < 2; modelPath = fullfile(projectRoot, 'data', 'trained_models', 'c3_bidir_attn.mat'); end

dataDir = resolve_project_path(projectRoot, dataDir);
modelPath = resolve_project_path(projectRoot, modelPath);

fprintf('[C3] loading 15D datasets from %s\n', dataDir);
trRaw = load(fullfile(dataDir, 'dataset_15d_train.mat')); tr = trRaw.tr;
vaRaw = load(fullfile(dataDir, 'dataset_15d_val.mat')); va = vaRaw.va;

X_tr = seq2cell(tr.X);
X_va = seq2cell(va.X);
Y_tr = double(tr.Y);
Y_va = double(va.Y);

assert(isnumeric(Y_tr) && isreal(Y_tr) && all(isfinite(Y_tr(:))), ...
    '[C3] Training responses must be finite real numbers.');
assert(isnumeric(Y_va) && isreal(Y_va) && all(isfinite(Y_va(:))), ...
    '[C3] Validation responses must be finite real numbers.');

fprintf('[C3] train=%d  val=%d\n', numel(X_tr), numel(X_va));

HIDDEN = 64;
BILSTM_H = HIDDEN * 2;
ATTN_DIM = 64;

layers = [
    sequenceInputLayer(15, 'Name','input')
    attention_layer_feature(15, 'feature_attn')
    bilstmLayer(HIDDEN, 'OutputMode','sequence', 'Name','bilstm1')
    dropoutLayer(0.1, 'Name','drop1')
    bilstmLayer(HIDDEN, 'OutputMode','sequence', 'Name','bilstm2')
    attention_layer_temporal(BILSTM_H, ATTN_DIM, 'temporal_attn')
    fullyConnectedLayer(32, 'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(4, 'Name','output')
    regressionLayer('Name','reg')
];

opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.9, ...
    'LearnRateDropPeriod', 10, ...
    'MaxEpochs', 80, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {X_va, Y_va}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 8, ...
    'ExecutionEnvironment', 'cpu', ...
    'Plots', 'none', ...
    'Verbose', true, ...
    'VerboseFrequency', 100);

fprintf('[C3] training BiLSTM-DA...\n');
net = trainNetwork(X_tr, Y_tr, layers, opts);

mkdir_safe(fileparts(modelPath));
save(modelPath, 'net');
fprintf('[C3] saved to %s\n', modelPath);
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

function p = resolve_project_path(projectRoot, p)
if ~(ischar(p) || isstring(p))
    return;
end

p = char(p);
if startsWith(p, '\\') || startsWith(p, '/') || ~isempty(regexp(p, '^[A-Za-z]:[\\/]', 'once'))
    return;
end

p = fullfile(projectRoot, p);
end
