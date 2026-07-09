function net = train_lstm_15d(dataDir, modelPath)
%TRAIN_LSTM_15D Train the 15D single-direction LSTM baseline.

thisDir = fileparts(mfilename('fullpath'));
matlabDir = fileparts(thisDir);
projectRoot = fileparts(matlabDir);

if nargin < 1; dataDir = fullfile(projectRoot, 'data', 'processed'); end
if nargin < 2; modelPath = fullfile(projectRoot, 'data', 'trained_models', 'c2_lstm15d.mat'); end

dataDir = resolve_project_path(projectRoot, dataDir);
modelPath = resolve_project_path(projectRoot, modelPath);

fprintf('[C2] loading 15D datasets from %s\n', dataDir);
trRaw = load(fullfile(dataDir, 'dataset_15d_train.mat')); tr = trRaw.tr;
vaRaw = load(fullfile(dataDir, 'dataset_15d_val.mat')); va = vaRaw.va;

X_tr = seq2cell(tr.X);
X_va = seq2cell(va.X);
Y_tr = double(tr.Y);
Y_va = double(va.Y);

assert(isnumeric(Y_tr) && isreal(Y_tr) && all(isfinite(Y_tr(:))), ...
    '[C2] Training responses must be finite real numbers.');
assert(isnumeric(Y_va) && isreal(Y_va) && all(isfinite(Y_va(:))), ...
    '[C2] Validation responses must be finite real numbers.');

fprintf('[C2] train=%d  val=%d  (each window: 15x%d)\n', ...
    numel(X_tr), numel(X_va), size(tr.X, 1));

layers = [
    sequenceInputLayer(15, 'Name','input')
    lstmLayer(192, 'OutputMode','last', 'Name','lstm')
    fullyConnectedLayer(64, 'Name','fc1')
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

fprintf('[C2] training 15D-LSTM...\n');
trainFile = 'dataset_15d_train.mat';
valFile = 'dataset_15d_val.mat';
[net, info] = trainNetwork(X_tr, Y_tr, layers, opts);

metadata = build_training_metadata('c2_lstm15d', dataDir, trainFile, valFile, tr, va);
metadata.training_options = struct( ...
    'initial_learn_rate', 1e-3, ...
    'learn_rate_drop_factor', 0.9, ...
    'learn_rate_drop_period', 10, ...
    'max_epochs', 80, ...
    'mini_batch_size', 64, ...
    'validation_patience', 8, ...
    'execution_environment', 'cpu');
save_training_artifact(modelPath, net, info, metadata);
fprintf('[C2] saved to %s\n', modelPath);
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
