function artifact = train_paper_model(modelKey, seed, channelWeights, dataDir, modelRoot, logDir, smoke)
%TRAIN_PAPER_MODEL Formal stage-1 supervised paper training only.
% Synthetic feature perturbation is deliberately excluded: perturbing
% normalized states without recomputing physically consistent Teacher labels
% is not an admissible paper-training protocol.
if nargin < 7; smoke = false; end
thisDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(thisDir));
if nargin < 4 || isempty(dataDir); dataDir = fullfile(projectRoot,'data','processed'); end
if nargin < 5 || isempty(modelRoot); modelRoot = fullfile(projectRoot,'data','trained_models','formal_5x5'); end
if nargin < 6 || isempty(logDir); logDir = fullfile(projectRoot,'results','training','formal_5x5'); end
if nargin < 3 || isempty(channelWeights); channelWeights = [1 1 2 1]; end

rng(seed,'twister');
[inputDim, trainFile, valFile] = model_spec(modelKey);
trS = load(fullfile(dataDir,trainFile)); vaS = load(fullfile(dataDir,valFile));
tr = trS.tr; va = vaS.va;
if smoke
    tr.X = tr.X(:,:,1:min(256,size(tr.X,3))); tr.Y = tr.Y(1:size(tr.X,3),:);
    va.X = va.X(:,:,1:min(128,size(va.X,3))); va.Y = va.Y(1:size(va.X,3),:);
end
Xtr = seq2cell(tr.X); Xva = seq2cell(va.X); Ytr = double(tr.Y); Yva = double(va.Y);
layers = build_layers(modelKey,inputDim,channelWeights);

if ~isfolder(logDir); mkdir(logDir); end
logPath=fullfile(logDir,sprintf('%s_seed_%d_%s.log',modelKey,seed,datestr(now,'yyyymmdd_HHMMSS')));
diary(logPath); diaryCleanup=onCleanup(@() diary('off'));
fprintf('model=%s seed=%d smoke=%d weights=[%s]\n',modelKey,seed,smoke,num2str(channelWeights));

maxEpochs = ternary(smoke,1,50); patience = ternary(smoke,1,5);
validationFrequency=max(1,ceil(numel(Xtr)/64));
opts = trainingOptions('adam','InitialLearnRate',1e-3,'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.9,'LearnRateDropPeriod',10,'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',64,'Shuffle','every-epoch','ValidationData',{Xva,Yva}, ...
    'ValidationFrequency',validationFrequency,'ValidationPatience',patience,'OutputNetwork','best-validation-loss', ...
    'ExecutionEnvironment','cpu','Plots','none','Verbose',true,'VerboseFrequency',100);

runDir = fullfile(modelRoot,modelKey,sprintf('seed_%d',seed)); if ~isfolder(runDir); mkdir(runDir); end
t0 = tic; [baseNet,baseInfo] = trainNetwork(Xtr,Ytr,layers,opts); baseSeconds = toc(t0);
metadata = make_metadata(modelKey,seed,channelWeights,dataDir,trainFile,valFile,tr,va,baseSeconds,smoke);
metadata.best_val_loss=best_val(baseInfo); metadata.best_val_iteration=best_val_iteration(baseInfo);
metadata.log_path=logPath; metadata.training_stage='formal_stage1_supervised';
metadata.synthetic_drift_augmentation=false;
metadata.artifact_created_at = datestr(now,'yyyy-mm-dd HH:MM:SS');
save(fullfile(runDir,'base.mat'),'baseNet','baseInfo','metadata','-v7.3');
net=baseNet; info=baseInfo;
artifactPath = fullfile(runDir,'model.mat');
save(artifactPath,'net','info','metadata','-v7.3');
artifact = struct('path',artifactPath,'metadata',metadata);
clear diaryCleanup;
end

function layers = build_layers(key,d,w)
switch key
    case {'c1_lstm9d','c2_lstm15d'}
        layers=[sequenceInputLayer(d,'Name','input'); lstmLayer(64,'OutputMode','sequence','Name','lstm1'); ...
            dropoutLayer(0.1,'Name','drop1'); lstmLayer(64,'OutputMode','last','Name','lstm2'); ...
            fullyConnectedLayer(32,'Name','fc1'); reluLayer('Name','relu1'); ...
            fullyConnectedLayer(4,'Name','output'); weighted_mse_regression_layer(w,'weighted_mse')];
    case 'c3a_bilstm'
        layers=[sequenceInputLayer(d,'Name','input'); bilstmLayer(64,'OutputMode','sequence','Name','bilstm1'); ...
            dropoutLayer(0.1,'Name','drop1'); bilstmLayer(64,'OutputMode','last','Name','bilstm2'); ...
            fullyConnectedLayer(32,'Name','fc1'); reluLayer('Name','relu1'); ...
            fullyConnectedLayer(4,'Name','output'); weighted_mse_regression_layer(w,'weighted_mse')];
    case 'c3_bidir_attn'
        layers=[sequenceInputLayer(d,'Name','input'); attention_layer_feature(d,'feature_attn'); ...
            bilstmLayer(64,'OutputMode','sequence','Name','bilstm1'); dropoutLayer(0.1,'Name','drop1'); ...
            bilstmLayer(64,'OutputMode','sequence','Name','bilstm2'); attention_layer_temporal(128,64,'temporal_attn'); ...
            fullyConnectedLayer(32,'Name','fc1'); reluLayer('Name','relu1'); ...
            fullyConnectedLayer(4,'Name','output'); weighted_mse_regression_layer(w,'weighted_mse')];
    otherwise; error('Unknown model key: %s',key);
end
end

function [d,tr,va]=model_spec(k)
if strcmp(k,'c1_lstm9d'); d=9; tr='dataset_9d_train.mat'; va='dataset_9d_val.mat';
else; d=15; tr='dataset_15d_train.mat'; va='dataset_15d_val.mat'; end
end
function c=seq2cell(x); c=cell(1,size(x,3)); for i=1:size(x,3); c{i}=squeeze(x(:,:,i))'; end; end
function m=make_metadata(k,s,w,dataDir,trf,vaf,tr,va,seconds,smoke)
manifest=fullfile(dataDir,'dataset_build_manifest.json'); m=struct(); m.model_name=k; m.random_seed=s;
m.channel_weights=w; m.data_manifest_path=manifest; m.data_manifest_sha256=file_sha256(manifest);
m.git_commit=git_head(fileparts(fileparts(dataDir))); m.train_file=trf; m.val_file=vaf;
m.train_windows=size(tr.X,3); m.val_windows=size(va.X,3); m.window_length=size(tr.X,1);
m.input_dim=size(tr.X,2); m.output_dim=size(tr.Y,2); m.base_training_seconds=seconds; m.smoke=smoke;
m.protocol=struct('optimizer','adam','initial_lr',1e-3,'drop_factor',0.9,'drop_period',10, ...
 'batch_size',64,'max_epochs',50,'validation_patience_epochs',5, ...
 'validation_frequency_policy','once per epoch','execution_environment','cpu');
end
function h=file_sha256(p)
md=java.security.MessageDigest.getInstance('SHA-256'); fid=fopen(p,'rb'); assert(fid>=0,'Cannot open %s',p);
cleanup=onCleanup(@() fclose(fid)); while ~feof(fid); b=fread(fid,1024*1024,'*uint8'); md.update(b); end
h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'))',1,[])); clear cleanup;
end
function h=git_head(root)
head=strtrim(fileread(fullfile(root,'.git','HEAD')));
if startsWith(head,'ref: '); h=strtrim(fileread(fullfile(root,'.git',head(6:end)))); else; h=head; end
end
function y=ternary(c,a,b); if c; y=a; else; y=b; end; end
function v=best_val(info); x=double(info.ValidationLoss(:)); x=x(isfinite(x)); assert(~isempty(x)); v=min(x); end
function i=best_val_iteration(info); x=double(info.ValidationLoss(:)); ok=find(isfinite(x)); [~,j]=min(x(ok)); i=ok(j); end
