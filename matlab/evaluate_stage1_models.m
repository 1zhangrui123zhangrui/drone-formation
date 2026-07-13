%EVALUATE_STAGE1_MODELS Common test metrics for formal stage-1 checkpoints.
clear; clc; clear classes; rehash;
root=fileparts(fileparts(mfilename('fullpath'))); addpath(fullfile(root,'matlab','train'));
modelRoot=fullfile(root,'data','trained_models','formal_5x5'); dataDir=fullfile(root,'data','processed');
outDir=fullfile(root,'data','eval_results','formal_stage1'); if ~isfolder(outDir); mkdir(outDir); end
models={'c1_lstm9d','c2_lstm15d'}; seeds=[42 123 256 512 1024]; rows=struct([]); q=0;
for i=1:numel(models)
 dim=15; if strcmp(models{i},'c1_lstm9d'); dim=9; end
 D=load(fullfile(dataDir,sprintf('dataset_%dd_test.mat',dim))); te=D.te;
 X=cell(1,size(te.X,3)); for n=1:numel(X); X{n}=squeeze(te.X(:,:,n))'; end; Y=double(te.Y);
 for seed=seeds
  q=q+1; S=load(fullfile(modelRoot,models{i},sprintf('seed_%d',seed),'model.mat'),'net','metadata');
  assert(strcmp(S.metadata.training_stage,'formal_stage1_supervised') && ~S.metadata.synthetic_drift_augmentation);
  P=predict(S.net,X,'MiniBatchSize',64); if size(P,1)~=size(Y,1); P=P'; end; E=P-Y;
  rv=sqrt(mean(E.^2,1)); av=mean(abs(E),1);
  rows(q).model=string(models{i}); rows(q).seed=seed; rows(q).samples=size(Y,1);
  rows(q).rmse=sqrt(mean(E(:).^2)); rows(q).mae=mean(abs(E(:)));
  rows(q).rmse_vx=rv(1); rows(q).rmse_vy=rv(2); rows(q).rmse_vz=rv(3); rows(q).rmse_wz=rv(4);
  rows(q).mae_vx=av(1); rows(q).mae_vy=av(2); rows(q).mae_vz=av(3); rows(q).mae_wz=av(4);
  rows(q).best_val_loss=S.metadata.best_val_loss;
 end
end
T=struct2table(rows); writetable(T,fullfile(outDir,'per_seed_metrics.csv'));
summary=groupsummary(T,'model',{'mean','std'},{'rmse','mae','rmse_vx','rmse_vy','rmse_vz','rmse_wz'});
writetable(summary,fullfile(outDir,'summary.csv')); save(fullfile(outDir,'stage1_evaluation.mat'),'T','summary');
disp(T); disp(summary);
