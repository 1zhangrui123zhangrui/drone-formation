%TUNE_CHANNEL_WEIGHTS Validation-only grid search on C1, seed 42.
clear; clc; clear classes; rehash;
root=fileparts(fileparts(mfilename('fullpath'))); addpath(fullfile(root,'matlab','train'));
dataDir=fullfile(root,'data','processed'); modelRoot=fullfile(root,'data','trained_models','weight_tuning');
logDir=fullfile(root,'results','training','weight_tuning');
az=[1 2 4]; ar=[0.5 1]; rows=struct([]); k=0;
for i=1:numel(az)
 for j=1:numel(ar)
  k=k+1; w=[1 1 az(i) ar(j)]; key=sprintf('c1_lstm9d_az_%g_ar_%g',az(i),ar(j));
  a=train_paper_model('c1_lstm9d',42,w,dataDir,fullfile(modelRoot,key),logDir,false);
  rows(k).alpha_xy=1; rows(k).alpha_z=az(i); rows(k).alpha_r=ar(j);
  rows(k).best_val_loss=a.metadata.best_val_loss; rows(k).artifact=string(a.path);
 end
end
T=struct2table(rows); T=sortrows(T,'best_val_loss');
outDir=fullfile(root,'data','eval_results','weight_tuning'); if ~isfolder(outDir); mkdir(outDir); end
writetable(T,fullfile(outDir,'weight_tuning.csv')); save(fullfile(outDir,'weight_tuning.mat'),'T'); disp(T);
