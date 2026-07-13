%RUN_PAPER_TRAINING_PROTOCOL Strict ordered five-seed paper training.
clear; clc; clear classes; rehash;
root=fileparts(fileparts(mfilename('fullpath'))); addpath(fullfile(root,'matlab','train'));
dataDir=fullfile(root,'data','processed'); modelRoot=fullfile(root,'data','trained_models','formal_5x5');
logDir=fullfile(root,'results','training','formal_5x5');
models={'c1_lstm9d','c2_lstm15d','c3a_bilstm','c3_bidir_attn'}; seeds=[42 123 256 512 1024];
% Frozen from validation-only stage-1 base checkpoints on formal_5x5_v2.
% Synthetic fine-tuned checkpoints are explicitly excluded from selection.
weights=[1 1 2 0.5];
for i=1:numel(models)
    for j=1:numel(seeds)
        fprintf('\n=== %s seed=%d ===\n',models{i},seeds(j));
        train_paper_model(models{i},seeds(j),weights,dataDir,modelRoot,logDir,false);
    end
    audit_paper_checkpoints(models{i},seeds,dataDir,modelRoot);
end
