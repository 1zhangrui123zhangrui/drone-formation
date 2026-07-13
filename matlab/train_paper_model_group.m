function train_paper_model_group(modelKey)
%TRAIN_PAPER_MODEL_GROUP Train and audit one model across all paper seeds.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'matlab','train'));
valid={'c1_lstm9d','c2_lstm15d','c3a_bilstm','c3_bidir_attn'};
assert(any(strcmp(modelKey,valid)),'Unknown modelKey: %s',modelKey);
dataDir=fullfile(root,'data','processed');
modelRoot=fullfile(root,'data','trained_models','formal_5x5');
logDir=fullfile(root,'results','training','formal_5x5');
% Frozen from validation-only stage-1 base checkpoints (C1, seed 42).
seeds=[42 123 256 512 1024]; weights=[1 1 2 0.5];
for seed=seeds
    fprintf('\n=== %s seed=%d ===\n',modelKey,seed);
    train_paper_model(modelKey,seed,weights,dataDir,modelRoot,logDir,false);
end
audit_paper_checkpoints(modelKey,seeds,dataDir,modelRoot);
end
