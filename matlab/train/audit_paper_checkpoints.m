function audit_paper_checkpoints(modelKey,seeds,dataDir,modelRoot)
manifestHash=file_sha256(fullfile(dataDir,'dataset_build_manifest.json'));
expectedDim=15; if strcmp(modelKey,'c1_lstm9d'); expectedDim=9; end
for s=seeds
 p=fullfile(modelRoot,modelKey,sprintf('seed_%d',s),'model.mat'); assert(isfile(p),'Missing %s',p);
 a=load(p,'net','info','metadata'); assert(isfield(a,'net')&&isfield(a,'info')&&isfield(a,'metadata'));
 assert(a.metadata.random_seed==s && a.metadata.input_dim==expectedDim && a.metadata.output_dim==4);
 assert(strcmp(a.metadata.data_manifest_sha256,manifestHash));
 assert(isfield(a.metadata,'training_stage') && strcmp(a.metadata.training_stage,'formal_stage1_supervised'));
 assert(isfield(a.metadata,'synthetic_drift_augmentation') && ~a.metadata.synthetic_drift_augmentation);
 assert(isequal(double(a.metadata.channel_weights(:))',[1 1 2 0.5]));
 vl=double(a.info.ValidationLoss(:)); assert(any(isfinite(vl)) && all(isfinite(vl(isfinite(vl)))));
end
fprintf('[audit] %s: %d checkpoints PASS\n',modelKey,numel(seeds));
end
function h=file_sha256(p)
md=java.security.MessageDigest.getInstance('SHA-256'); fid=fopen(p,'rb'); assert(fid>=0);
cleanup=onCleanup(@() fclose(fid)); while ~feof(fid); md.update(fread(fid,1024*1024,'*uint8')); end
h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'))',1,[])); clear cleanup;
end
