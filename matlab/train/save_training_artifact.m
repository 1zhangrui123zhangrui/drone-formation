function save_training_artifact(modelPath, net, info, metadata)
%SAVE_TRAINING_ARTIFACT Save trained network with training history and metadata.

mkdir_safe(fileparts(modelPath));
save(modelPath, 'net', 'info', 'metadata', '-v7.3');
fprintf('[save_training_artifact] saved net/info/metadata to %s\n', modelPath);
end

function mkdir_safe(d)
if ~isempty(d) && ~isfolder(d); mkdir(d); end
end
