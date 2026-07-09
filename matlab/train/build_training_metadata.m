function metadata = build_training_metadata(modelName, dataDir, trainFile, valFile, tr, va)
%BUILD_TRAINING_METADATA Build a traceable metadata struct for one training run.

manifestPath = fullfile(dataDir, 'dataset_build_manifest.json');

metadata = struct();
metadata.model_name = modelName;
metadata.data_dir = dataDir;
metadata.train_file = trainFile;
metadata.val_file = valFile;
metadata.dataset_manifest_path = manifestPath;
metadata.dataset_manifest_exists = isfile(manifestPath);
metadata.created_at = datestr(now, 'yyyy-mm-dd HH:MM:SS');
metadata.train_windows = size(tr.X, 3);
metadata.val_windows = size(va.X, 3);
metadata.window_length = size(tr.X, 1);
metadata.input_dim = size(tr.X, 2);
metadata.output_dim = size(tr.Y, 2);
metadata.train_target_shape = size(tr.Y);
metadata.val_target_shape = size(va.Y);
end
