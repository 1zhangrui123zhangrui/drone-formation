function tests = test_metrics
repoRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(repoRoot, "matlab", "evaluation"));
tests = functiontests(localfunctions);
end

function testComputeMetricsColumns(testCase)
metrics = compute_metrics([1; 2], [0.5; 0.5], [0; 1]);
expectedNames = {'rmse', 'mae', 'mean_effort', 'collision_count'};
verifyTrue(testCase, all(ismember(expectedNames, metrics.Properties.VariableNames)));
verifyEqual(testCase, metrics.collision_count, 1);
end
