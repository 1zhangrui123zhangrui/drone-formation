function tests = test_data_pipeline
repoRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(repoRoot, "matlab", "data_pipeline"));
tests = functiontests(localfunctions);
end

function testSlidingWindowShape(testCase)
sequence = reshape(1:30, [10, 3]);
windows = sliding_window(sequence, 4, 2);
verifySize(testCase, windows, [4, 3, 4]);
end

function testNormalizeZeroMean(testCase)
data = [1 2; 3 4; 5 6];
[normalized, stats] = normalize(data);
verifyLessThan(testCase, abs(mean(normalized, 1)), [1e-10 1e-10]);
verifyEqual(testCase, size(stats.mean), [1 2]);
end
