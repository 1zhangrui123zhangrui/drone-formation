% Compatibility wrapper for running from inside the matlab/ directory:
%   run matlab/run_online_c3a_controller.m

wrapperDir = fileparts(mfilename('fullpath'));
matlabDir = fileparts(wrapperDir);
run(fullfile(matlabDir, 'run_online_c3a_controller.m'));
