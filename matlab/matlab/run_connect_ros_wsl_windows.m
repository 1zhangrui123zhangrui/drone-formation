% Compatibility wrapper for running from inside the matlab/ directory:
%   run matlab/run_connect_ros_wsl_windows.m

wrapperDir = fileparts(mfilename('fullpath'));
matlabDir = fileparts(wrapperDir);
run(fullfile(matlabDir, 'run_connect_ros_wsl_windows.m'));
