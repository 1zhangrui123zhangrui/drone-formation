%RUN_ONLINE_C1_CONTROLLER Start the minimal MATLAB ROS online controller for c1.
%
% Example:
%   cd('\\wsl.localhost\Ubuntu-20.04\home\jiuyao\drone-formation-e2e')
%   run matlab/run_connect_ros_wsl_windows.m
%   run matlab/run_online_c1_controller.m

clc;

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
addpath(fullfile(scriptDir, 'deployment'));

connect_ros_wsl_windows('ShutdownFirst', false);

run_online_model_controller('c1', ...
    'StudentEnableMinSimTime', 12.0, ...
    'StudentBlend', 0.10, ...
    'MaxCmdXY', 1.2, ...
    'MaxCmdZ', 0.8, ...
    'MaxCmdYaw', 0.5);
