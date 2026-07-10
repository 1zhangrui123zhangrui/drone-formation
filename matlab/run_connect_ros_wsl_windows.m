%RUN_CONNECT_ROS_WSL_WINDOWS Connect Windows MATLAB to the current WSL ROS master.
%
% Current machine baseline:
%   - WSL ROS master: 10.16.33.80:11311
%   - Windows MATLAB node host: 10.16.33.80
%
% If the active network changes, update the IPs below before running.

clc;

scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, 'deployment'));

connect_ros_wsl_windows( ...
    'MasterHost', '10.16.33.80', ...
    'NodeHost', '10.16.33.80');
