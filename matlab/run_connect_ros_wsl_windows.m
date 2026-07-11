%RUN_CONNECT_ROS_WSL_WINDOWS Connect Windows MATLAB to the current WSL ROS master.
%
% Current machine baseline:
%   - WSL ROS master from Windows MATLAB: localhost:11311
%   - Windows MATLAB node host: localhost
%
% If networking is deliberately reconfigured later, update the arguments
% below only after revalidating the MATLAB <-> WSL ROS transport.

clc;

scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, 'deployment'));

connect_ros_wsl_windows( ...
    'MasterHost', 'localhost', ...
    'NodeHost', 'localhost');
