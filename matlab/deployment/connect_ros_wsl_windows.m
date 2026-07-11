function connect_ros_wsl_windows(varargin)
%CONNECT_ROS_WSL_WINDOWS Connect Windows MATLAB to WSL ROS1.
%
% This helper exists because ROS1 peer-to-peer transport breaks if MATLAB
% registers itself with an unresolvable Windows hostname. On the current
% machine, the verified stable setup is to keep both the ROS master and the
% MATLAB node on "localhost" from MATLAB's point of view.
%
% On the current machine, the stable pattern is:
%   - ROS master URI from Windows MATLAB: http://localhost:11311
%   - MATLAB NodeHost registered as: localhost
%
% Example on the current machine:
%   cd('\\wsl.localhost\Ubuntu-20.04\home\jiuyao\drone-formation-e2e')
%   addpath(fullfile(pwd, 'matlab', 'deployment'))
%   connect_ros_wsl_windows('MasterHost','localhost', 'NodeHost','localhost')
%
% Name-value pairs:
%   'MasterHost' : ROS master host/IP reachable from Windows MATLAB
%   'MasterPort' : ROS master port, default 11311
%   'NodeHost'   : host/IP registered by MATLAB for ROS peer connections
%   'ShutdownFirst' : whether to call rosshutdown first, default true

parser = inputParser;
parser.FunctionName = mfilename;
addParameter(parser, 'MasterHost', 'localhost');
addParameter(parser, 'MasterPort', 11311);
addParameter(parser, 'NodeHost', 'localhost');
addParameter(parser, 'ShutdownFirst', true);
parse(parser, varargin{:});

masterHost = char(string(parser.Results.MasterHost));
masterPort = double(parser.Results.MasterPort);
nodeHost = char(string(parser.Results.NodeHost));
shutdownFirst = logical(parser.Results.ShutdownFirst);
validate_host_string(masterHost, 'MasterHost');
validate_host_string(nodeHost, 'NodeHost');

masterUri = sprintf('http://%s:%d', masterHost, masterPort);

if shutdownFirst
    try
        rosshutdown;
    catch
    end
else
    try
        rostopic('list');
        fprintf('[connect_ros_wsl_windows] existing global ROS node detected, reusing current session.\n');
        return;
    catch
    end
end

setenv('ROS_MASTER_URI', masterUri);
setenv('ROS_IP', nodeHost);
setenv('ROS_HOSTNAME', nodeHost);

fprintf('[connect_ros_wsl_windows] ROS_MASTER_URI=%s\n', masterUri);
fprintf('[connect_ros_wsl_windows] ROS_IP=%s\n', nodeHost);
fprintf('[connect_ros_wsl_windows] ROS_HOSTNAME=%s\n', nodeHost);

try
    rosinit(masterUri, 'NodeHost', nodeHost);
catch ME
    error(['[connect_ros_wsl_windows] Failed to connect with MasterHost=%s and NodeHost=%s. ' ...
        'On this machine, both values should usually stay "localhost" unless networking was deliberately reconfigured. ' ...
        'Original error: %s'], masterHost, nodeHost, ME.message);
end

topics = rostopic('list');
fprintf('[connect_ros_wsl_windows] connected, visible topics=%d\n', numel(topics));
end

function validate_host_string(value, name)
assert(~isempty(strtrim(value)), ...
    '[connect_ros_wsl_windows] %s must be a non-empty host string.', name);
end
