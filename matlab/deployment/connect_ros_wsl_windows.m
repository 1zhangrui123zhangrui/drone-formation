function connect_ros_wsl_windows(varargin)
%CONNECT_ROS_WSL_WINDOWS Connect Windows MATLAB to WSL ROS1 with explicit IPv4.
%
% This helper exists because ROS1 peer-to-peer transport breaks if MATLAB
% registers itself with an unresolvable Windows hostname (for example an
% IDNA/punycode hostname such as "xn--5nxo7b"). We therefore force both the
% ROS master URI and the MATLAB node host to concrete IPv4 addresses.
%
% Example on the current machine:
%   cd('\\wsl.localhost\Ubuntu-20.04\home\jiuyao\drone-formation-e2e')
%   addpath(fullfile(pwd, 'matlab', 'deployment'))
%   connect_ros_wsl_windows('MasterHost','10.16.33.80', 'NodeHost','10.16.33.80')
%
% Name-value pairs:
%   'MasterHost' : ROS master host/IP reachable from Windows MATLAB
%   'MasterPort' : ROS master port, default 11311
%   'NodeHost'   : Windows MATLAB host/IP reachable from WSL/Gazebo
%   'ShutdownFirst' : whether to call rosshutdown first, default true

parser = inputParser;
parser.FunctionName = mfilename;
addParameter(parser, 'MasterHost', '10.16.33.80');
addParameter(parser, 'MasterPort', 11311);
addParameter(parser, 'NodeHost', '10.16.33.80');
addParameter(parser, 'ShutdownFirst', true);
parse(parser, varargin{:});

masterHost = char(string(parser.Results.MasterHost));
masterPort = double(parser.Results.MasterPort);
nodeHost = char(string(parser.Results.NodeHost));
shutdownFirst = logical(parser.Results.ShutdownFirst);

validate_ipv4(masterHost, 'MasterHost');
validate_ipv4(nodeHost, 'NodeHost');

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

rosinit(masterUri, 'NodeHost', nodeHost);

topics = rostopic('list');
fprintf('[connect_ros_wsl_windows] connected, visible topics=%d\n', numel(topics));
end

function validate_ipv4(value, name)
expr = '^\d{1,3}(\.\d{1,3}){3}$';
assert(~isempty(regexp(value, expr, 'once')), ...
    '[connect_ros_wsl_windows] %s must be an IPv4 string, got: %s', name, value);
end
