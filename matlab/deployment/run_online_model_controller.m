function run_online_model_controller(modelKey, varargin)
%RUN_ONLINE_MODEL_CONTROLLER Minimal MATLAB ROS online controller for c1/c3a.
%
% This controller is intentionally narrow in scope:
% - supports only c1 (9D-LSTM) and c3a (15D-BiLSTM)
% - publishes world-frame commands to /droneN/command/twist
% - uses Teacher /cmd_vel_teacher as an online feature source
% - falls back to Teacher commands during warmup or invalid inference
%
% Prerequisites:
%   1. A ROS connection is already active in MATLAB via rosinit(...)
%   2. Gazebo/ROS scene is running
%   3. For student runs, Teacher must be launched with publish_world_cmd=false
%
% Example:
%   cd('\\wsl.localhost\Ubuntu-20.04\home\jiuyao\drone-formation-e2e')
%   rosinit('http://<ros-master>:11311')
%   addpath(fullfile(pwd, 'matlab', 'deployment'))
%   run_online_model_controller('c1')
%
% Optional name-value pairs:
%   'ProjectRoot'        : repo root path
%   'NumDrones'          : number of drones (default 4)
%   'WarmupUseTeacher'   : use Teacher during buffer warmup (default true)
%   'FallbackOnInvalid'  : use Teacher when inference invalid (default true)
%   'StudentEnableMinSimTime' : keep Teacher control before this sim time (default 12.0)
%   'MaxCmdXY'           : horizontal command saturation (default 2.5)
%   'MaxCmdZ'            : vertical command saturation (default 1.5)
%   'MaxCmdYaw'          : yaw-rate command saturation (default 1.5)
%   'LoopSleepSec'       : polling loop sleep (default 0.01)
%   'LogPeriodSec'       : status log period (default 3.0)

thisDir = fileparts(mfilename('fullpath'));
matlabDir = fileparts(thisDir);
projectRootDefault = fileparts(matlabDir);

opts = parse_options(projectRootDefault, varargin{:});
spec = resolve_spec(opts.project_root, modelKey);

ensure_ros_connection();

loaded = load(spec.model_path);
assert(isfield(loaded, 'net'), '[online] model file missing net: %s', spec.model_path);
net = loaded.net;

if isfield(loaded, 'metadata')
    metadata = loaded.metadata;
    if isfield(metadata, 'window_length')
        spec.window_length = double(metadata.window_length);
    end
    if isfield(metadata, 'input_dim')
        spec.input_dim = double(metadata.input_dim);
    end
end

stats = load_norm_stats(spec.stats_path);
assert(numel(stats.mean) == spec.input_dim, ...
    '[online] stats dimension mismatch for %s: expected %d, got %d', ...
    spec.model_name, spec.input_dim, numel(stats.mean));

subs = make_subscribers(opts.num_drones);
pubs = make_publishers(opts.num_drones);
buffers = cell(opts.num_drones, 1);
last_pdes_stamp = -inf(opts.num_drones, 1);
last_mode = repmat("idle", opts.num_drones, 1);
last_log = tic;

cleanupObj = onCleanup(@() stop_all_drones(pubs)); %#ok<NASGU>

fprintf('[online] model=%s  input_dim=%d  W=%d\n', ...
    spec.model_name, spec.input_dim, spec.window_length);
fprintf('[online] model_path=%s\n', spec.model_path);
fprintf('[online] stats_path=%s\n', spec.stats_path);
fprintf('[online] num_drones=%d  warmup_teacher=%d  fallback_invalid=%d\n', ...
    opts.num_drones, opts.warmup_use_teacher, opts.fallback_on_invalid);
fprintf('[online] student_enable_min_sim_time=%.1fs  cmd_limits=[xy=%.2f z=%.2f yaw=%.2f]\n', ...
    opts.student_enable_min_sim_time, opts.max_cmd_xy, opts.max_cmd_z, opts.max_cmd_yaw);

while true
    batchDroneIds = zeros(0, 1);
    batchInputs = {};
    batchTeacherCmds = {};

    for droneId = 1:opts.num_drones
        [raw15, teacher_cmd, pdes_stamp, ready] = collect_inputs(subs, droneId);
        if ~ready
            continue;
        end

        if ~(isfinite(pdes_stamp) && pdes_stamp > last_pdes_stamp(droneId) + 1e-6)
            continue;
        end
        last_pdes_stamp(droneId) = pdes_stamp;

        buffers{droneId} = append_feature(buffers{droneId}, raw15, spec.window_length);

        [cmd, mode, inputCell, needsPredict] = stage_or_fallback( ...
            buffers{droneId}, teacher_cmd, pdes_stamp, spec, stats, opts);
        if needsPredict
            batchDroneIds(end+1, 1) = droneId; %#ok<AGROW>
            batchInputs{end+1, 1} = inputCell; %#ok<AGROW>
            batchTeacherCmds{end+1, 1} = teacher_cmd; %#ok<AGROW>
            continue;
        end

        if ~isempty(cmd)
            publish_world_cmd(pubs{droneId}, cmd);
            last_mode(droneId) = mode;
        end
    end

    if ~isempty(batchDroneIds)
        [cmds, modes] = infer_batch_or_fallback(batchInputs, batchTeacherCmds, net, opts);
        for k = 1:numel(batchDroneIds)
            droneId = batchDroneIds(k);
            if isempty(cmds{k})
                continue;
            end
            publish_world_cmd(pubs{droneId}, cmds{k});
            last_mode(droneId) = modes(k);
        end
    end

    if toc(last_log) >= opts.log_period_sec
        log_status(buffers, last_mode, spec.window_length);
        last_log = tic;
    end

    pause(opts.loop_sleep_sec);
end
end

function opts = parse_options(projectRootDefault, varargin)
parser = inputParser;
parser.FunctionName = mfilename;
addParameter(parser, 'ProjectRoot', projectRootDefault);
addParameter(parser, 'NumDrones', 4);
addParameter(parser, 'WarmupUseTeacher', true);
addParameter(parser, 'FallbackOnInvalid', true);
addParameter(parser, 'StudentEnableMinSimTime', 12.0);
addParameter(parser, 'MaxCmdXY', 2.5);
addParameter(parser, 'MaxCmdZ', 1.5);
addParameter(parser, 'MaxCmdYaw', 1.5);
addParameter(parser, 'LoopSleepSec', 0.01);
addParameter(parser, 'LogPeriodSec', 3.0);
parse(parser, varargin{:});
opts = parser.Results;
opts.project_root = resolve_project_path(projectRootDefault, opts.ProjectRoot);
opts.num_drones = double(opts.NumDrones);
opts.warmup_use_teacher = logical(opts.WarmupUseTeacher);
opts.fallback_on_invalid = logical(opts.FallbackOnInvalid);
opts.student_enable_min_sim_time = double(opts.StudentEnableMinSimTime);
opts.max_cmd_xy = double(opts.MaxCmdXY);
opts.max_cmd_z = double(opts.MaxCmdZ);
opts.max_cmd_yaw = double(opts.MaxCmdYaw);
opts.loop_sleep_sec = double(opts.LoopSleepSec);
opts.log_period_sec = double(opts.LogPeriodSec);
end

function spec = resolve_spec(projectRoot, modelKey)
key = lower(string(modelKey));
spec = struct();

switch key
    case {"c1", "c1_lstm9d"}
        spec.model_key = 'c1';
        spec.model_name = 'c1_lstm9d';
        spec.model_path = fullfile(projectRoot, 'data', 'trained_models', 'c1_lstm9d.mat');
        spec.stats_path = fullfile(projectRoot, 'data', 'processed', 'norm_stats_9d.mat');
        spec.input_dim = 9;
        spec.feature_cols = 7:15;
        spec.window_length = 20;
    case {"c3a", "c3a_bilstm"}
        spec.model_key = 'c3a';
        spec.model_name = 'c3a_bilstm';
        spec.model_path = fullfile(projectRoot, 'data', 'trained_models', 'c3a_bilstm.mat');
        spec.stats_path = fullfile(projectRoot, 'data', 'processed', 'norm_stats_15d.mat');
        spec.input_dim = 15;
        spec.feature_cols = 1:15;
        spec.window_length = 20;
    otherwise
        error('[online] unsupported modelKey "%s". Only c1 and c3a are enabled.', modelKey);
end

assert(isfile(spec.model_path), '[online] missing model file: %s', spec.model_path);
assert(isfile(spec.stats_path), '[online] missing stats file: %s', spec.stats_path);
end

function assert_ros_connection()
try
    rostopic('list');
catch ME
    error(['[online] No active ROS connection in MATLAB. Call rosinit(...) first. ' ...
        'Original error: %s'], ME.message);
end
end

function ensure_ros_connection()
try
    assert_ros_connection();
catch ME
    fprintf('[online] ROS connection missing, attempting automatic reconnect...\n');
    fprintf('[online] original check failed: %s\n', ME.message);
    connect_ros_wsl_windows('ShutdownFirst', false);
    assert_ros_connection();
end
end

function subs = make_subscribers(numDrones)
subs = cell(numDrones, 1);
for droneId = 1:numDrones
    ns = sprintf('/drone%d', droneId);
    s = struct();
    s.odom = rossubscriber([ns '/ground_truth/state'], 'nav_msgs/Odometry');
    s.p_des = rossubscriber([ns '/p_des'], 'geometry_msgs/PointStamped');
    s.v_des = rossubscriber([ns '/v_des'], 'geometry_msgs/Vector3Stamped');
    s.teacher = rossubscriber([ns '/cmd_vel_teacher'], 'geometry_msgs/Twist');
    subs{droneId} = s;
end
end

function pubs = make_publishers(numDrones)
pubs = cell(numDrones, 1);
for droneId = 1:numDrones
    pubs{droneId} = rospublisher(sprintf('/drone%d/command/twist', droneId), ...
        'geometry_msgs/TwistStamped');
end
end

function [raw15, teacher_cmd, pdes_stamp, ready] = collect_inputs(subs, droneId)
odom = subs{droneId}.odom.LatestMessage;
pdes = subs{droneId}.p_des.LatestMessage;
vdes = subs{droneId}.v_des.LatestMessage;
teacher = subs{droneId}.teacher.LatestMessage;

raw15 = [];
teacher_cmd = [];
pdes_stamp = NaN;
ready = false;

if isempty(odom) || isempty(pdes) || isempty(vdes) || isempty(teacher)
    return;
end

p_actual = [
    double(odom.Pose.Pose.Position.X), ...
    double(odom.Pose.Pose.Position.Y), ...
    double(odom.Pose.Pose.Position.Z)];
v_actual = [
    double(odom.Twist.Twist.Linear.X), ...
    double(odom.Twist.Twist.Linear.Y), ...
    double(odom.Twist.Twist.Linear.Z)];
p_des = [
    double(pdes.Point.X), ...
    double(pdes.Point.Y), ...
    double(pdes.Point.Z)];
v_des = [
    double(vdes.Vector.X), ...
    double(vdes.Vector.Y), ...
    double(vdes.Vector.Z)];
teacher_xyz = [
    double(teacher.Linear.X), ...
    double(teacher.Linear.Y), ...
    double(teacher.Linear.Z)];
teacher_cmd = [
    double(teacher.Linear.X), ...
    double(teacher.Linear.Y), ...
    double(teacher.Linear.Z), ...
    double(teacher.Angular.Z)];

raw15 = [p_des, v_des, p_actual, v_actual, teacher_xyz];
if ~all(isfinite(raw15)) || ~all(isfinite(teacher_cmd))
    raw15 = [];
    teacher_cmd = [];
    return;
end

pdes_stamp = stamp_to_sec(pdes.Header.Stamp);
ready = isfinite(pdes_stamp);
end

function t = stamp_to_sec(stamp)
try
    t = double(stamp.Sec) + 1e-9 * double(stamp.Nsec);
catch
    t = NaN;
end
end

function buf = append_feature(buf, raw15, windowLength)
if isempty(buf)
    buf = raw15;
else
    buf = [buf; raw15]; %#ok<AGROW>
end
if size(buf, 1) > windowLength
    buf = buf(end-windowLength+1:end, :);
end
end

function [cmd, mode, inputCell, needsPredict] = stage_or_fallback(buf, teacherCmd, simTime, spec, stats, opts)
cmd = [];
mode = "idle";
inputCell = [];
needsPredict = false;

if size(buf, 1) < spec.window_length
    if opts.warmup_use_teacher && ~isempty(teacherCmd)
        cmd = teacherCmd;
        mode = "teacher_warmup";
    end
    return;
end

if simTime < opts.student_enable_min_sim_time
    if opts.warmup_use_teacher && ~isempty(teacherCmd)
        cmd = teacherCmd;
        mode = "teacher_pre_student_window";
    end
    return;
end

window = buf(:, spec.feature_cols);
windowNorm = normalize_window(window, stats);
if ~all(isfinite(windowNorm), 'all')
    if opts.fallback_on_invalid && ~isempty(teacherCmd)
        cmd = teacherCmd;
        mode = "teacher_invalid_input";
    end
    return;
end

inputCell = single(windowNorm');
needsPredict = true;
end

function windowNorm = normalize_window(window, stats)
windowNorm = (double(window) - stats.mean) ./ stats.std;
end

function [cmds, modes] = infer_batch_or_fallback(inputCells, teacherCmds, net, opts)
numItems = numel(inputCells);
cmds = cell(numItems, 1);
modes = repmat("idle", numItems, 1);

try
    pred = predict(net, inputCells, 'MiniBatchSize', numItems);
    pred = normalize_batch_prediction_shape(pred, numItems);
catch ME
    for i = 1:numItems
        if opts.fallback_on_invalid && ~isempty(teacherCmds{i})
            cmds{i} = teacherCmds{i};
            modes(i) = "teacher_predict_error";
        end
    end
    warning('[online] batched inference failed, falling back to Teacher: %s', ME.message);
    return;
end

for i = 1:numItems
    cmd = pred(i, :);
    if ~all(isfinite(cmd)) || numel(cmd) ~= 4
        if opts.fallback_on_invalid && ~isempty(teacherCmds{i})
            cmds{i} = teacherCmds{i};
            modes(i) = "teacher_invalid_pred";
        else
            cmds{i} = [];
            modes(i) = "invalid_pred";
        end
        continue;
    end

    cmd = double(cmd(:))';
    cmdBeforeClip = cmd;
    cmd = clip_command(cmd, opts);
    cmds{i} = cmd;
    if any(abs(cmd - cmdBeforeClip) > 1e-9)
        modes(i) = "student_clipped";
    else
        modes(i) = "student";
    end
end
end

function Y_pred = normalize_batch_prediction_shape(Y_pred, expectedN)
if iscell(Y_pred)
    if numel(Y_pred) ~= expectedN
        error('[online] expected %d prediction cells, got %d.', expectedN, numel(Y_pred));
    end
    tmp = zeros(expectedN, 4);
    for i = 1:expectedN
        tmp(i, :) = reshape_prediction(Y_pred{i});
    end
    Y_pred = tmp;
    return;
end

sz = size(Y_pred);
if ismatrix(Y_pred) && sz(1) == expectedN && sz(2) == 4
    return;
end
if ismatrix(Y_pred) && sz(1) == 4 && sz(2) == expectedN
    Y_pred = Y_pred';
    return;
end
if expectedN == 1
    Y_pred = reshape_prediction(Y_pred);
    return;
end

error('[online] unexpected batched prediction shape: [%s]', num2str(sz));
end

function cmd = reshape_prediction(pred)
if iscell(pred)
    if numel(pred) ~= 1
        error('[online] expected single prediction cell, got %d cells.', numel(pred));
    end
    pred = pred{1};
end

pred = double(pred);
if isequal(size(pred), [1, 4])
    cmd = pred;
elseif isequal(size(pred), [4, 1])
    cmd = pred';
elseif isvector(pred) && numel(pred) == 4
    cmd = reshape(pred, 1, 4);
else
    error('[online] unexpected prediction shape: [%s]', num2str(size(pred)));
end
end

function publish_world_cmd(pub, cmd)
msg = rosmessage(pub);
msg.Header.Stamp = rostime('now');
msg.Header.FrameId = 'world';
msg.Twist.Linear.X = cmd(1);
msg.Twist.Linear.Y = cmd(2);
msg.Twist.Linear.Z = cmd(3);
msg.Twist.Angular.Z = cmd(4);
send(pub, msg);
end

function cmd = clip_command(cmd, opts)
cmd = double(cmd(:))';
cmd(1) = max(min(cmd(1), opts.max_cmd_xy), -opts.max_cmd_xy);
cmd(2) = max(min(cmd(2), opts.max_cmd_xy), -opts.max_cmd_xy);
cmd(3) = max(min(cmd(3), opts.max_cmd_z), -opts.max_cmd_z);
cmd(4) = max(min(cmd(4), opts.max_cmd_yaw), -opts.max_cmd_yaw);
end

function stop_all_drones(pubs)
for i = 1:numel(pubs)
    msg = rosmessage(pubs{i});
    msg.Header.Stamp = rostime('now');
    msg.Header.FrameId = 'world';
    msg.Twist.Linear.X = 0.0;
    msg.Twist.Linear.Y = 0.0;
    msg.Twist.Linear.Z = 0.0;
    msg.Twist.Angular.Z = 0.0;
    try
        send(pubs{i}, msg);
    catch
    end
end
end

function log_status(buffers, lastMode, windowLength)
parts = strings(numel(buffers), 1);
for i = 1:numel(buffers)
    fillCount = size(buffers{i}, 1);
    parts(i) = sprintf('d%d:%s(%d/%d)', i, lastMode(i), fillCount, windowLength);
end
fprintf('[online] %s\n', strjoin(parts, '  '));
end

function stats = load_norm_stats(statsPath)
loaded = load(statsPath);
names = fieldnames(loaded);
assert(~isempty(names), '[online] empty stats file: %s', statsPath);
stats = loaded.(names{1});

assert(isfield(stats, 'mean') && isfield(stats, 'std'), ...
    '[online] stats file missing mean/std fields: %s', statsPath);

stats.mean = reshape(double(stats.mean), 1, []);
stats.std = reshape(double(stats.std), 1, []);
stats.std(stats.std < 1e-8) = 1.0;
end

function p = resolve_project_path(projectRoot, p)
if ~(ischar(p) || isstring(p))
    return;
end

p = char(p);
if startsWith(p, '\\') || startsWith(p, '/') || ~isempty(regexp(p, '^[A-Za-z]:[\\/]', 'once'))
    return;
end

p = fullfile(projectRoot, p);
end
