function control = mpc_controller(state, reference, gains)
%MPC_CONTROLLER Lightweight baseline controller placeholder.

if nargin < 3 || isempty(gains)
    gains = struct("Kp", 0.8, "Kd", 0.1);
end

state = state(:);
reference = reference(:);
errorVec = reference - state;
control = gains.Kp .* errorVec(1:min(4, numel(errorVec)));
if numel(control) < 4
    control(end + 1:4, 1) = 0;
end
end
