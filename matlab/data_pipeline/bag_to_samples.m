function samples = bag_to_samples(matPath)
%BAG_TO_SAMPLES  加载 Python bag_to_mat.py 输出的 .mat，验证并返回结构体
% matPath: Python 生成的 .mat 文件路径
%
% 返回 samples:
%   .X        [N, 15]  特征向量
%   .Y        [N, 4]   Teacher cmd_vel 标签
%   .drone_id [N, 1]   无人机编号
%   .t_vec    [N, 1]   时间（s）

if nargin < 1
    matPath = "data/processed/raw/scene01_hover.mat";
end

raw = load(matPath);
d   = raw.data;

X        = double(d.X);
Y        = double(d.Y);
drone_id = double(d.drone_id);
t_vec    = double(d.t_vec);

assert(size(X,2) == 15, 'X must be Nx15');
assert(size(Y,2) == 4,  'Y must be Nx4');
assert(~any(isnan(X(:))), 'NaN found in X');
assert(~any(isnan(Y(:))), 'NaN found in Y');

samples.X        = X;
samples.Y        = Y;
samples.drone_id = drone_id;
samples.t_vec    = t_vec;
samples.source   = matPath;

fprintf('[bag_to_samples] loaded: %d samples, X=[%dx%d]\n', ...
    size(X,1), size(X,1), size(X,2));
end
