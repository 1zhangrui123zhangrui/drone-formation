classdef attention_layer_feature < nnet.layer.Layer
    % 特征注意力层（论文§4.4.1）
    % 在 BiLSTM 前对每帧的 D 维特征向量做 soft-weighting。
    %
    % 输入:  序列格式，每步 [D, 1]  (由 sequenceInputLayer 提供)
    % 输出:  同尺寸，加权后的特征向量序列
    %
    % 实现方式: trainNetwork 把序列 cell 中的每个 [D, T] 矩阵传入 forward/predict。
    % 我们在时间维度上循环应用同一组参数：
    %   e_t = W * x_t + b        (D→D 线性)
    %   alpha_t = softmax(e_t)   (在 D 维做 softmax)
    %   z_t = alpha_t .* x_t    (element-wise)

    properties (Learnable)
        Weights   % [D, D]
        Bias      % [D, 1]
    end

    properties
        InputDim
    end

    methods
        function layer = attention_layer_feature(inputDim, name)
            if nargin < 2; name = 'feature_attn'; end
            layer.Name        = name;
            layer.Description = 'Feature Attention (FA)';
            layer.InputDim    = inputDim;
            % Kaiming 初始化
            scale = sqrt(2 / inputDim);
            layer.Weights = randn(inputDim, inputDim) * scale;
            layer.Bias    = zeros(inputDim, 1);
        end

        function Z = predict(layer, X)
            % X: [D, T] or [D, T, N] — MATLAB 把 mini-batch 组织为 [D, T, N]
            % 对 T 维的每列独立做 FA
            sz = size(X);
            D  = sz(1);
            T  = sz(2);
            N  = prod(sz(3:end));   % batch size (可能为1)

            W = layer.Weights;      % [D, D]
            b = layer.Bias;         % [D, 1]

            X2 = reshape(X, D, T*N);         % [D, T*N]
            E  = W * X2 + repmat(b, 1, T*N); % [D, T*N]
            E  = tanh(E);

            % softmax over D dimension
            E_exp = exp(E - max(E, [], 1));
            alpha = E_exp ./ sum(E_exp, 1);  % [D, T*N]

            Z2 = alpha .* X2;                % [D, T*N]
            Z  = reshape(Z2, sz);            % back to [D, T, N]
        end
    end
end
