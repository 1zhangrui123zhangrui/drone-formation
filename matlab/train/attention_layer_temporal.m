classdef attention_layer_temporal < nnet.layer.Layer
    % 时序注意力层（论文§4.4.2）
    % 输入:  BiLSTM 输出序列 [H, T] (H = hidden*2, T = timesteps)
    % 输出:  上下文向量 [H, 1] = sum_t(alpha_t * h_t)
    %
    % attention score:
    %   u_t = tanh(Wta * h_t + bta)  [attn_dim, T]
    %   e_t = vta * u_t              [1, T]
    %   alpha = softmax(e)           [1, T]
    %   c = H * alpha'               [H, 1]

    properties (Learnable)
        Wta   % [attn_dim, H]
        bta   % [attn_dim, 1]
        vta   % [1, attn_dim]
    end

    properties
        HiddenDim
        AttnDim
    end

    methods
        function layer = attention_layer_temporal(hiddenDim, attnDim, name)
            if nargin < 3; name = 'temporal_attn'; end
            layer.Name        = name;
            layer.Description = 'Temporal Attention (TA)';
            layer.HiddenDim   = hiddenDim;
            layer.AttnDim     = attnDim;

            scale_w = sqrt(2 / hiddenDim);
            scale_v = sqrt(2 / attnDim);
            layer.Wta = randn(attnDim, hiddenDim) * scale_w;
            layer.bta = zeros(attnDim, 1);
            layer.vta = randn(1, attnDim) * scale_v;
        end

        function C = predict(layer, H)
            % H: [H_dim, T] or [H_dim, T, N]
            sz    = size(H);
            H_dim = sz(1);
            T     = sz(2);
            N     = prod(sz(3:end));

            Wta = layer.Wta;  % [attn_dim, H_dim]
            bta = layer.bta;  % [attn_dim, 1]
            vta = layer.vta;  % [1, attn_dim]

            H2 = reshape(H, H_dim, T*N);             % [H_dim, T*N]
            U  = tanh(Wta * H2 + repmat(bta, 1, T*N)); % [attn_dim, T*N]
            e  = vta * U;                             % [1, T*N]
            e  = reshape(e, T, N);                   % [T, N]

            % softmax over T
            e_exp = exp(e - max(e, [], 1));
            alpha = e_exp ./ sum(e_exp, 1);           % [T, N]

            % weighted sum: C[H,N] = H3[H,T,N] * alpha[T,N] per sample
            H3 = reshape(H, H_dim, T, N);             % [H, T, N]
            C_data = zeros(H_dim, N);
            for n = 1:N
                C_data(:,n) = H3(:,:,n) * alpha(:,n); % [H,T]*[T,1]
            end

            % output: 保持序列格式 [H, 1, N] 方便后接 FC 层
            out_sz = [H_dim, 1, sz(3:end)];
            if numel(out_sz) < 3
                C = reshape(C_data, H_dim, 1);
            else
                C = reshape(C_data, out_sz);
            end
        end
    end
end
