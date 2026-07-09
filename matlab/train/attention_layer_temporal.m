classdef attention_layer_temporal < nnet.layer.Layer
    %ATTENTION_LAYER_TEMPORAL Temporal attention over BiLSTM outputs.

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
            layer.Name = name;
            layer.Description = 'Temporal Attention (TA)';
            layer.HiddenDim = hiddenDim;
            layer.AttnDim = attnDim;

            scale_w = sqrt(2 / hiddenDim);
            scale_v = sqrt(2 / attnDim);
            layer.Wta = randn(attnDim, hiddenDim) * scale_w;
            layer.bta = zeros(attnDim, 1);
            layer.vta = randn(1, attnDim) * scale_v;
        end

        function C = predict(layer, H)
            % H: [H_dim, T] or [H_dim, T, N]
            sz = size(H);
            H_dim = sz(1);
            T = sz(2);
            N = prod(sz(3:end));

            Wta = layer.Wta;
            bta = layer.bta;
            vta = layer.vta;

            H2 = reshape(H, H_dim, T * N);                  % [H_dim, T*N]
            U = tanh(Wta * H2 + repmat(bta, 1, T * N));     % [attn_dim, T*N]
            e = vta * U;                                    % [1, T*N]
            e = reshape(e, T, N);                           % [T, N]

            e_exp = exp(e - max(e, [], 1));
            alpha = e_exp ./ sum(e_exp, 1);                 % [T, N]

            % Keep dlarray outputs as dlarray instead of materializing doubles.
            H3 = reshape(H, H_dim, T, N);                   % [H, T, N]
            alpha3 = reshape(alpha, 1, T, N);               % [1, T, N]
            C = sum(H3 .* alpha3, 2);                       % [H, 1, N]

            out_sz = [H_dim, 1, sz(3:end)];
            if numel(out_sz) < 3
                C = reshape(C, H_dim, 1);
            else
                C = reshape(C, out_sz);
            end
        end
    end
end
