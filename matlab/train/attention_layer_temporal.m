classdef attention_layer_temporal < nnet.layer.Layer ...
        & nnet.layer.Formattable ...
        & nnet.layer.Acceleratable
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
            % H uses MATLAB sequence format:
            %   single sequence -> [H_dim, T]
            %   mini-batch      -> [H_dim, B, T]  (format "CBT")
            %
            % Output should be sequence-to-one context:
            %   single sample -> [H_dim, 1]
            %   mini-batch    -> [H_dim, B]
            %
            % Do not keep a singleton time dimension [H_dim, 1, N], otherwise
            % MATLAB can still classify the network as sequence-output and
            % require sequence-form regression responses.
            sz = size(H);
            H_dim = sz(1);

            Wta = layer.Wta;
            bta = layer.bta;
            vta = layer.vta;

            if numel(sz) < 3
                % Single sequence: H is [C, T]
                T = sz(2);
                H2 = reshape(H, H_dim, T);                  % [C, T]
                U = tanh(Wta * H2 + repmat(bta, 1, T));     % [A, T]
                e = vta * U;                                % [1, T]
                e_exp = exp(e - max(e, [], 2));
                alpha = e_exp ./ sum(e_exp, 2);             % [1, T]
                C = sum(H2 .* alpha, 2);                    % [C, 1]
            else
                % Mini-batch sequence: H is [C, B, T] in MATLAB's CBT format.
                B = sz(2);
                T = sz(3);

                H3 = reshape(H, H_dim, B, T);               % [C, B, T]
                H2 = reshape(permute(H3, [1 3 2]), H_dim, T * B);   % [C, T*B]
                U = tanh(Wta * H2 + repmat(bta, 1, T * B));         % [A, T*B]
                e = vta * U;                                        % [1, T*B]
                e = reshape(e, T, B);                               % [T, B]

                e_exp = exp(e - max(e, [], 1));
                alpha = e_exp ./ sum(e_exp, 1);                     % [T, B]

                alpha3 = permute(reshape(alpha, T, B), [3 2 1]);    % [1, B, T]
                C = sum(H3 .* alpha3, 3);                           % [C, B]
            end

            % This layer changes sequence data [C,B,T] into feature data [C,B].
            if isa(H, 'dlarray')
                C = dlarray(C, 'CB');
            end
        end

        function C = forward(layer, H)
            C = predict(layer, H);
        end
    end
end
