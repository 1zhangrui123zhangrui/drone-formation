classdef weighted_mse_regression_layer < nnet.layer.RegressionLayer
    properties
        ChannelWeights
    end
    methods
        function layer = weighted_mse_regression_layer(weights, name)
            layer.Name = name;
            layer.Description = 'Channel-weighted MSE';
            layer.ChannelWeights = reshape(double(weights), 1, []);
        end
        function loss = forwardLoss(layer, Y, T)
            w = cast(layer.ChannelWeights, 'like', Y);
            if size(Y, 1) == numel(w)
                w = reshape(w, [], 1);
            end
            loss = mean((Y - T).^2 .* w, 'all');
        end
    end
end
