function gpu_net = updating_net (gpu_net, res_back, lr, N)
    weight_decay = 5*1e-4 ;
    n_layers = numel(gpu_net.layers);
    batch_size = 512 ;
    for ii = 1:n_layers
            if (isfield(gpu_net.layers{ii}, 'weights')&& ~isempty(gpu_net.layers{ii}.weights))% isfield()   gpu_net.layers{ii}.weights{1} = gpu_net.layers{ii}.weights{1}+...
                        lr*(res_back(ii).dzdw{1}/(batch_size*N) - weight_decay*gpu_net.layers{ii}.weights{1});
                    gpu_net.layers{ii}.weights{2} = gpu_net.layers{ii}.weights{2}+...
                        lr*(res_back(ii).dzdw{2}/(batch_size*N) - weight_decay*gpu_net.layers{ii}.weights{2});
            end
    end
end
