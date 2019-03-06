function   [ net, P, F,Z ] =  training( data_set, net,P, F,Z, iter,lr, param) 
%Z: N*dim
N = size(data_set,4);
c = param.c;
dimension =  param.dimensions ;
gamma = param.gamma;
alpha = param.alpha;
beta = param.beta;
eta = param.eta;
NEITER=30;
batchsize = param.batchsize;
index = randperm(N);
F0 = rand(batchsize,c); 
for j=0:ceil(N/batchsize)-1
    batch_time = tic;
    %% random select a minibatch
    ix = index((1+j*batchsize):min((j+1)*batchsize,N)) ;
    if(length(ix)==batchsize)
        %% load and preprocess an image
        temp = ix;
        im = data_set(:,:,:,temp) ;
        im_ = single(im) ; 
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
        im_ = im_ - repmat(net.meta.normalization.averageImage,1,1,1,size(im_,4)) ;
        im_ = gpuArray(im_) ;
    else
        ix1 = [];
        a=randperm(N);
        a = a(1:batchsize-length(ix));
        ix1 = [ix a];
        temp = ix1;
        im = data_set(:,:,:,temp) ;
        im_ = single(im) ; % note: 0-255 range single()
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
        im_ = im_ - repmat(net.meta.normalization.averageImage,1,1,1,size(im_,4)) ;
        im_ = gpuArray(im_) ;
    end
    options = [];
    options.Metric = 'Euclidean';
    options.NeighborMode = 'KNN';
    options.k = 50;
    options.WeightMode = 'HeatKernel';
    options.t = 1;
    %% run the CNN
    res = vl_simplenn(net, im_);
%     label_train =  squeeze(gather(res(end).x))';
    Z0 = squeeze(gather(res(end).x))'; 
    Z(temp,:) = Z0;
%     L0 = squeeze(gather(res(end).x))' ; 
    M = constructW(Z0,options);
%     for o = 1:20
        %% update P
        temp_gamma = zeros(dimension,1);
        for i = 1:dimension
            temp_gamma(i) = 1/(2*sqrt(P(i,:)*P(i,:)')+eta);
        end
        Gamma  = diag(temp_gamma);
        P = (Z0'*Z0 + gamma*Gamma)\Z0'*F0;
        %% update F
        Q = Z0'*Z0 + gamma*Gamma;
        L_s = full(diag(sum(M,2))-M);
        Mat_F = L_s + beta*ones(batchsize, batchsize) - beta*(Z0/Q*Z0');
        [M, lambda] = eig(Mat_F);
        % Sort eigenvalues and eigenvectors in descending order
        lambda(isnan(lambda)) = 0;
        [lambda, ind] = sort(diag(lambda), 'ascend');
        F0 = M(:,ind(1:c));
%     end
    if(batchsize==length(ix))
        F(temp,:) = F0;
    else
        F(temp,:) = F0;
    end
%     
    dJdZ =(L_s+L_s')*Z0+param.beta*2*(Z0*P*P'-F0*P'); % for Q0
    dJdoutput = gpuArray(reshape(dJdZ',[1,1,size(dJdZ',1),size(dJdZ',2)])) ;
    res = vl_simplenn(net, im_, dJdoutput);    
    %% update the parameters of CNN
    net = updating_net(net, res, lr, N);
    batch_time = toc(batch_time);
%     svmStr = svmtrain(Z0,label1,'Showplot',true);
%     pred_label = svmclassify(svmStr,Z0,'Showplot',true);
%     [~,pred_label] =max(label_train,[],2);
%     acc = length(find(pred_label==label1))/batchsize;
    acc = 0;
    fprintf(' iter %d  batch %d/%d (%.1f images/s) ,lr is %d, acc is %d\n ', iter, j+1,ceil(size(data_set,4)/batchsize), batchsize/ batch_time,lr,acc) ;
end
%     label1 = [];
%     for rt = 1:length(dataset_L)
%         label1 = [label1;find(dataset_L(rt,:)==1)];
%     end
%     options.gnd = label1;
%     M_all = constructW(Z0,options);
%     L_all_s = diag(sum(M_all,2))-M_all;
%     temp_formulation32 = 0;
%     for i = 1:dimension
%         temp_formulation32 = temp_formulation32 + norm(P(i,:),2);
%     end
    loss = 0;
%      loss =  gamma*temp_formulation32+alpha*trace(F'*L_all_s*F); %beta*(norm(Z0*P-F,'fro')^2+
end

