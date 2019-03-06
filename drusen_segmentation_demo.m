function  drusen_segmentation_demo(dim,param)
% c: class number
% S:  learned symmetric similarity matrix
% dimensions: output feature dimensions
%% prepare the dataset
% data_prepare_cifar10;
%% load the pre-trained CNN
net = load('C:\RXX\Medical_segmentation\data\imagenet-vgg-f.mat') ;
%% load the Dataset
load('C:\RXX\Medical_segmentation\data\new_train_50w.mat') ;
train_data =  data_set;
train_L=dataset_L;

%% initialization
lr = logspace(-2,-6,param.maxIter) ; %generate #maxIter of points between 10^(-2) ~ 10^(-6)
net = net_struc(net,dim,param.c) ;
train_num = size(data_set,4);
P = zeros(param.dimensions, param.c);
F = zeros(train_num,param.c);
Z = zeros(train_num, param.dimensions);
loss1 = [];
 %% net training
for iter = 1: param.maxIter
 [ net, P, F,Z] = training( train_data, net, P, F, Z, iter,lr(iter),  param) ;
 %loss1 = [loss1, loss];
 %save(['C:\RXX\Medical_segmentation\Loss_drusen_' num2str(dimensions) '.mat'],'loss1') ;
 if(rem(iter,10)==0)
    save(['C:\RXX\Medical_segmentation\net_param\net_param__' num2str(iter) '.mat'],'net','P','F','Z','train_data','train_L', '-v7.3');
 end
end


%% SVM training
   batchsize = 128;
   
    for j = 0:ceil(size(train_data,4)/batchsize)-1
        im = train_data(:,:,:,(1+j*batchsize):min((j+1)*batchsize,size(train_data,4))) ;
        im_ = single(im) ; % note: 0-255 range
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
        im_ = im_ - repmat(net.meta.normalization.averageImage,1,1,1,size(im_,4)) ;
        im_ = gpuArray(im_) ;
        % run the CNN
        res = vl_simplenn(net, im_) ;
        %Z_test = squeeze(gather(res(end-2).x));
        SVMtrain_data1((1+j*batchsize):min((j+1)*batchsize,size(train_data,4)),:) = squeeze(gather(res(end).x))';    
    end

    fprintf('data_generator_end');
    
    [~,train_truth] =max(train_L,[],2);
    train_truth = double(train_truth);

   train_truth(find(train_truth==1))=0;
   train_truth(find(train_truth==2))=1;
    
    train_data = double(SVMtrain_data1);
    model = libsvmtrain(train_truth,train_data,'-b 1');
    fprintf('model_generator_end');
    save 'C:\RXX\Medical_segmentation\class_model\classify_model.mat' model -v7.3
   
       
end

