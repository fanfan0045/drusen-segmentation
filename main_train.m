clc;
clear all;
warning('on');
for dim = [ 128 ] 
    w=15;%windowsize
    param.dimensions = dim;
    param.c = 2;
    param.alpha = 1;
    param.gamma = 0.01;
    param.beta = 0.01;
    param.maxIter = 60000;
    param.batchsize = 128;
    param.eta = 0.001;
    drusen_segmentation_demo(dim, param);
end