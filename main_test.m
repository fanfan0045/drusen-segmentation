clc;
clear;

w=15;
testImg=imread('C:\RXX\Medical_segmentation\cnnImg\test_data4\im0063.png');
figure,imshow(testImg),title('testImgOriginal');
testImgMask=imread('C:\RXX\Medical_segmentation\cnnImg\test_data4\im0063maskBW.png');
figure,imshow(testImgMask),title('testImgMask');

[data_set1,test_label1] = test_patches(testImg,testImgMask,w);%obtain the training data and lables
%save('C:\RXX\Medical_segmentation\cnnImg\img0017_sample_test.mat','data_set1','test_label1','-v7.3');

test_L = test_label1;
test_data = data_set1;
load('C:\RXX\Medical_segmentation\net_param\net_param_10_128.mat');% load the trained net 

windows_size = 15;
[ test_data1, predict_label, dec_values, SegResultBW, model ] = test_evaluation( net, test_L, test_data,testImg, testImgMask );%test and show segmentation results

%save('C:\RXX\Medical_segmentation\extract_feature_128\0017feature_2.5w_net10.mat','test_data1', 'test_L'); 

% testImgMask1=testImgMask(:);
% SegResultBW1=SegResultBW(:);
% [ACC, SEN, SPEC, auc]=evaluation(testImgMask1, SegResultBW1);
% 
% save('C:\RXX\Medical_segmentation\0063_w15_25000trainD_net10\SE_200.mat','predict_label', 'dec_values','ACC', 'SEN', 'SPEC', 'auc');
