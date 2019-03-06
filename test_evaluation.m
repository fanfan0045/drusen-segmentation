function [ test_data1,predict_label, dec_values,SegResultBW, model ] = test_evaluation( net, test_L, test_data,testImg, testImgMask )
 
    batchsize = 128;
    class_ind1 = 2;
    class_ind2 = 1;
    for j = 0:ceil(size(test_data,4)/batchsize)-1
        im = test_data(:,:,:,(1+j*batchsize):min((j+1)*batchsize,size(test_data,4))) ;
        im_ = single(im) ; % note: 0-255 range
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
        im_ = im_ - repmat(net.meta.normalization.averageImage,1,1,1,size(im_,4)) ;
        im_ = gpuArray(im_) ;
        % run the CNN
        res = vl_simplenn(net, im_) ;
        %Z_test = squeeze(gather(res(end-2).x));
        test_data1((1+j*batchsize):min((j+1)*batchsize,size(test_data,4)),:) = squeeze(gather(res(end).x))';    
    end
%    save('C:\RXX\Medical_segmentation\extract_feature_128\0192feature_2.5w_net10.mat','test_data1', 'test_L'); 
   
    fprintf('data_generator_end');

 
    ground_truth=test_L;
%     [~,ground_truth] =max(test_L,[],2);
%     ground_truth = double(ground_truth);
% %     ground_truth(find(ground_truth==1))=0;
% %     ground_truth(find(ground_truth==2))=1;
    test_data1 = double(test_data1);
    load( 'C:\RXX\Medical_segmentation\class_model\classify_model_2.5w_128.mat');
    addpath(genpath('C:\RXX\Medical_segmentation\utilise\libsvm-3.23\matlab'));
    [predict_label, accuracy, dec_values]=svmpredict(ground_truth,test_data1,model,'-b 1');
    

    w =15;
    [m,n,~]=size(testImg);
    r=(w-1)/2;

    Background=dec_values(:,2);
    Drusen=dec_values(:,1);

    Background=reshape(Background,n-w+1,m-w+1);
    Background=padarray(Background,[r r],'replicate');
    Background=uint8(Background*256);
    Background=Background';
    figure,imshow(Background);title('Background P');

    Drusen=reshape(Drusen,n-w+1,m-w+1);
    Drusen=padarray(Drusen,[r r],'replicate');
    Drusen=uint8(Drusen*256);
    Drusen=Drusen';
    figure,imshow(Drusen);title('Drusen P');
    
    t=200;
    [m,n,p]=size(Background);
    for i=1:m
        for j=1:n
            if(Background(i,j,1)>t)       
                 SegResultBW(i,j)=255; 
            else
                SegResultBW(i,j)=0;
            end
        end
    end
    figure,imshow(SegResultBW);title('segment resultBW');
    
%     index1 = find(SegResultBW==1);
%     index2 = find(SegResultBW==0);
%     SegResultBW(index1) = 0;
%     SegResultBW(index2) = 1;
    segResult=uint8(SegResultBW);
    
    testSegOr(:,:,1) = segResult*0+testImg(:,:,1).*(1-segResult);  %改变图像中对应曲线的值，画出红色线
    testSegOr(:,:,2) = segResult*0+testImg(:,:,2).*(1-segResult);
    testSegOr(:,:,3) = segResult*255+testImg(:,:,3).*(1-segResult);
    figure,imshow(testSegOr);title('segment result');
    
    str=strcat('0063_w',num2str(w),'_25000trainD_net10');%the name of folder
    mkdir(str);%Create the name of above folder 
    strD=strcat(str,'\','drusenP_',num2str(w),'x',num2str(w),'.png');
    imwrite(Drusen,strD);%write the probability map of drusen
    strB=strcat(str,'\','backgroundP_',num2str(w),'x',num2str(w),'.png');
    imwrite(Background,strB);%write the probability map of background
    strG=strcat(str,'\','groundtruth_',num2str(w),'x',num2str(w),'.png');
    imwrite(testImgMask,strG);%save groundtruth
    strO=strcat(str,'\','testImgOr_',num2str(w),'x',num2str(w),'.png');
    imwrite(testImg,strO);%save original image 
    strC=strcat(str,'\','segment resultOr_',num2str(w),'x',num2str(w),num2str(t),'.png');
    imwrite(testSegOr,strC);%save segmentation result
    strBW=strcat(str,'\','segment resultBW_',num2str(w),'x',num2str(w),num2str(t),'.png');
    imwrite(SegResultBW,strBW);%save the binary result
 
%imwrite(SegResultBW, '.\0003_w15_25000trainD_net10\segment resultBW_15x15_230.png');
end

