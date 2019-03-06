

function [data_set1,test_label1] = test_patches(testImg,testImgMask,w)
%% Set Parameters:
   
% orimg=rgb2gray(orimg);
[m,n,~]=size(testImg);
r=(w-1)/2;

%N=(m-2*r)*(n-2*r);
Num=0;
d=0;

for i = (1+r) : (m-r) 
     for j = (1+r) : (n-r)
         
        Xmin = i-r;
        Xmax = i+r;
        Ymin = j-r;
        Ymax = j+r;
        Patch=testImg(Xmin:Xmax,Ymin:Ymax,:);
        d=d+1;
        %F=W'*Patch*V;
        %FeatureMap(i-r,j-r,:)=F(:)'; 
        data_set(:,:,:,d)=Patch;
         Num =  Num+1;
    end
end

data_set1=double(data_set);

M=m-2*r;
N=n-2*r;
test_label=zeros(M,N);

for i = (1+r) : (m-r) 
     for j = (1+r) : (n-r)
         if testImgMask(i,j)==255  
            test_label(i-r,j-r)=1;
         else
            test_label(i-r,j-r)=0;
         end
     end
end
test_label1=test_label(:);%obtain the lable od test image
end
