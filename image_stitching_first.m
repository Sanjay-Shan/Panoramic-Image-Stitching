function x = panorama()

% Please enter the name of the directory consisting of the images for panaroma
% To change the sequence , keep only those images in the directory
direc='../mov2/';
listing=dir([direc '/*.jpg']); % please modify if png image to be read
numImages=5;

if length(listing) <5
    warning('please make sure there are atleast 5 images');
    return
end

% read the first 5 images in the directory
I1=imread(strcat(direc,listing(1).name));
I2=imread(strcat(direc,listing(2).name));
I3=imread(strcat(direc,listing(3).name));    
I4=imread(strcat(direc,listing(4).name));
I5=imread(strcat(direc,listing(5).name));

% Get the size of the image
numrows=size(I1,1);
numcols=size(I1,2);

% check the aspect ratio of the image and resize if rows and columns are
% greater than 1000 
for j=200:-20:50
    if numrows > 1000 & numcols > 1000
        [rows,cols]=rat(numrows./numcols) ;
        numrows=rows*j ;
        numcols=cols*j ;
    else
        break
    end
end

%running resize operation
I1=imresize(I1,[numrows numcols]) ;
I2=imresize(I2,[numrows numcols]) ;
I3=imresize(I3,[numrows numcols]) ;
I4=imresize(I4,[numrows numcols]) ;
I5=imresize(I5,[numrows numcols]) ;

Images = {I1, I2, I3, I4, I5};
figure(1) ; clf ;
montage({I1, I2, I3, I4, I5});
numImages=5;

% make single and grayscale
I1 = rgb2gray(im2single(I1)) ;
I2 = rgb2gray(im2single(I2)) ;
I3 = rgb2gray(im2single(I3)) ;
I4 = rgb2gray(im2single(I4)) ;
I5 = rgb2gray(im2single(I5)) ;
gray = {I1, I2, I3, I4, I5} ;

% Get the descriptors and key points for all 5 images
% vl_sift can be used for the purpose
for j=1:numImages 
   [k,ds]= vl_sift(gray{j}) ;
   key{j}=k; des{j}=ds;
end

%{
start with first image. Stitch images to it's right
So for 5 images - start with image1. Stitch 4 images on the right to it. 
%}

firstImageIndex=1;
rotationIndices=4;

imgIndex1= firstImageIndex; 
imgIndex2=firstImageIndex+1;
[X1,X2, matches] = featurematch(key{imgIndex1},des{imgIndex1},key{imgIndex2},des{imgIndex2});
H=ransac(X1,X2,matches);
pano = stitch(H,Images{imgIndex1},Images{imgIndex2});
panoSingle = rgb2gray(im2single(pano)) ;   
Ignore = 1; %

% now peform the above operation on rest of the images
for j=1:rotationIndices
   imageIndex=firstImageIndex + j; 
   if j~=Ignore
       %ignore already matched image
        [k,ds] = vl_sift(panoSingle);
        [X1,X2, matches] = featurematch(k,ds,key{imageIndex},des{imageIndex});
        H=ransac(X1,X2,matches);
        pano = stitch(H,pano,Images{imageIndex});
        panoSingle = rgb2gray(im2single(pano)) ;
   end
end

figure(2) ; clf ;
imagesc(pano) ; axis image off ;
title('Stitched Image') ;
imwrite(pano,'../stitched.png');

%feature match function.
function [X1,X2,matches] = featurematch(f1,d1,f2,d2)
    [matches, scores] = vl_ubcmatch(d1,d2) ;
    X1 = f1(1:2,matches(1,:)) ; X1(3,:) = 1 ; 
    X2 = f2(1:2,matches(2,:)) ; X2(3,:) = 1 ; 
end

% Ransac operation function.
function H = ransac(X1,X2,matches)
    clear H score ok ; %set variables to 0.
    numMatches = size(matches,2) ;
    % running the sampling operation for prescribed number of times
    for t = 1:300
        %get a random subset of 4 points
        subset = vl_colsubset(1:numMatches,4) ;
        A = [] ;
        %create a homography matrix using the 4 key points.
        for i = subset
            A = cat(1, A, kron(X1(:,i)', vl_hat(X2(:,i)))) ;    
        end
        % solve the linear equation Ah=0 using SVD
        [U,S,V] = svd(A) ; 
        % finally get the last column after doing SVD.
        H{t} = reshape(V(:,9),3,3) ; 

      % simply calculate the squared euclidean distance between the given
      % key points and the homography transformed points
      X2_H = H{t} * X1 ; % projection of X1 after applying homography
      du = X2_H(1,:)./X2_H(3,:) - X2(1,:)./X2(3,:) ;
      dv = X2_H(2,:)./X2_H(3,:) - X2(2,:)./X2(3,:) ;
      ok{t} = (du.*du + dv.*dv) < 6*6 ; % if distance < threshold.
      score(t) = sum(ok{t}) ;
    end
    % get the H matrix with a best score
    [score, best] = max(score);
    H = H{best} ;
end

% HELPER METHOD - join 2 images based on homography model.
function mosaic = stitch(H,im1,im2)
    box = [1  size(im2,2) size(im2,2)  1 ;
        1  1           size(im2,1)  size(im2,1) ;
        1  1           1            1 ] ;
    box_ = inv(H) * box ;
    box_(1,:) = box_(1,:) ./ box_(3,:) ;
    box_(2,:) = box_(2,:) ./ box_(3,:) ;
    ur = min([1 box_(1,:)]):max([size(im1,2) box_(1,:)]) ;
    vr = min([1 box_(2,:)]):max([size(im1,1) box_(2,:)]) ;

    [u,v] = meshgrid(ur,vr) ;
        
    im1_ = vl_imwbackward(im2double(im1),u,v) ;
    
    z_ = H(3,1) * u + H(3,2) * v + H(3,3) ;
    u_ = (H(1,1) * u + H(1,2) * v + H(1,3)) ./ z_ ;
    v_ = (H(2,1) * u + H(2,2) * v + H(2,3)) ./ z_ ;
    im2_ = vl_imwbackward(im2double(im2),u_,v_) ;
    
    mass = ~isnan(im1_) + ~isnan(im2_) ;
    im1_(isnan(im1_)) = 0 ;
    im2_(isnan(im2_)) = 0 ;
    % This step is required dividing by 
    mosaic = (im1_ + im2_) ./ mass ;
end
end
