function [ images ] = corruptImages( images,bitdepth,noise )
%CORRUPTIMAGES Summary of this function goes here
%   Detailed explanation goes here

% Add Gaussian image noise
    images = images+randn(size(images)).*noise;
    % Saturate
    images(images>1)=1;
    images(images<0)=0;
    % Quantize
    images = round(images.*(2^bitdepth-1))./(2^bitdepth-1);
    
end

