function [ X2 ] = pad( X )
%PAD Summary of this function goes here
%   Detailed explanation goes here

[rows,cols]=size(X);
X2 = zeros(rows+2,cols+2);
X2(2:rows+1,2:cols+1)=X(:,:);
if islogical(X)
    X2 = logical(X2);
end

end

