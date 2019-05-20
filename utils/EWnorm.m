function Xnorm = EWnorm( X )
%EWNORM Element-wise normalisation of an m*n array of 3D vectors
%   X: M by N by 3 array, where you wish to normalise X(i,j,:) to unit
%   length

norms = sqrt(X(:,:,1).^2+X(:,:,2).^2+X(:,:,3).^2);

Xnorm = X./repmat(norms,[1 1 3]);

end

