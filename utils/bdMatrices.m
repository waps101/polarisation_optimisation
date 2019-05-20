function [ Dx,Dy ] = bdMatrices( mask,method,bd_ind )
%GRADMATRICES Compute matrices to evaluate gradients on pixels on mask boundary
%   Method:
%       'SmoothedCentral' - Convolution of central difference kernel with
%       Gaussian smoothing kernel, if not available use central, then
%       forward, then backward differences
%   mask:
%       mask for captured object
%   bd_ind:
%       indices of boundary pixel on mask


% mask is used for finding availible neighbours
mask = bwareaopen(mask,5);

% indMask is used for finding central pixel for each equation
indMask = [];
for i = 1:length(bd_ind)
    indMask = [indMask; bd_ind{i}];
end
% rows and cols for target pixels in padded mask
[rows, cols] = ind2sub(size(mask), indMask);
rows = rows+1;
cols = cols+1;

[nrows, ncols] = size(mask);
nrows = nrows+2;
ncols = ncols+2;

% Pad to avoid boundary problems
mask = pad(mask);

% The number of usable pixels and boundary pixel
nbd_pix = length(indMask);
npix = sum(mask(:));


% Build lookup table relating x,y coordinate of valid pixels to index
% position in vectorised representation
count = 0;
indices = zeros(size(mask));
for col=1:ncols
    for row=1:nrows
        if mask(row,col)
            count=count+1;
            indices(row,col)=count;
        end
    end
end


if strcmp(method,'SmoothedCentral')
    % Preallocate maximum required space
    % This would be if all valid pixels had equations for all 8 neighbours for
    % all possible pairs of images - it will be less in practice
    i = zeros(nbd_pix*6,1); % equation/boundary pixels indices
    j = zeros(npix*6,1); % neighbour pixels indices
    s = zeros(npix*6,1); % parameters used in calculating gradients
    k=0;
    NumEq=0;
    
    % X derivatives
    for pix = 1:nbd_pix
        col = cols(pix);
        row = rows(pix);
        if mask(row,col-1) && mask(row,col+1)
            % Both X neighbours present
            if mask(row-1,col-1) && mask(row-1,col+1) && mask(row+1,col-1) && mask(row+1,col+1)
                % All 6 neighbours present - smoothed
                % central differences
                NumEq=NumEq+1;
                % Edge neighbours
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col-1); s(k)=-(4/12);
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col+1); s(k)=(4/12);
                % Corner neighbours
                k=k+1;
                i(k)=NumEq; j(k)=indices(row-1,col-1); s(k)=-(1/12);
                k=k+1;
                i(k)=NumEq; j(k)=indices(row-1,col+1); s(k)=(1/12);
                k=k+1;
                i(k)=NumEq; j(k)=indices(row+1,col-1); s(k)=-(1/12);
                k=k+1;
                i(k)=NumEq; j(k)=indices(row+1,col+1); s(k)=(1/12);
            else
                % Central difference
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col-1); s(k)=-1/2;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col+1); s(k)=1/2;
            end

        elseif mask(row,col-1)
            % Only backward in X
            NumEq=NumEq+1;
            k=k+1;
            i(k)=NumEq; j(k)=indices(row,col-1); s(k)=-1;
            k=k+1;
            i(k)=NumEq; j(k)=indices(row,col); s(k)=1;

        elseif mask(row,col+1)
            % Only forward in X
            NumEq=NumEq+1;
            k=k+1;
            i(k)=NumEq; j(k)=indices(row,col+1); s(k)=1;
            k=k+1;
            i(k)=NumEq; j(k)=indices(row,col); s(k)=-1;
        end
        % Finished with a pixel
    end
    i=i(1:k,1);
    j=j(1:k,1);
    s=s(1:k,1);
    Dx = sparse(i,j,s,nbd_pix,npix);
    
    % Preallocate maximum required space
    % This would be if all valid pixels had equations for all 8 neighbours for
    % all possible pairs of images - it will be less in practice
    i = zeros(nbd_pix*6,1);
    j = zeros(npix*6,1);
    s = zeros(npix*6,1);
    k=0;
    NumEq=0;
    
    
    % Y Derivatives
    for pix = 1:nbd_pix
        col = cols(pix);
        row = rows(pix);
        if mask(row-1,col) && mask(row+1,col)
            % Both Y neighbours present
            if mask(row-1,col-1) && mask(row-1,col+1) && mask(row+1,col-1) && mask(row+1,col+1)
                % All 6 neighbours present - smoothed
                % central differences
                NumEq=NumEq+1;
                % Edge neighbours
                k=k+1;
                i(k)=NumEq; j(k)=indices(row-1,col); s(k)=(4/12);
                k=k+1;
                i(k)=NumEq; j(k)=indices(row+1,col); s(k)=-(4/12);
                % Corner neighbours
                k=k+1;
                i(k)=NumEq; j(k)=indices(row-1,col-1); s(k)=(1/12);
                k=k+1;
                i(k)=NumEq; j(k)=indices(row-1,col+1); s(k)=(1/12);
                k=k+1;
                i(k)=NumEq; j(k)=indices(row+1,col-1); s(k)=-(1/12);
                k=k+1;
                i(k)=NumEq; j(k)=indices(row+1,col+1); s(k)=-(1/12);
            else
                % Central difference
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row-1,col); s(k)=1/2;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row+1,col); s(k)=-1/2;
            end

        elseif mask(row-1,col)
            % Only backward in Y
            NumEq=NumEq+1;
            k=k+1;
            i(k)=NumEq; j(k)=indices(row,col); s(k)=-1;
            k=k+1;
            i(k)=NumEq; j(k)=indices(row-1,col); s(k)=1;

        elseif mask(row+1,col)
            % Only forward in Y
            NumEq=NumEq+1;
            k=k+1;
            i(k)=NumEq; j(k)=indices(row,col); s(k)=1;
            k=k+1;
            i(k)=NumEq; j(k)=indices(row+1,col); s(k)=-1;
        end
        % Finished with a pixel
    end
    i=i(1:k,1);
    j=j(1:k,1);
    s=s(1:k,1);
    Dy = sparse(i,j,s,nbd_pix,npix);
    
elseif strcmp(method,'Backward')
        % Preallocate maximum required space
    % This would be if all valid pixels had equations for all 8 neighbours for
    % all possible pairs of images - it will be less in practice
    i = zeros(nbd_pix*2,1);
    j = zeros(npix*2,1);
    s = zeros(npix*2,1);
    k=0;
    NumEq=0;
    
    % X derivatives
    for pix = 1:nbd_pix
        col = cols(pix);
        row = rows(pix);
        if mask(row,col-1)
            % Only backward in X
            NumEq=NumEq+1;
            k=k+1;
            i(k)=NumEq; j(k)=indices(row,col-1); s(k)=-1;
            k=k+1;
            i(k)=NumEq; j(k)=indices(row,col); s(k)=1;

        elseif mask(row,col+1)
            % Only forward in X
            NumEq=NumEq+1;
            k=k+1;
            i(k)=NumEq; j(k)=indices(row,col+1); s(k)=1;
            k=k+1;
            i(k)=NumEq; j(k)=indices(row,col); s(k)=-1;
        end
        % Finished with a pixel
    end
    i=i(1:k,1);
    j=j(1:k,1);
    s=s(1:k,1);
    Dx = sparse(i,j,s,npix,npix);
    
    % Preallocate maximum required space
    % This would be if all valid pixels had equations for all 8 neighbours for
    % all possible pairs of images - it will be less in practice
    i = zeros(npnbd_pixix*2,1);
    j = zeros(npix*2,1);
    s = zeros(npix*2,1);
    k=0;
    NumEq=0;
    
    % Y Derivatives
    for pix = 1:nbd_pix
        col = cols(pix);
        row = rows(pix);
        if mask(row-1,col)
            % Only backward in Y
            NumEq=NumEq+1;
            k=k+1;
            i(k)=NumEq; j(k)=indices(row,col); s(k)=-1;
            k=k+1;
            i(k)=NumEq; j(k)=indices(row-1,col); s(k)=1;

        elseif mask(row+1,col)
            % Only forward in Y
            NumEq=NumEq+1;
            k=k+1;
            i(k)=NumEq; j(k)=indices(row,col); s(k)=1;
            k=k+1;
            i(k)=NumEq; j(k)=indices(row+1,col); s(k)=-1;
        end
        % Finished with a pixel
    end
    i=i(1:k,1);
    j=j(1:k,1);
    s=s(1:k,1);
    Dy = sparse(i,j,s,nbd_pix,npix);
end

end

