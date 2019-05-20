function [ L ] = SmoothMatrix( mask )
%LAPLACIANMAT Summary of this function goes here
%   Detailed explanation goes here

rows = size(mask,1);
cols = size(mask,2);

% Pad to avoid boundary problems
mask = pad(mask);

rows = rows+2;
cols = cols+2;

% The number of usable pixels
npix = sum(mask(:));

% Build lookup table relating x,y coordinate of valid pixels to index
% position in vectorised representation
count = 0;
indices = zeros(size(mask));
for col=1:cols
    for row=1:rows
        if mask(row,col)
            count=count+1;
            indices(row,col)=count;
        end
    end
end

h = [0 1 0; 1 0 1; 0 1 0];
mask4n = imfilter(1.*mask,h,'conv')==4;

h = ones(3,3);
mask3x3 = imfilter(1.*mask,h,'conv')==9;
[~, ~, ~, ~, SG] = SavGol(2,3);
SG3x3 = squeeze(SG(:,:,1));
SG3x3(2,2) = SG3x3(2,2)-1;

h = ones(5,5);
mask5x5 = imfilter(1.*mask,h,'conv')==25;
[~, ~, ~, ~, SG] = SavGol(4,5);
SG5x5 = squeeze(SG(:,:,1));
SG5x5(3,3) = SG5x5(3,3)-1;

i = zeros(npix*25,1);
j = zeros(npix*25,1);
s = zeros(npix*25,1);
k=0;
NumEq=0;
for col=1:cols
    for row=1:rows
        if mask(row,col)
            if mask5x5(row,col)
                NumEq=NumEq+1;
                for row2=1:5
                    for col2=1:5
                        k=k+1;
                        i(k)=NumEq; j(k)=indices(row+row2-3,col+col2-3); s(k)=SG5x5(row2,col2);
                    end
                end                
            elseif mask3x3(row,col)
                NumEq=NumEq+1;
                for row2=1:3
                    for col2=1:3
                        k=k+1;
                        i(k)=NumEq; j(k)=indices(row+row2-2,col+col2-2); s(k)=SG3x3(row2,col2);
                    end
                end
            elseif mask4n(row,col)
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col-1); s(k)=0.25;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col+1); s(k)=0.25;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row-1,col); s(k)=0.25;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row+1,col); s(k)=0.25;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col); s(k)=-1;
            end
        end
    end
end
i=i(1:k,1);
j=j(1:k,1);
s=s(1:k,1);
L = sparse(i,j,s,npix,npix);

end

