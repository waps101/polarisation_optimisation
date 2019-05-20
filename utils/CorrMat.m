function [L, counter] = CorrMat(mask, zNeigh)
% extract local area used in correlation calculation
% zNeigh is 9-npix matrix containing pixel indices over mask

% find rows and cols for target pixels
ind = zNeigh(1,:);
[rows, cols] = ind2sub(size(mask), ind);

nrows = size(mask,1);
ncols = size(mask,2);

% The number of usable pixels
npix = sum(mask(:));
nequ = length(ind);

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

i = zeros(nequ*9*9,1);
j = zeros(nequ*9*9,1);
s = zeros(nequ*9*9,1);
k=0;
counter=[];
for equ = 1:nequ
    col=cols(equ);
    row=rows(equ);
    last_k = k;

    % cental pixel
    k=k+1;
    i(k)=k; j(k)=indices(row,col); s(k)=1;
    if mask(row-1,col-1)
    % up left
        k=k+1;
        i(k)=k; j(k)=indices(row-1,col-1); s(k)=1;
    end
    % left
    if mask(row,col-1)
        k=k+1;
        i(k)=k; j(k)=indices(row,col-1); s(k)=1;
    end
    if mask(row+1,col-1)
    % down left
        k=k+1;
        i(k)=k; j(k)=indices(row+1,col-1); s(k)=1;
    end
    if mask(row-1,col)
    % up
        k=k+1;
        i(k)=k; j(k)=indices(row-1,col); s(k)=1;
    end
    if mask(row+1,col)
    % down
        k=k+1;
        i(k)=k; j(k)=indices(row+1,col); s(k)=1;
    end
    if mask(row-1,col+1)
    % up right
        k=k+1;
        i(k)=k; j(k)=indices(row-1,col+1); s(k)=1;
    end
    if mask(row,col+1)
    % right
        k=k+1;
        i(k)=k; j(k)=indices(row,col+1); s(k)=1;
    end
    if mask(row+1,col+1)
    % down right
        k=k+1;
        i(k)=k; j(k)=indices(row+1,col+1); s(k)=1;
    end

    % [block_size, start_pix]
    counter = [counter; k-last_k, last_k+1];

end
i=i(1:k,1);
j=j(1:k,1);
s=s(1:k,1);
L = sparse(i,j,s,k,npix);


