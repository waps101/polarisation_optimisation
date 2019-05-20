function [ L ] = LaplacianMat( mask )
%LAPLACIANMAT compute matrices to apply surface smoothnesss constraint
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

i = zeros(npix*5,1);
j = zeros(npix*5,1);
s = zeros(npix*5,1);
k=0;
NumEq=0;
for col=1:cols
    for row=1:rows
        if mask(row,col) 
            % four available neighbours
            if mask(row-1,col) && mask(row+1,col) && mask(row,col+1) && mask(row,col-1)
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col-1); s(k)=1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col+1); s(k)=1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row-1,col); s(k)=1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row+1,col); s(k)=1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col); s(k)=-4;
            

            % three available neighbours
            % no up
            elseif not(mask(row-1,col)) && mask(row+1,col) && mask(row,col+1) && mask(row,col-1)
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col-1); s(k)=1.33;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col+1); s(k)=1.33;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row+1,col); s(k)=1.33;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col); s(k)=-4;    

            % no down
            elseif mask(row-1,col) && not(mask(row+1,col)) && mask(row,col+1) && mask(row,col-1)
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col-1); s(k)=1.33;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col+1); s(k)=1.33;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row-1,col); s(k)=1.33;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col); s(k)=-4;

            % no left
            elseif mask(row-1,col) && mask(row+1,col) && mask(row,col+1) && not(mask(row,col-1))
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col+1); s(k)=1.33;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row-1,col); s(k)=1.33;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row+1,col); s(k)=1.33;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col); s(k)=-4;
                
            % no right
            elseif mask(row-1,col) && mask(row+1,col) && not(mask(row,col+1)) && mask(row,col-1)
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col-1); s(k)=1.33;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row-1,col); s(k)=1.33;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row+1,col); s(k)=1.33;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col); s(k)=-4;
               

            % two available neighbours   
            % no up & down
            elseif not(mask(row-1,col)) && not(mask(row+1,col)) && mask(row,col+1) && mask(row,col-1)
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col-1); s(k)=2;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col+1); s(k)=2;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col); s(k)=-4;

            % no up & left
            elseif not(mask(row-1,col)) && mask(row+1,col) && mask(row,col+1) && not(mask(row,col-1))
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col+1); s(k)=2;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row+1,col); s(k)=2;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col); s(k)=-4;

            % no up & right
            elseif not(mask(row-1,col)) && mask(row+1,col) && not(mask(row,col+1)) && mask(row,col-1)
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col-1); s(k)=2;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row+1,col); s(k)=2;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col); s(k)=-4;

            % no down & left
            elseif mask(row-1,col) && not(mask(row+1,col)) && mask(row,col+1) && not(mask(row,col-1))
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col+1); s(k)=2;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row-1,col); s(k)=2;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col); s(k)=-4;
                
            % np down & right    
            elseif mask(row-1,col) && not(mask(row+1,col)) && not(mask(row,col+1)) && mask(row,col-1)
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col-1); s(k)=2;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row-1,col); s(k)=2;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col); s(k)=-4;

            % no left & right
            elseif mask(row-1,col) && mask(row+1,col) && not(mask(row,col+1)) && not(mask(row,col-1))
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row-1,col); s(k)=2;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row+1,col); s(k)=2;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col); s(k)=-4;


            % one available neighbour
            % only up
            elseif mask(row-1,col) && not(mask(row+1,col)) && not(mask(row,col+1)) && not(mask(row,col-1))
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col-1); s(k)=4;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col); s(k)=-4;
            % only down
            elseif not(mask(row-1,col)) && mask(row+1,col) && not(mask(row,col+1)) && not(mask(row,col-1))
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row+1,col); s(k)=4;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col); s(k)=-4;
            % only left
            elseif not(mask(row-1,col)) && not(mask(row+1,col)) && not(mask(row,col+1)) && mask(row,col-1)
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col-1); s(k)=4;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col); s(k)=-4;
                
            % only right    
            elseif not(mask(row-1,col)) && not(mask(row+1,col)) && mask(row,col+1) && not(mask(row,col-1))    
                NumEq=NumEq+1;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col+1); s(k)=4;
                k=k+1;
                i(k)=NumEq; j(k)=indices(row,col); s(k)=-4;
            end
        end
    end
end
i=i(1:k,1);
j=j(1:k,1);
s=s(1:k,1);
L = sparse(i,j,s,npix,npix);

end

