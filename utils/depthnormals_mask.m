function [N,Jac,Gx,Gy] = depthnormals_mask( z,mask )
%DEPTHNORMALS_MASK Compute normal map from depth map with mask
%   Uses finite difference approximation to surface graidents from supplied
%   depth map. For pixels with missing neighbours (according to binary
%   foreground mask), either central or simple forward/backward finite
%   differences are used.
%
% Inputs:
% z    - rows by cols depth map
% mask - rows by cols logical matrix, 1 for foreground, 0 for background
%
% Output:
% N    - rows by cols by 3 array containing surface normals at each pixel
%
% Please cite: W. A. P. Smith and F. Fang. "Height from Photometric Ratio
% with Model-based Light Source Selection." Computer Vision and Image
% Understanding (2015). if you use this code in your research.
%
% William A. P. Smith
% University of York
% 2015


if nargout<2
    IsJacobian=false;
else
    IsJacobian=true;
end

if nargout<3
    IsGradient=false;
else
    IsGradient=true;
end

rows = size(mask,1);
cols = size(mask,2);

offsetm=maskoffset(mask);
% Pad to avoid boundary problems
z2 = zeros(rows+2,cols+2);
z2(2:rows+1,2:cols+1)=z;
z = z2;
clear z2
mask2 = zeros(rows+2,cols+2);
mask2(2:rows+1,2:cols+1)=mask;
mask = mask2;
clear mask2
offsetm2=zeros(rows+2,cols+2);
offsetm2(2:rows+1,2:cols+1)=offsetm;
offsetm=offsetm2;
clear offsetm2;
rows = rows+2;
cols = cols+2;

p = zeros(size(mask));
q = zeros(size(mask));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Dizhong.Zhu revised for testing or others
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
noofPixelsUsed=sum(mask(:));
% sign matrix for gradient p and q seperately
% 1 means forward difference, -1 means backward difference
signP=ones(size(mask));
signQ=ones(size(mask));

% Construct a gradient matrix:
% Index form [pixel_index,position,value];
% If the second pixel used it self and the forward pixel for difference:
% It should be [2,2,-1]and[2,3,1];

% shrink after procedure finished
% No more than its double size, for forward or backward only
Gx_idx_val=zeros(noofPixelsUsed*2,3);  Gx_count=1;
Gy_idx_val=zeros(noofPixelsUsed*2,3);  Gy_count=1;

% No more than triple
jidx_v=zeros(noofPixelsUsed*3,1); jidx_v_count=1; % Jacobian matrix variable colume idx
jidx_f=zeros(noofPixelsUsed*3,1); jidx_f_count=1; % Jacobian martrix function row idx

for row=1:rows
    for col=1:cols
        if mask(row,col)
            % Now decide which combination of neighbours are present
            % This determines which version of the numerical
            % approximation to the surface gradients will be used
            
            if IsJacobian
                fidx=(rows-2)*(col-2)+row-1-offsetm(row,col); % jacobian function idx;
                vidx=(rows-2)*(col-2)+row-1-offsetm(row,col); % jacobian variable idx;
                %                 jidx_v=[jidx_v;vidx];
                %                 jidx_f=[jidx_f;fidx];
                jidx_v(jidx_v_count)=vidx; jidx_v_count=jidx_v_count+1;
                jidx_f(jidx_f_count)=fidx; jidx_f_count=jidx_f_count+1;
            end
            % x direction
            if mask(row,col+1)
                % Only forward in X
                p(row,col)=z(row,col+1)-z(row,col);
                
                if IsJacobian
                    %                     Gx_idx_val=[Gx_idx_val;[fidx,vidx,-1]];
                    Gx_idx_val(Gx_count,:)=[fidx,vidx,-1]; Gx_count=Gx_count+1;
                    vidx=(rows-2)*(col-1)+row-1-offsetm(row,col+1);
                    %                     jidx_v=[jidx_v;vidx];
                    %                     jidx_f=[jidx_f;fidx];
                    jidx_v(jidx_v_count)=vidx; jidx_v_count=jidx_v_count+1;
                    jidx_f(jidx_f_count)=fidx; jidx_f_count=jidx_f_count+1;
                    %                     Gx_idx_val=[Gx_idx_val;[fidx,vidx,1]];
                    Gx_idx_val(Gx_count,:)=[fidx,vidx,1]; Gx_count=Gx_count+1;
                end
            elseif mask(row,col-1)
                % Only backward in X
                p(row,col)=z(row,col)-z(row,col-1);
                if IsJacobian
                    %                     Gx_idx_val=[Gx_idx_val;[fidx,vidx,1]];
                    Gx_idx_val(Gx_count,:)=[fidx,vidx,1]; Gx_count=Gx_count+1;
                    vidx=(rows-2)*(col-3)+row-1-offsetm(row,col-1);
                    %                     jidx_v=[jidx_v;vidx];
                    %                     jidx_f=[jidx_f;fidx];
                    jidx_v(jidx_v_count)=vidx; jidx_v_count=jidx_v_count+1;
                    jidx_f(jidx_f_count)=fidx; jidx_f_count=jidx_f_count+1;
                    %                     Gx_idx_val=[Gx_idx_val;[fidx,vidx,-1]];
                    Gx_idx_val(Gx_count,:)=[fidx,vidx,-1]; Gx_count=Gx_count+1;
                end
                signP(row,col)=-1;
            end
            
            %y direction
            vidx=(rows-2)*(col-2)+row-1-offsetm(row,col);
            if mask(row+1,col)
                % Only forward in Y
                q(row,col)=z(row+1,col)-z(row,col);
                if IsJacobian
                    %                     Gy_idx_val=[Gy_idx_val;[fidx,vidx,-1]];
                    Gy_idx_val(Gy_count,:)=[fidx,vidx,-1]; Gy_count=Gy_count+1;
                    vidx=(rows-2)*(col-2)+row-offsetm(row+1,col);
                    %                     jidx_v=[jidx_v;vidx];
                    %                     jidx_f=[jidx_f;fidx];
                    jidx_v(jidx_v_count)=vidx; jidx_v_count=jidx_v_count+1;
                    jidx_f(jidx_f_count)=fidx; jidx_f_count=jidx_f_count+1;
                    %                     Gy_idx_val=[Gy_idx_val;[fidx,vidx,1]];
                    Gy_idx_val(Gy_count,:)=[fidx,vidx,1]; Gy_count=Gy_count+1;
                end
            elseif mask(row-1,col)
                % Only backward in Y
                q(row,col)=z(row,col)-z(row-1,col);
                if IsJacobian
                    %                     Gy_idx_val=[Gy_idx_val;[fidx,vidx,1]];
                    Gy_idx_val(Gy_count,:)=[fidx,vidx,1]; Gy_count=Gy_count+1;
                    vidx=(rows-2)*(col-2)+row-2-offsetm(row-1,col);
                    %                     jidx_v=[jidx_v;vidx];
                    %                     jidx_f=[jidx_f;fidx];
                    jidx_v(jidx_v_count)=vidx; jidx_v_count=jidx_v_count+1;
                    jidx_f(jidx_f_count)=fidx; jidx_f_count=jidx_f_count+1;
                    %                     Gy_idx_val=[Gy_idx_val;[fidx,vidx,-1]];
                    Gy_idx_val(Gy_count,:)=[fidx,vidx,-1]; Gy_count=Gy_count+1;
                end
                signQ(row,col)=-1;
            end
            % Finished with a pixel
        end
    end
end

p(~mask)=NaN;
q(~mask)=NaN;

N(:,:,1)=-p;
N(:,:,2)=-q;
N(:,:,3)=1;
% Normalise to unit vectors
norms = sqrt(sum(N.^2,3));
N = N./repmat(norms,[1 1 3]);
N = N(2:rows-1,2:cols-1,:);

% Construct a sparse gradient matrix
if IsJacobian
    jidx_f=jidx_f(1:jidx_f_count-1);
    jidx_v=jidx_v(1:jidx_v_count-1);
    Jac=sparse(jidx_f,jidx_v,ones(length(jidx_v),1),noofPixelsUsed,noofPixelsUsed);
end

if IsGradient
    Gx_idx_val=Gx_idx_val(1:Gx_count-1,:);
    Gy_idx_val=Gy_idx_val(1:Gy_count-1,:);
    Gx=sparse(Gx_idx_val(:,1),Gx_idx_val(:,2),Gx_idx_val(:,3),noofPixelsUsed,noofPixelsUsed);
    Gy=sparse(Gy_idx_val(:,1),Gy_idx_val(:,2),Gy_idx_val(:,3),noofPixelsUsed,noofPixelsUsed);
end

end
