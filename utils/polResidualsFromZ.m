function [ residuals,J ] = polResidualsFromZ( z,D_x,D_y,eta,s,albedo,pol_angles,I_obs,spec,smooth_type,bd_Dx,bd_Dy,bd_weight,bd_penalty,smooth_L,smooth_weight, smooth_counter, smooth_kernel)
%polResidualsFromZ Residuals between observed intensities and diffuse
%polarisation model
%
% Inputs:
%   z is vector of height values of length npix
%   D_x*z computes x gradient of height, similarly for D_y and y
%   eta index of refraction
%   s point light source
%   albedo scalar or vector of length npix containing albedo
%   pol_angles vector of length npolang containing polariser orientations
%   in radians
%   I_obs is a vector of length npix*npolang with all intensities for first
%   polariser angle first, followed by all for second angle etc.


% Compute gradients
G = [D_x; D_y]*z;
% Compute polarisation image from gradients:
[F,J_F] = G2F(G,eta,s,albedo,spec);
% Compute intensities from polarisation image:
allI = [];
allJ_I = [];
for i=1:length(pol_angles)
    [I,J_I] = F2I(F,pol_angles(i));
    allI = [allI; I];
    allJ_I = [allJ_I; J_I];
end
% Derivatives of intensities with respect to gradient:
J_I_F = allJ_I*J_F; % Dimension: NpixNpol * 2Npix
% Derivatives of intensities with respect to surface height:
J_I_Z = J_I_F*[D_x; D_y]; % Dimension NpixNpol * Npix
% Derivatives of residuals with respect to surface height:
J = -speye(size(J_I_Z,1))*J_I_Z;
residuals = I_obs-allI;

if nargin > 9
    if strcmp(smooth_type,'over_smooth')
        % Add residuals and derivatives for Laplacian smoothness
        residuals = [residuals; smooth_weight.*smooth_L*z];
        % Add derivatives for smoothness term
        J = [J;smooth_weight*smooth_L];
    elseif strcmp(smooth_type,'fine_tune')
        % sum((localz - mean(localz)) ./ sqrt(sum((localz - mean(localz).^2))) .* local_kernel )
        localZs = smooth_L * z;

        [smoothness, J_smooth] = correlation(localZs, smooth_counter, smooth_kernel);
        % Add residuals and derivatives for Laplacian smoothness
        residuals = [residuals; smooth_weight.*(smoothness-1)];
        % Add derivatives for smoothness term
        
%         J_ressmt_smtL = sparse(1:nsmt_pix,1:nsmt_pix,-1./(smooth_L*z).^2,nsmt_pix,nsmt_pix);
%         J = [J;smooth_weight*J_ressmt_smtL*smooth_L];
        J = [J;smooth_weight*J_smooth*smooth_L];
        
    end


    % Add residuals and derivatives for convex Boundary
    % only G(gredients) on boundary pixels
    nbd_pix = size(bd_Dx,1);
    G_bd = [bd_Dx; bd_Dy]*z;
    [A, J_A] = G2A(G_bd);
    J_bd_Z = J_A * [bd_Dx;bd_Dy];

    diff_alpha = [cos(A(1:nbd_pix)); sin(A(1:nbd_pix))] - [cos(bd_penalty(1:nbd_pix)); sin(bd_penalty(1:nbd_pix))];
    diff_theta = A(nbd_pix+1:end)/1000 - bd_penalty(nbd_pix+1:end);
    % find small angels < pi between two
    % add 2*pi on angles greater than pi
    % only element-wise operations
    residuals = [residuals; [bd_weight(1:nbd_pix);bd_weight].*[diff_alpha;diff_theta]];
    % Add derivatives for smoothness term
%     J = [J;bd_weight.*J_bd_Z];
    J_resbd_bd = [sparse(1:nbd_pix,1:nbd_pix,bd_weight(1:nbd_pix).*(-sin(A(1:nbd_pix))),nbd_pix,nbd_pix), sparse(nbd_pix,nbd_pix); ...
        sparse(1:nbd_pix,1:nbd_pix,bd_weight(1:nbd_pix).*cos(A(1:nbd_pix)),nbd_pix,nbd_pix), sparse(nbd_pix,nbd_pix); ...
        sparse(nbd_pix,nbd_pix), sparse(1:nbd_pix,1:nbd_pix, bd_weight(nbd_pix+1:end)/1000, nbd_pix, nbd_pix)];
    J = [J;J_resbd_bd*J_bd_Z];

end

residuals(isnan(residuals)) = 1;
residuals(residuals==inf | residuals==-inf) = 1;
J(isnan(J)) = 1;
J(J==inf | J==-inf) = 1;



end

function [A, J_A] = G2A(G)
% Compute theta and alpha from surface gradients
% Output is vector of length 2*npix

% Compute unnormalised surface normal vectors
[ N,J_N ] = G2N(G);
% Compute normalised surface normal vectors
[barN,J_barN] = N2barN(N);
% Derivative of normalised normals with respect to gradients:
J_barN_N = J_barN*J_N;

% DoP
% Compute zenith angle
[theta,J_theta] = barN2theta(barN);
% Phase
% Compute azimuth angle:
[alpha,J_alpha] = barN2alpha(barN);

A = [alpha; theta] ;
J_A = [J_alpha*J_barN_N; J_theta*J_barN_N];

end


function [I,J_I] = F2I(F,pol_ang)
% Function to transform polarisation image into intensity given a polariser
% angle. F is a vector containing all phis, followed by all i_un followed
% by all rho.

npix = size(F,1)/3;

phi = F(1:npix);
i_un = F(npix+1:2*npix);
rho = F(2*npix+1:3*npix);

I = i_un.*(1+rho.*cos(2*pol_ang - 2.*phi));

J_I = [sparse(1:npix,1:npix,2.*i_un.*rho.*sin(2.*pol_ang-2.*phi),npix,npix) ...
       sparse(1:npix,1:npix,1+rho.*cos(2.*pol_ang-2.*phi),npix,npix) ...
       sparse(1:npix,1:npix,i_un.*cos(2.*pol_ang-2.*phi),npix,npix)];

end

function [ N,J_N ] = G2N(G)
% Function to transform vector containing gradients: [p_1, ..., p_n, q_1,
% ..., q_n]^T into vector containing unnormalised surface normal vectors:
% [n_x1, ..., n_xn, n_y1, ..., n_yn, n_z1, ..., n_zn]^T and also return the
% Jacobian matrix of this function

npix = size(G,1)/2;

N = [-G; ones(npix,1)];

J_N = [-speye(npix) sparse(npix,npix); ...
       sparse(npix,npix) -speye(npix); ...
       sparse(npix,2*npix)];
end

function [barN,J_barN] = N2barN(N)
% Function to transform vector containing unnormalised surface normal vectors:
% [n_x1, ..., n_xn, n_y1, ..., n_yn, n_z1, ..., n_zn]^T and return
% normalised vectors in vector of same size. Also return Jacobian matrix of
% this function.

npix = size(N,1)/3;

Nx = N(1:npix);
Ny = N(npix+1:2*npix);
Nz = N(2*npix+1:3*npix);

norms = sqrt(Nx.^2 + Ny.^2 + Nz.^2);
normsc = norms.^3;

barN = [Nx./norms; ...
        Ny./norms; ...
        Nz./norms];

% Computation for derivative:
% syms nx ny nz normn real
% n = [nx; ny; nz]
% eye(3)/normn-(n*n')./normn^3
% [ 1/normn - nx^2/normn^3,       -(nx*ny)/normn^3,       -(nx*nz)/normn^3]
% [       -(nx*ny)/normn^3, 1/normn - ny^2/normn^3,       -(ny*nz)/normn^3]
% [       -(nx*nz)/normn^3,       -(ny*nz)/normn^3, 1/normn - nz^2/normn^3]
J_barN = [sparse(1:npix,1:npix,1./norms - Nx.^2 ./ normsc,npix,npix) ...
          sparse(1:npix,1:npix,-(Nx.*Ny)./normsc,npix,npix) ...
          sparse(1:npix,1:npix,-(Nx.*Nz)./normsc,npix,npix);
          sparse(1:npix,1:npix,-(Nx.*Ny)./normsc,npix,npix) ...
          sparse(1:npix,1:npix,1./norms - Ny.^2 ./ normsc,npix,npix) ...
          sparse(1:npix,1:npix,-(Ny.*Nz)./normsc,npix,npix);
          sparse(1:npix,1:npix,-(Nx.*Nz)./normsc,npix,npix) ...
          sparse(1:npix,1:npix,-(Ny.*Nz)./normsc,npix,npix) ...
          sparse(1:npix,1:npix,1./norms - Nz.^2 ./ normsc,npix,npix)];
end

function [theta,J_theta] = barN2theta(barN)
% Convert surface normal to zenith angle

npix = size(barN,1)/3;

Nz = barN(2*npix+1:3*npix);
theta = acos(Nz);

J_theta = [sparse(npix,2*npix) sparse(1:npix,1:npix,-1./sqrt(1-Nz.^2),npix,npix)];

end

function [rho,J_rho] = theta2rho(theta,eta,spec)
% Convert zenith angle to (diffuse) degree of polarisation given index of 
% refraction

% separate diffuse and specular pixels
diffuse = not(spec);

npix = size(theta,1);

% diffuse
rhoD = (sin(theta(diffuse)).^2.*(eta-1/eta)^2)./(4.*cos(theta(diffuse)).*sqrt(eta^2-sin(theta(diffuse)).^2)-sin(theta(diffuse)).^2*(eta+1/eta)^2+2*eta^2+2);
% Obtained by:
% diff((sin(theta(diffuse))^2*(eta-1/eta)^2)/(4*cos(theta(diffuse))*sqrt(eta^2-sin(theta(diffuse))^2)-sin(theta(diffuse))^2*(eta+1/eta)^2+2*eta^2+2),theta(diffuse))
J_rhoD = (2.*cos(theta(diffuse)).*sin(theta(diffuse)).*(eta - 1/eta).^2)./(4.*cos(theta(diffuse)).*(eta.^2 - sin(theta(diffuse)).^2).^(1/2) - sin(theta(diffuse)).^2.*(eta + 1/eta).^2 + 2*eta^2 + 2) + (sin(theta(diffuse)).^2.*(eta - 1/eta).^2.*(4.*sin(theta(diffuse)).*(eta^2 - sin(theta(diffuse)).^2).^(1/2) + 2.*cos(theta(diffuse)).*sin(theta(diffuse)).*(eta + 1/eta).^2 + (4.*cos(theta(diffuse)).^2.*sin(theta(diffuse)))./(eta^2 - sin(theta(diffuse)).^2).^(1/2)))./(4.*cos(theta(diffuse)).*(eta^2 - sin(theta(diffuse)).^2).^(1/2) - sin(theta(diffuse)).^2.*(eta + 1/eta).^2 + 2*eta^2 + 2).^2;

% specular
rhoS = (2.*sin(theta(spec)).^2.*cos(theta(spec)).*sqrt(eta.^2-sin(theta(spec)).^2))./(eta.^2-sin(theta(spec)).^2-eta.^2.*sin(theta(spec)).^2+2.*sin(theta(spec)).^4);
J_rhoS = (2.*sin(theta(spec)).^3.*(eta^2 - sin(theta(spec)).^2).^(1/2))./(sin(theta(spec)).^2 - 2.*sin(theta(spec)).^4 + eta^2.*sin(theta(spec)).^2 - eta^2) + (2*cos(theta(spec)).^2.*sin(theta(spec)).^3)./((eta^2 - sin(theta(spec)).^2).^(1/2).*(sin(theta(spec)).^2 - 2.*sin(theta(spec)).^4 + eta^2.*sin(theta(spec)).^2 - eta^2)) - (4.*cos(theta(spec)).^2.*sin(theta(spec)).*(eta^2 - sin(theta(spec)).^2).^(1/2))./(sin(theta(spec)).^2 - 2.*sin(theta(spec)).^4 + eta^2.*sin(theta(spec)).^2 - eta^2) + (2.*cos(theta(spec)).*sin(theta(spec)).^2.*(eta^2 - sin(theta(spec)).^2).^(1/2).*(2*cos(theta(spec)).*sin(theta(spec)) - 8.*cos(theta(spec)).*sin(theta(spec)).^3 + 2.*eta^2.*cos(theta(spec)).*sin(theta(spec))))./(eta^2.*sin(theta(spec)).^2 - eta^2 - 2.*sin(theta(spec)).^4 + sin(theta(spec)).^2).^2;

% merge
rho = zeros(npix,1);
J_rho_s = zeros(npix,1);

rho(diffuse) = rhoD;
rho(spec) = rhoS;

J_rho_s(diffuse) = J_rhoD;
J_rho_s(spec) = J_rhoS;
J_rho = sparse(1:npix,1:npix,J_rho_s,npix,npix);

end

function [alpha,J_alpha] = barN2alpha(barN)
% Convert surface normal to azimuth angle

npix = size(barN,1)/3;

Nx = barN(1:npix);
Ny = barN(npix+1:2*npix);

alpha = atan2(Ny,Nx);

demoninator = Nx.^2+Ny.^2+10e-6;
J_alpha = [sparse(1:npix,1:npix,-Ny./demoninator,npix,npix) ...
           sparse(1:npix,1:npix,Nx./demoninator,npix,npix) ...
           sparse(npix,npix) ];

end

function [phi,J_phi] = alpha2phi(alpha,spec)
% Convert azimuth to phase angle

% separate diffuse and specular pixels
diffuse = not(spec);

npix = length(alpha);

% diffuse
phiD = mod(alpha(diffuse),pi);
% specular
phiS = mod(alpha(spec),pi)+pi;
% merge
phi = zeros(npix,1);
phi(diffuse) = phiD;
phi(spec) = phiS;

J_phi = speye(npix);

end

function [i_un,J_i_un] = barN2iun(barN,s,albedo,spec)
% Compute unpolarised intensity assuming Lambertian point source model.
% albedo can be scalar or vector of length npix

npix = size(barN,1)/3;

Nx = barN(1:npix);
Ny = barN(npix+1:2*npix);
Nz = barN(2*npix+1:3*npix);

albedo = albedo .* ones(npix,1);
albedo(spec) = 2^16-1; % specular albedo for uint16
% albedo(spec) = 1; % specular albedo for double ~ [0,1]
i_un = albedo.*(Nx.*s(1) + Ny.*s(2) + Nz.*s(3));

% rectified linear unit
% pos = (i_un./sqrt(i_un.^2)+1)/2;
% i_un = pos .* i_un;

i_un = (i_un + sqrt(i_un.^2))/2;

% temp = i_un.*(1./(2.*(i_un.^2).^(1/2)) - i_un.^2./(2.*(i_un.^2).^(3/2))) + i_un./(2.*(i_un.^2).^(1/2)) + 1/2;
temp = i_un./(2.*(i_un.^2).^(1/2)) + 1/2;

J_i_un = [sparse(1:npix,1:npix,temp.*albedo.*s(1),npix,npix) ...
    sparse(1:npix,1:npix,temp.*albedo.*s(2),npix,npix) ...
    sparse(1:npix,1:npix,temp.*albedo.*s(3),npix,npix)];

end

function [F,J_F] = G2F(G,eta,s,albedo,spec)
% Compute polarisation image from surface gradients and index of refraction
% Output is vector of length 3*npix

% Compute unnormalised surface normal vectors
[ N,J_N ] = G2N(G);
% Compute normalised surface normal vectors
[barN,J_barN] = N2barN(N);
% Derivative of normalised normals with respect to gradients:
J_barN_N = J_barN*J_N;
% Compute zenith angle
[theta,J_theta] = barN2theta(barN);
% Derivative of zenith angle with respect to gradients:
J_theta_barN_N = J_theta*J_barN_N;
% Compute degree of polarisation:
[rho,J_rho] = theta2rho(theta,eta,spec);
% Derivative of degree of polarisation with respect to gradients:
J_rho_theta_barN_N = J_rho*J_theta_barN_N;
% Compute azimuth angle:
[alpha,J_alpha] = barN2alpha(barN);
% Compute phase angle:
[phi,J_phi] = alpha2phi(alpha,spec);
% Derivative of phase angle with respect to gradients:
J_phi_alpha_barN_N = J_phi*J_alpha*J_barN_N;
% Compute unpolarised intensity:
[i_un,J_i_un] = barN2iun(barN,s,albedo,spec);
% Derivative of unpol intensity with respect to gradients:
J_i_un_barN_N = J_i_un*J_barN_N;

% Polarisation image as long vector:
F = [phi; i_un; rho];
% Jacobian of polarisation image with respect to gradients:
J_F = [J_phi_alpha_barN_N; J_i_un_barN_N; J_rho_theta_barN_N];

end


%% correlation: function description
function [corrCoeff, J_corr] = correlation(z, z_size, kernel)

% nblocks = size(z_size,1);
npix = length(z);

corrCoeff = [];
J_finalz_centralz = sparse(npix,npix);
J_corr_finalz = [];
J_centralz_localz = sparse(npix,npix);
processed=0;

% process blocks in parallel
block_type{1} = find(z_size(:,1)==4);
block_type{2} = find(z_size(:,1)==5);
block_type{3} = find(z_size(:,1)==6);
block_type{4} = find(z_size(:,1)==7);
block_type{5} = find(z_size(:,1)==8);
block_type{6} = find(z_size(:,1)==9);

for i = 1:length(block_type)
    if isempty(block_type{i})
        continue
    end
    
    % number of blocks in this type
    nblocks = length(block_type{i});
    % find all z's indices for current block type
    block_ind = [];
    % first stack in index for start pixel
    block_ind = [block_ind, z_size(block_type{i},2)];
    for j = 1:i+2
        % block_ind is a matrix having size nblocks-by-(i+3)
        block_ind = [block_ind, z_size(block_type{i},2)+j];
    end
    block_ind = block_ind';

    % localz is matrix (i+3)-by-nblocks
    localz = z(block_ind);
    centralz = bsxfun(@minus, localz, mean(localz,1));
    finalz = bsxfun(@times, centralz, 1./sqrt(sum(centralz.^2,1)));
    finalz_denominator = 1./sqrt(sum(centralz.^2,1));

    local_kernel = kernel(block_ind);
    local_kernel = bsxfun(@minus, local_kernel, mean(local_kernel,1));
    local_kernel = bsxfun(@times, local_kernel, 1./sqrt(sum(local_kernel.^2,1)));

    corrCoeff = [corrCoeff; sum(finalz.*local_kernel, 1)'];

    m = repmat((1:nblocks),i+3,1);
    m = m(:);
    n = block_ind(:);
    s = local_kernel;
    s = s(:);
    J_corr_finalz = [J_corr_finalz; sparse(m,n,s,nblocks,npix)];
    
    m = repmat(block_ind(:)', i+2, 1);
    m = m(:);
    n=[];
    temp1 = [];
    temp2 = [];
    for j =1:i+3
        n = [n;block_ind(1:j-1,:)];
        n = [n;block_ind(j+1:end,:)];
        temp1 = [temp1; centralz(1:j-1,:)];
        temp1 = [temp1; centralz(j+1:end,:)];
        temp2 = [temp2; repmat(centralz(j,:),i+2,1)];
    end
    n = n(:);
    s1 = repmat(finalz_denominator,i+3,1) - centralz.^2.*repmat(finalz_denominator,i+3,1).^3;
    s2 = -(temp1.*temp2).*repmat(finalz_denominator,(i+2)*(i+3),1).^3;

    J_finalz_centralz = J_finalz_centralz + sparse(m,n,s2(:),npix,npix)+sparse(block_ind(:),block_ind(:),s1(:),npix,npix);
    
    m = repmat(block_ind(:)',i+3,1);
    m = m(:);
    n = repmat(block_ind,i+3,1);
    n = n(:);
    J_centralz_localz = J_centralz_localz + sparse(m,n,-1/(i+3),npix,npix)+sparse(block_ind(:),block_ind(:),1,npix,npix);

end

J_corr = J_corr_finalz * J_finalz_centralz * J_centralz_localz;

end



