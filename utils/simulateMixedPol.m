function [ images,mask,spec,Iun ] = simulateMixedPol( z,mask,albedo,s,n,angles,specular_albedo,shininess )
%SIMULATEMIXEDPOL Summary of this function goes here
%   Detailed explanation goes here

% Compute gradients
[ Dx,Dy,mask ] = gradMatrices( mask,'Backward' );
p = Dx*z(mask);
q = Dy*z(mask);

% Compute surface normals
Nx = zeros(size(mask));
Nx(mask)=-p;
Ny = zeros(size(mask));
Ny(mask)=-q;
N = ones(size(mask,1),size(mask,2),3);
N(:,:,1)=Nx;
N(:,:,2)=Ny;
N = EWnorm(N);

% Compute polarisation image
theta = acos(N(:,:,3));
phi = mod(atan2(N(:,:,2),N(:,:,1)),pi);
im_spec=render_BP( N,zeros(size(mask)),specular_albedo,shininess,s );
im_diffuse=render_diffuse(N,albedo,s);
Iun = im_spec+im_diffuse;

mask = mask&(Iun>0);
mask = bwareaopen(mask,100);
rho_d = ((n-1./n).^2.*sin(theta).^2)./(2+2.*n.^2-(n+1./n).^2.*sin(theta).^2+4.*cos(theta).*sqrt(n.^2-sin(theta).^2));
phi_s = mod(phi+pi/2,pi);
rho_s = (2.*sin(theta).^2.*cos(theta).*sqrt(n.^2-sin(theta).^2))./(n.^2-sin(theta).^2-n.^2.*sin(theta).^2+2.*sin(theta).^4);
ImaxplusImin_d = 2.*im_diffuse;
ImaxminusImin_d = rho_d.*ImaxplusImin_d;
ImaxplusImin_s = 2.*im_spec;
ImaxminusImin_s = rho_s.*ImaxplusImin_s;
images = NaN(size(mask,1),size(mask,2),length(angles));
for i=1:length(angles)
    for row=1:size(mask,1)
        for col=1:size(mask,2)
            if mask(row,col)
                diff_trs = im_diffuse(row,col)+(ImaxminusImin_d(row,col)./2).*cos(2.*angles(i)-2.*phi(row,col));
                spec_trs = im_spec(row,col)+(ImaxminusImin_s(row,col)./2).*cos(2.*angles(i)-2.*phi_s(row,col));
                images(row,col,i)=(diff_trs+spec_trs);
            end
        end
    end
end

% spec = im_spec>im_diffuse;
spec = im_spec>.1;

end

