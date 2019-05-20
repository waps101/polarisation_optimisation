function [ images,mask ] = simulateDiffusePol( z,mask,albedo,s,eta,angles )
%SIMULATEDIFFUSEPOL Summary of this function goes here
%   Detailed explanation goes here

[ Dx,Dy,mask ] = gradMatrices( mask,'Backward' );
p = Dx*z(mask);
q = Dy*z(mask);

Nx = zeros(size(mask));
Nx(mask)=-p;
Ny = zeros(size(mask));
Ny(mask)=-q;
N = ones(size(mask,1),size(mask,2),3);
N(:,:,1)=Nx;
N(:,:,2)=Ny;
N = EWnorm(N);

theta = acos(N(:,:,3));
phi = mod(atan2(N(:,:,2),N(:,:,1)),pi);

Iun=render_diffuse(N,albedo.*ones(size(mask)),s);
mask = mask&(Iun>0);
mask = bwareaopen(mask,2048);
clear images
rho = ((eta-1./eta).^2.*sin(theta).^2)./(2+2.*eta.^2-(eta+1./eta).^2.*sin(theta).^2+4.*cos(theta).*sqrt(eta.^2-sin(theta).^2));
ImaxplusImin_d = 2.*Iun;
ImaxminusImin_d = rho.*ImaxplusImin_d;
images = NaN(size(mask,1),size(mask,2),length(angles));
for i=1:length(angles)
    for row=1:size(mask,1)
        for col=1:size(mask,2)
            if mask(row,col)
                diff_trs = Iun(row,col)+(ImaxminusImin_d(row,col)./2).*cos(2.*angles(i)-2.*phi(row,col));
                images(row,col,i)=(diff_trs);
            end
        end
    end
end

end

