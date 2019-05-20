function image = render_diffuse( normals,albedo,s )
%RENDER_DIFFUSE Render normals or hybrid normals with diffuse reflectance
%   normals: M*N*3 or M*N*3*3 matrix of normals
%   albedo:  M*N*3 matrix of diffuse albedo values
%   s:       3D vector containing point light source direction
s = s./norm(s);
if ndims(normals)==4
    for chan=1:3
        image(:,:,chan)=(normals(:,:,1,chan).*s(1)+normals(:,:,2,chan).*s(2)+normals(:,:,3,chan).*s(3)).*albedo(:,:,chan);
    end
else
    if ndims(albedo)==2
        image = albedo.*(normals(:,:,1).*s(1)+normals(:,:,2).*s(2)+normals(:,:,3).*s(3));
    else
        image = albedo.*repmat((normals(:,:,1).*s(1)+normals(:,:,2).*s(2)+normals(:,:,3).*s(3)),[1 1 3]);

    end
end
image = max(0,image);
end

