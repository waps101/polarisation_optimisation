function im = render_BP( N,A,rho_s,eta,L )
%RENDER_BP Render with Blinn-Phong model
%   Detailed explanation goes here

diffuse = render_diffuse( N,A,L );
H = L+[0 0 1]'; % half vector
H = H./norm(H);
specular = render_diffuse( N,1,H ).^eta.*rho_s;

im = specular+diffuse;

end

