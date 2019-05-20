function [height1, Jacob] = SfPol_full(img, theta_pol, mask, spec_mask, process_type, opt_weights, init_z, s)
% Shape-from-polarisation full model
% Inputs:
%   img: image matrix with shape H*W*Nimgs
%   theta_pol: polariser angles for input images
%   mask: object mask
%   spec_mask: mask on specularity
%   process_type: {'light_est', 'albedo_est'}
%        'light_est': estimate light source firstly, followed by SfP optimisation, which is suitable for uniform albedo 
%        'albedo_est': given light source, algorithm estimate albedo firstly, followed by SfP optimisation, which is suitable for varying albedo and known light source
%   init_z: initial guess of height map


% define coordinates system
[x, y] = meshgrid(1:size(mask,2), size(mask,1):-1:1);

% gradient calculation matrix and refine mask
[ D_x,D_y,mask ] = gradMatrices( mask,'SmoothedCentral' );
spec_mask = spec_mask(mask);



% process boundary pixels
[bd_penalty, ind, weights] = boundaryPpg(mask);

% locate boundary pixels as indices
indMask = [];
for i = 1:length(ind)
    indMask = [indMask; ind{i}];
end


% indices of most outside boundary pix
bd = ind{1};

% transform raw image cell into a long vector containing masked pixs only
imgVector = [];
for i = 1:size(img,3)
    temp = img(:,:,i);
    imgVector = [imgVector; temp(mask)];
end

% pre-process mask to calculated necessary parameter matrix
[ bd_Dx,bd_Dy ] = bdMatrices( mask,'SmoothedCentral',ind );

% define optimisation mode
options1 = optimoptions( 'lsqnonlin', 'Jacobian', 'on', 'display', 'iter', 'MaxIterations', 10, 'FunctionTolerance', 1e-4, 'StepTolerance', 1e-4 );
options2 = optimoptions( 'lsqnonlin', 'Jacobian', 'on', 'display', 'iter', 'MaxIterations', 50, 'FunctionTolerance', 1e-4, 'StepTolerance', 1e-4 );

    

if strcmp(process_type, 'albedo_est')
    G = [D_x; D_y]*init_z(mask);
    albedo = albedo_est(G, mask, img, theta_pol, s);
    save albedo.mat albedo
    albedo = load('albedo.mat');
    albedo = albedo.albedo;
    
    % setup new configuration for optimisation
    zNeigh = kernelFromModel(mask);
    [ smooth_L,smooth_counter ] = CorrMat( mask, zNeigh );
    
    % convert to vector
    opt_z1 = init_z(mask);
    % construct a vector stacking local area of over-smoothed result
    smooth_kernel = smooth_L * init_z(mask);
    
    [init_cost, ~] = polResidualsFromZ( opt_z1,D_x,D_y,1.5,s,albedo,theta_pol,imgVector,spec_mask);
    cost_scale = mean(abs(init_cost))/opt_weights.scale;

    % smooth_weight = cost_scale*50*mean(abs(init_cost)); % weights for non-cost_scale angel=15,watergun=3,bunny=15
    smooth_weight = cost_scale*opt_weights.smooth*mean(abs(init_cost)); % weights for non-cost_scale angel=15,watergun=3,bunny=15
    bd_weight = cost_scale*opt_weights.bd*mean(abs(init_cost)).*weights;% angel=0.5,bear=.1
    
    for iter = 1:5
        [opt_z1,~,~,exitflag,~,~, Jacob ] = lsqnonlin( @(z)polResidualsFromZ( z,D_x,D_y,1.5,s,albedo,theta_pol,imgVector,spec_mask,'fine_tune',bd_Dx,bd_Dy,bd_weight,bd_penalty,smooth_L,smooth_weight,smooth_counter,smooth_kernel ), opt_z1(:), [], [], options1);

        if exitflag ~= 0
            break
        end
        [temp_cost, ~] = polResidualsFromZ( opt_z1,D_x,D_y,1.5,s,albedo,theta_pol,imgVector,spec_mask);
        smooth_weight = cost_scale*opt_weights.smooth*mean(abs(temp_cost)); % 0.01
        bd_weight = cost_scale*opt_weights.bd*mean(abs(temp_cost)).*weights; % 0.001
    end
    
elseif strcmp(process_type, 'light_est')
    G = [D_x; D_y]*init_z(mask);
    light = light_est(G, mask, img, theta_pol);
    save light.mat light
    % light = load('light.mat');
    % light = light.light;
    
    % setup new configuration for optimisation
    zNeigh = kernelFromModel(mask);
    [ smooth_L,smooth_counter ] = CorrMat( mask, zNeigh );
    
    % convert to vector
    opt_z1 = init_z(mask);
    % construct a vector stacking local area of over-smoothed result
    smooth_kernel = smooth_L * init_z(mask);
    
    [init_cost, ~] = polResidualsFromZ( opt_z1,D_x,D_y,1.5,light,1,theta_pol,imgVector,spec_mask);
    cost_scale = mean(abs(init_cost))/opt_weights.scale;
    smooth_weight = cost_scale*opt_weights.smooth*mean(abs(init_cost)); % weights for non-cost_scale angel=15,watergun=3,bunny=15
    bd_weight = cost_scale*opt_weights.bd*mean(abs(init_cost)).*weights;% angel=0.5,bear=.1
    
    for iter = 1:5
        [opt_z1,~,~,exitflag,~,~, Jacob ] = lsqnonlin( @(z)polResidualsFromZ( z,D_x,D_y,1.5,light,1,theta_pol,imgVector,spec_mask,'fine_tune',bd_Dx,bd_Dy,bd_weight,bd_penalty,smooth_L,smooth_weight,smooth_counter,smooth_kernel ), opt_z1(:), [], [], options1);

        if exitflag ~= 0
            break
        end
        [temp_cost, ~] = polResidualsFromZ( opt_z1,D_x,D_y,1.5,light,1,theta_pol,imgVector,spec_mask);
        smooth_weight = cost_scale*opt_weights.smooth*mean(abs(temp_cost));
        bd_weight = cost_scale*opt_weights.bd*mean(abs(temp_cost)).*weights;
    end
end
    
z = zeros(size(mask));
z(mask) = opt_z1;
height2 = reshape(z, size(mask));
% height2(not(mask)) = 0;
height1 = height2;
height1(not(mask)) = nan;
figure
surf(x, y, height1)
title 'Optimal Result'
axis equal


function albedo = albedo_est(G, mask, img, theta_pol, s)

npix = size(G,1)/2;
N = [-G; ones(npix,1)];

Nx = N(1:npix);
Ny = N(npix+1:2*npix);
Nz = N(2*npix+1:3*npix);
norms = sqrt(Nx.^2 + Ny.^2 + Nz.^2);
barN = [Nx./norms; ...
        Ny./norms; ...
        Nz./norms];

Nz = barN(2*npix+1:3*npix);
theta = acos(Nz);
Nx = barN(1:npix);
Ny = barN(npix+1:2*npix);
alpha = atan2(Ny,Nx);


eta = 1.5;

phi_v = alpha;
rho_v = (sin(theta) .* sin(theta) * (eta - 1/eta)^2) ./ (4 * cos(theta) .* sqrt(eta^2 - sin(theta).^2) ...
    - sin(theta).^2 * (eta + 1/eta)^2 + 2*eta^2 + 2);

phi = zeros(size(mask));
rho = zeros(size(mask));
n = zeros(numel(mask),3);
phi(mask) = phi_v;
rho(mask) = rho_v;
n(mask,:) = [Nx, Ny, Nz];

albedo = [];
for i = 1:length(theta_pol)
	factors = 1 + rho .* cos( 2*( theta_pol(i) - phi ) );
    image = img(:,:,i);
	iun = image./factors;
    non_shadow = iun(:)>0 & mask(:)>0;
    temp = zeros(size(iun));
    temp(non_shadow) = iun(non_shadow)./(n(non_shadow,:) * s);
    albedo(:,:,i) = temp;
end

albedo = mean(albedo,3);
albedo = albedo(mask);


function light = light_est(G, mask, img, theta_pol)

npix = size(G,1)/2;
N = [-G; ones(npix,1)];

Nx = N(1:npix);
Ny = N(npix+1:2*npix);
Nz = N(2*npix+1:3*npix);
norms = sqrt(Nx.^2 + Ny.^2 + Nz.^2);
barN = [Nx./norms; ...
        Ny./norms; ...
        Nz./norms];

Nz = barN(2*npix+1:3*npix);
theta = acos(Nz);
Nx = barN(1:npix);
Ny = barN(npix+1:2*npix);
alpha = atan2(Ny,Nx);


eta = 1.5;

phi_v = alpha;
rho_v = (sin(theta) .* sin(theta) * (eta - 1/eta)^2) ./ (4 * cos(theta) .* sqrt(eta^2 - sin(theta).^2) ...
    - sin(theta).^2 * (eta + 1/eta)^2 + 2*eta^2 + 2);

phi = zeros(size(mask));
rho = zeros(size(mask));
n = zeros(numel(mask),3);
phi(mask) = phi_v;
rho(mask) = rho_v;
n(mask,:) = [Nx, Ny, Nz];

light = [];
for i = 1:length(theta_pol)
	factors = 1 + rho .* cos( 2*( theta_pol(i) - phi ) );
    image = img(:,:,i);
	iun = image./factors;
    non_shadow = iun(:)>0 & mask(:)>0;
    light(:,i) = n(non_shadow,:) \ iun(non_shadow);
end
light = mean(light,2);


function [boundary, ind, weights] = boundaryPpg(mask)
% penalise boundary zs pointing inwards

bdAlpha = [];
bdTheta = [];
ind = [];
weights = [];
% [theta, alpha, ~] = de_pol(z, mask);

% find 2nd and 3rd most outside boundary layers
temp_mask = mask;
layers = ceil(size(mask,1)/4);
% for i = 1:layers
i=0;
while true
    i=i+1;
    boundaries = bwboundaries(temp_mask, 4, 'noholes');
    if isempty(boundaries)
        break
    end
    b = [];
    ref_b_next = [];
    ref_b_last = [];
    for j = 1:length(boundaries)
        if size(boundaries{j},1)<=2
            continue
        end
        temp_b = boundaries{j};
        b = [b; temp_b(1:end-1,:)];
        ref_b_next = [ref_b_next; temp_b(2:end-1,:); temp_b(1,:)];
        ref_b_last = [ref_b_last; temp_b(end-1,:); temp_b(1:end-2,:)];
    end
    
    if isempty(b)
        break
    end
    
    % end pixel on thin boundary
    branchEnd = ref_b_next(:,1)==ref_b_last(:,1) & ref_b_next(:,2)==ref_b_last(:,2);
    
    % find thin boundary that is one-way route
    [~,bd_pixs_ind,~] = unique(b, 'rows', 'stable');
    dup_bd_ind = not(ismember(1:size(b,1), bd_pixs_ind));
    dup_bd = b(dup_bd_ind,:);
    dup_bd = unique(dup_bd, 'rows', 'stable');% duplicate pixels

    redun_bd_ind = {};
    bad_bd_ind = [];
    bad_bd_start = [];
    for j = 1:size(dup_bd,1)
        % index of duplicate pixels in b
        dup_bd_ind = find(b(:,1)==dup_bd(j,1)&b(:,2)==dup_bd(j,2));
        bad_bd_ind(j) = dup_bd_ind(1); % first of duplicate pixels
        redun_bd_ind = [redun_bd_ind, {dup_bd_ind(2:end)'}]; % redundance of duplicate pixels
    end
    
    % find out start pixel of thin boundary
    % start pixel of thin boundary need to be treated distinctively from
    % thin boundary
    temp1 = cell2mat(redun_bd_ind);
    if not(isempty(redun_bd_ind))
        temp2 = diff(temp1);
        bad_bd_start = b(temp1([temp2~=1, true]),:); % pixel of start of thin boundary
        [~,~,bad_bd_start] = intersect(bad_bd_start, dup_bd, 'rows', 'stable'); % rows number in dup_bd matrix for pixels
    end
    % take out start pixel from redun_bd_ind and bad_bd_ind
    bad_bd_start_ind = bad_bd_ind(bad_bd_start);
    bad_bd_ind(bad_bd_start) = []; % bad_bd_ind has the same height with dup_bd
    redun_bd_start_ind = redun_bd_ind(bad_bd_start);
    redun_bd_ind(bad_bd_start) = [];
    
    % transfer indexing vector to indicating vector
    % bad_bd pixels should be treated individually
    bad_bd = zeros(size(b,1),1);
    bad_bd(bad_bd_ind) = 1;

    % save indices of extracted boundary points based on subsripts
    % remove redundant pixels on thin boundary(redun_bd_ind)
    temp_ind = sub2ind(size(mask), b(bd_pixs_ind,1), b(bd_pixs_ind,2));
    temp_mask(temp_ind) = 0;
    ind= [ind; {temp_ind}];

    p = ref_b_next(:,2) - b(:,2) + b(:,2) - ref_b_last(:,2);
    q = -(ref_b_next(:,1) - b(:,1) + b(:,1) - ref_b_last(:,1));
    % pixels on branch_end dont have two neighbours, so use one
    p(branchEnd) = b(branchEnd,2) - ref_b_last(branchEnd,2);
    q(branchEnd) = b(branchEnd,1) - ref_b_last(branchEnd,1);

    temp_bdTheta = pi/2 * ones(size(q));
    temp_bdTheta([cell2mat(redun_bd_ind),cell2mat(redun_bd_start_ind)]) = [];
    temp_bdAlpha = atan2(q,p);
    temp_bdAlpha = temp_bdAlpha + pi/2; % rotate alpha to perpendicular direction
    temp_bdAlpha(bad_bd | branchEnd) = temp_bdAlpha(bad_bd | branchEnd) - pi/2;
    for k = 1:length(redun_bd_start_ind)
        temp_bdAlpha(bad_bd_start_ind(k)) = mean([temp_bdAlpha(bad_bd_start_ind(k)),temp_bdAlpha(redun_bd_start_ind{k})']);
    end

    temp_bdAlpha(temp_bdAlpha > pi) = temp_bdAlpha(temp_bdAlpha > pi) - 2*pi;
    temp_bdAlpha(temp_bdAlpha < -pi) = temp_bdAlpha(temp_bdAlpha < -pi) +2*pi;
    temp_bdAlpha([cell2mat(redun_bd_ind),cell2mat(redun_bd_start_ind)]) = [];
        
    weights = [weights; ones(length(temp_ind), 1)/i];

    bdTheta = [bdTheta; temp_bdTheta];
    bdAlpha = [bdAlpha; temp_bdAlpha];


    if sum(temp_mask(:)) == 0
        break
    end
end
weights = [weights; weights];
boundary = [bdAlpha; bdTheta/1000];



function zNeigh = kernelFromModel(sm_mask)
[gap_hzNeigh, ~] = size(sm_mask);

ind = find(sm_mask == 1);

neighbours = [ind-gap_hzNeigh-1, ind-gap_hzNeigh, ...
            ind-gap_hzNeigh+1, ind-1, ind+1, ...
            ind+gap_hzNeigh-1, ind+gap_hzNeigh, ind+gap_hzNeigh+1];

zNeigh = [ind, neighbours]'; % shape = 9*npix

% kernel_hs = {z(zNeigh), zNeigh, [0 -1 0; -1 -4 -1; 0 -1 0]};



