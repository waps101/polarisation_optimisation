addpath('utils')

load bunnyheight.mat
mask = (~(z==-inf));
z(not(mask)) = nan;
depth_gt = z;

% Store copy of original mask to reuse each iteration
maskoriginal = mask;

%% Experimental parameters

angles = deg2rad([0:30:180]);
eta = 1.5;

bitdepth = 8;
% noises=[0 0.005 0.01 0.02];
noises=[0];

% Light source directions - we probably want to change these
% Ss = [sind(15) 0 cosd(15);
%     -sind(15) 0 cosd(15);
%     0 sind(15) cosd(15);
%     0 -sind(15) cosd(15);
%     sind(30) 0 cosd(30);
%     -sind(30) 0 cosd(30);
%     0 sind(30) cosd(30);
%     0 -sind(30) cosd(30);
%     sind(60) 0 cosd(60);
%     -sind(60) 0 cosd(60);
%     0 sind(60) cosd(60);
%     0 -sind(60) cosd(60)];
Ss = [sind(15) 0 cosd(15)];

% Specular parameters
specular_albedo = 0.2;
shininess = 200;

% Number of times to repeat each experiment with random noise
noiserepeats = 1;

% Flag to determine whether to use constant albedo
constantalbedo = true;

opts = optimoptions(@lsqnonlin,'Jacobian','on','Display','iter','MaxIter',200);

%% Prepare constants
if constantalbedo
    % Uniform albedo
    albedo = 1-specular_albedo;
else
    % Use Lena image as albedo
    albedo = imresize(rgb2gray(im2double(imread('lena_std.tif'))),0.5);
end

%% Main loop

errors_height = zeros(size(Ss,1),length(noises));
errors_normal = zeros(size(Ss,1),length(noises));

for light=1:size(Ss,1)
    for noisenum=1:length(noises)
        % Select light and noise level for this experiment
        noise = noises(noisenum);
        s = Ss(light,:)';
        for repeat = 1:noiserepeats
            % azimuth angle system for our method
            angles = deg2rad([0:30:180]);
            
            % Simulate mixed polarisation (i.e. spec + diffuse)
            [ images,mask,spec,Iun ] = simulateMixedPol( depth_gt,maskoriginal,albedo,s,eta,angles,specular_albedo,shininess );

            % Add noise to images
            [ images ] = corruptImages( images,bitdepth,noises(noisenum) );
            
            %% depth estimate
            imgs = images;
            imgs = imgs.*(2^16-1);
            specular_mask = spec;
            theta_pol = angles;

            figure; imshow(imgs(:,:,1), [])
            drawnow()
            
            [height, width] = size(mask);
            large_side = max(height, width);

            clear small_imgs
            downscale = 256/8;
            small_mask = mask(1:downscale:end, 1:downscale:end);
            small_specular_mask = specular_mask(1:downscale:end, 1:downscale:end);
            for i = 1:size(imgs,3)
                temp = imgs(:,:,i);
                small_imgs(:,:,i) = temp(1:downscale:end, 1:downscale:end);
            end

            opt_weights.scale = 0.05;
            opt_weights.smooth = 3;
            opt_weights.bd = 3;

            [small_height1, J] = SfPol_ratio(small_imgs, theta_pol, small_mask, small_specular_mask, 'over_smooth', opt_weights);

            while downscale ~=2
                clear small_imgs
                init_z = small_height1;
                init_z(isnan(init_z)) = 0;
                downscale = downscale/2;
                for i = 1:size(imgs,3)
                    temp = imgs(:,:,i);
                    small_imgs(:,:,i) = temp(1:downscale:end, 1:downscale:end);
                end
                small_mask = mask(1:downscale:end, 1:downscale:end);
                small_specular_mask = specular_mask(1:downscale:end, 1:downscale:end);

                [small_height1, J] = SfPol_ratio(small_imgs, theta_pol, small_mask, small_specular_mask, 'over_smooth', opt_weights, init_z);
                drawnow()
            end
%             [small_height1, J] = SfPol_ratio(small_imgs, theta_pol, small_mask, small_specular_mask, 'over_smooth', opt_weights, small_height1);
%             drawnow()

            opt_weights.scale = 500;
            opt_weights.smooth = 5;
            opt_weights.bd = .5;


            if constantalbedo
                % uniform albedo
                [small_height2, J] = SfPol_full(small_imgs, theta_pol, small_mask, false(size(small_specular_mask)), 'light_est', opt_weights, small_height1);
            else
                % varying albedo
                [small_height2, J] = SfPol_full(small_imgs, theta_pol, small_mask, false(size(small_specular_mask)), 'albedo_est', opt_weights, small_height1, s);
            end
            
            
            
            %% draw normal map
            tempM = not(isnan(small_height1));
            comp_result(small_height1, tempM)
            comp_result(small_height2, tempM)
            
        end
    end
end
