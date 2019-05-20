function comp_result(z, mask)
% show comparison between gradient map of input and ground truth sphere

% [GTx, GTy] = meshgrid(-1:2/128:1, -1:2/128:1);
% GTmask = (GTx.^2 + GTy.^2)<=1;
% GTz = zeros(size(GTmask));
% GTz(GTmask) = sqrt(1 - GTx(GTmask).^2 - GTy(GTmask).^2);
% GTnx = GTx ./ sqrt(GTx.^2+GTy.^2+1);
% GTny = GTy ./ sqrt(GTx.^2+GTy.^2+1);
% GTnz = GTz ./ sqrt(GTx.^2+GTy.^2+1);
% % rescale it into range (0,1)
% GTnx = (GTnx+1)/2;
% GTny = (GTny+1)/2;
% GTnz = (GTnz+1)/2;
% GTn = cat(3,GTnx, GTny, GTnz);
% GTn = GTn .* repmat(GTmask, 1,1,3);


Gx = 1/12*[-1 0 1; -4 0 4; -1 0 1];
Gy = 1/12*[1 4 1; 0 0 0; -1 -4 -1];
p = conv2(z, Gx, 'same');
q = conv2(z, Gy, 'same');

[~, ind, ~] = boundaryPpg(mask);
bd = ind{1}; % indices of boundary
pad_mask = padarray(mask, [1,1], 'post'); % pad mask to make bd not lay on boudaries of mask
[rows, cols] = ind2sub(size(mask), bd);
bd_padmask = sub2ind(size(pad_mask), rows, cols);
gap_hNeigh = size(pad_mask, 1);
u = pad_mask(bd_padmask-1)==1;
l = pad_mask(bd_padmask-gap_hNeigh)==1;
r = pad_mask(bd_padmask+gap_hNeigh)==1;
d = pad_mask(bd_padmask+1)==1;
bdPixType = {u, l, r, d};
bdType = bd_type(mask, bd, bdPixType);
[bd_p, bd_q] = bd_grad(z, bd, bdType);
p(bd) = bd_p;
q(bd) = bd_q;

nx = p ./ sqrt(p.^2+q.^2+1);
ny = q ./ sqrt(p.^2+q.^2+1);
nz = 1 ./ sqrt(p.^2+q.^2+1);

% rescale it into range (0,1)
nx = (nx+1)/2;
ny = (ny+1)/2;
nz = (nz+1)/2;
n = cat(3, nx, ny, nz);
n = n .* repmat(mask, 1,1,3);

% figure; imshow(GTn)
figure; imshow(n)


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
    for i = 1:layers
        boundaries = bwboundaries(temp_mask, 4, 'noholes');
        b = [];
        ref_b_next = [];
        ref_b_last = [];
        for j = 1:length(boundaries)
    %     for j = 1:1
    %         if sum(ismember(sub2ind(size(mask),boundaries{j}(:,1),boundaries{j}(:,2)),find(specular_mask)))
    %             continue
    %         end
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
    %     dup_bd = mat2cell(dup_bd, ones(size(dup_bd,1),1), 2);
    %     redun_bd_ind = [];
        redun_bd_ind = {};
        bad_bd_ind = [];
        bad_bd_start = [];
        for j = 1:size(dup_bd,1)
            % index of duplicate pixels in b
            dup_bd_ind = find(b(:,1)==dup_bd(j,1)&b(:,2)==dup_bd(j,2));
            bad_bd_ind(j) = dup_bd_ind(1); % first of duplicate pixels
    %         redun_bd_ind = [redun_bd_ind, dup_bd_ind(2:end)']; % redundance of duplicate pixels
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
        q = ref_b_next(:,1) - b(:,1) + b(:,1) - ref_b_last(:,1);
        % pixels on branch_end dont have two neighbours, so use one
        p(branchEnd) = b(branchEnd,2) - ref_b_last(branchEnd,2);
        q(branchEnd) = b(branchEnd,1) - ref_b_last(branchEnd,1);
    
        temp_bdTheta = pi/2 * ones(size(q));
        temp_bdTheta([cell2mat(redun_bd_ind),cell2mat(redun_bd_start_ind)]) = [];
        temp_bdAlpha = atan2(p, q);
        temp_bdAlpha = temp_bdAlpha - pi/2; % rotate alpha to perpendicular direction
        temp_bdAlpha(bad_bd | branchEnd) = temp_bdAlpha(bad_bd | branchEnd) - pi/2;
        for k = 1:length(redun_bd_start_ind)
            temp_bdAlpha(bad_bd_start_ind(k)) = mean([temp_bdAlpha(bad_bd_start_ind(k)),temp_bdAlpha(redun_bd_start_ind{k})']);
    %         if temp_bdAlpha(bad_bd_start_ind(k))>pi
    %             temp_bdAlpha(bad_bd_start_ind(k)) = 2*pi - temp_bdAlpha(bad_bd_start_ind(k));
    %         end
        end
    %     temp_bdAlpha(bad_bd_ind(bad_bd_start_ind)) = temp_bdAlpha(bad_bd_ind(bad_bd_start_ind)) + pi/2;
    %     temp_bdAlpha(bad_bd_ind(bad_bd_start_ind)) = mean(temp_bdAlpha(bad_bd_ind(bad_bd_start_ind)), )
    %     temp_bdAlpha(branchEnd) = temp_bdAlpha(branchEnd) - pi/2;
        temp_bdAlpha(temp_bdAlpha > 2*pi) = temp_bdAlpha(temp_bdAlpha > 2*pi) - 2*pi;
        temp_bdAlpha(temp_bdAlpha < 0) = temp_bdAlpha(temp_bdAlpha < 0) +2*pi;
        temp_bdAlpha([cell2mat(redun_bd_ind),cell2mat(redun_bd_start_ind)]) = [];
    %     rm = [];
    %     for k = 1:length(rm_ind)
    %         temp_bdAlpha(bad_bd_ind(k)) = mean(temp_bdAlpha(bad_bd_ind(k)) + temp_bdAlpha(rm_ind{k}));
    %         rm = [rm, rm_ind{k}];
    %     end
    %     temp_bdTheta(rm) = [];
    %     temp_bdAlpha(rm) = [];
    %     
    %     temp_bdAlpha(temp_bdAlpha > 2*pi) = temp_bdAlpha(temp_bdAlpha > 2*pi) - 2*pi;
    %     temp_bdAlpha(temp_bdAlpha < 0) = temp_bdAlpha(temp_bdAlpha < 0) +2*pi;
            
        weights = [weights; ones(length(temp_ind), 1)/i];
    
        bdTheta = [bdTheta; temp_bdTheta];
        bdAlpha = [bdAlpha; temp_bdAlpha];
    
    
        if sum(temp_mask(:)) == 0
            break
        end
    end
    weights = [weights; weights];
    boundary = [bdTheta/100; bdAlpha];
    



function bdtype = bd_type(mask, bd, bdPixType)
    % gap_hNeigh = size(z, 1);
    bdtype = zeros(size(mask));
    
    % four main neighbour
    fourNeighPix = bd(bdPixType{1} & bdPixType{2} & bdPixType{3} & bdPixType{4});
    bdtype(fourNeighPix) = 4;
    
    % three main neighbour
    % no u
    threeNeighPix = bd(not(bdPixType{1}) & bdPixType{2} & bdPixType{3} & bdPixType{4});
    bdtype(threeNeighPix) = 31;
    % no l
    threeNeighPix = bd(not(bdPixType{2}) & bdPixType{1} & bdPixType{3} & bdPixType{4});
    bdtype(threeNeighPix) = 32;
    % no r
    threeNeighPix = bd(not(bdPixType{3}) & bdPixType{2} & bdPixType{1} & bdPixType{4});
    bdtype(threeNeighPix) = 33;
    % no d
    threeNeighPix = bd(not(bdPixType{4}) & bdPixType{2} & bdPixType{3} & bdPixType{1});
    bdtype(threeNeighPix) = 34;
    
    % two main neighbour
    % no u l
    twoNeighPix = bd(not(bdPixType{1}) & not(bdPixType{2}) & bdPixType{3} & bdPixType{4});
    bdtype(twoNeighPix) = 21;
    % no u r
    twoNeighPix = bd(not(bdPixType{1}) & not(bdPixType{3}) & bdPixType{2} & bdPixType{4});
    bdtype(twoNeighPix) = 22;
    % no u d
    twoNeighPix = bd(not(bdPixType{1}) & not(bdPixType{4}) & bdPixType{2} & bdPixType{3});
    bdtype(twoNeighPix) = 23;
    % no l r
    twoNeighPix = bd(not(bdPixType{3}) & not(bdPixType{2}) & bdPixType{1} & bdPixType{4});
    bdtype(twoNeighPix) = 24;
    % no l d
    twoNeighPix = bd(not(bdPixType{4}) & not(bdPixType{2}) & bdPixType{1} & bdPixType{3});
    bdtype(twoNeighPix) = 25;
    % no r d
    twoNeighPix = bd(not(bdPixType{3}) & not(bdPixType{4}) & bdPixType{1} & bdPixType{2});
    bdtype(twoNeighPix) = 26;
    
    % one main neighbour
    % only u
    oneNeighPix = bd(not(bdPixType{3}) & not(bdPixType{4}) & not(bdPixType{2}) & bdPixType{1});
    bdtype(oneNeighPix) = 11;
    % only l
    oneNeighPix = bd(not(bdPixType{3}) & not(bdPixType{4}) & not(bdPixType{1}) & bdPixType{2});
    bdtype(oneNeighPix) = 12;
    % only r
    oneNeighPix = bd(not(bdPixType{1}) & not(bdPixType{4}) & not(bdPixType{2}) & bdPixType{3});
    bdtype(oneNeighPix) = 13;
    % only d
    oneNeighPix = bd(not(bdPixType{3}) & not(bdPixType{1}) & not(bdPixType{2}) & bdPixType{4});
    bdtype(oneNeighPix) = 14;
    
    bdtype = bdtype(bd); % bdtype contains neighbour type code for boundary pixel, bd is boudnary pixel indices




function [bd_p, bd_q] = bd_grad(z, bd, bdType)
    gap_hNeigh = size(z, 1);
    output_p = zeros(size(z));
    output_q = zeros(size(z));
    
    pix = bd(bdType==4);
    %% four main neighbour
    output_p(pix) = (z(pix-gap_hNeigh) - z(pix+gap_hNeigh))/2;
    output_q(pix) = (z(pix+1) - z(pix-1))/2;
    
    %% three main neighbour
    % no u
    pix = bd(bdType==31);
    output_p(pix) = (z(pix-gap_hNeigh) - z(pix+gap_hNeigh))/2;
    output_q(pix) = (z(pix+1) - z(pix));
    % no l
    pix = bd(bdType==32);
    output_p(pix) = (z(pix) - z(pix+gap_hNeigh));
    output_q(pix) = (z(pix+1) - z(pix-1))/2;
    % no r
    pix = bd(bdType==33);
    output_p(pix) = (z(pix-gap_hNeigh) - z(pix));
    output_q(pix) = (z(pix+1) - z(pix-1))/2;
    % no d
    pix = bd(bdType==34);
    output_p(pix) = (z(pix-gap_hNeigh) - z(pix+gap_hNeigh))/2;
    output_q(pix) = (z(pix) - z(pix-1));
    
    %% two main neighbour
    % no u l
    pix = bd(bdType==21);
    output_p(pix) = (z(pix) - z(pix+gap_hNeigh));
    output_q(pix) = (z(pix+1) - z(pix));
    % no u r
    pix = bd(bdType==22);
    output_p(pix) = (z(pix-gap_hNeigh) - z(pix));
    output_q(pix) = (z(pix+1) - z(pix));
    % no u d
    pix = bd(bdType==23);
    output_p(pix) = (z(pix-gap_hNeigh) - z(pix+gap_hNeigh))/2;
    output_q(pix) = 0;
    % no l r
    pix = bd(bdType==24);
    output_p(pix) = 0;
    output_q(pix) = (z(pix+1) - z(pix-1))/2;
    % no l d
    pix = bd(bdType==25);
    output_p(pix) = (z(pix) - z(pix+gap_hNeigh));
    output_q(pix) = (z(pix) - z(pix-1));
    % no r d
    pix = bd(bdType==26);
    output_p(pix) = (z(pix-gap_hNeigh) - z(pix));
    output_q(pix) = (z(pix) - z(pix-1));
    
    %% one main neighbour
    % only u
    pix = bd(bdType==11);
    output_p(pix) = 0;
    output_q(pix) = (z(pix) - z(pix-1));
    % only l
    pix = bd(bdType==12);
    output_p(pix) = (z(pix-gap_hNeigh) - z(pix));
    output_q(pix) = 0;
    % only r
    pix = bd(bdType==13);
    output_p(pix) = (z(pix) - z(pix+gap_hNeigh));
    output_q(pix) = 0;
    % only d
    pix = bd(bdType==14);
    output_p(pix) = 0;
    output_q(pix) = (z(pix+1) - z(pix));
    
    %%
    bd_p = output_p(bd);
    bd_q = output_q(bd);