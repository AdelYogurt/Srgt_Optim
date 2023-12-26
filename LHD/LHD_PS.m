clc;
clear;
close all hidden;

sample_number = 12;
variable_number = 2;
low_bou = 0*ones(1,variable_number);
up_bou = 1*ones(1,variable_number);

% x_exist_list = [0.1,0.5;0.6,0.3];
x_exist_list = [];

cheapcon_function = @(x) conHole(x)
% cheapcon_function = [];

tic
[X_sample,dist_min_nomlz,X_total] = getLatinHypercube...
    (sample_number,variable_number,...
    low_bou,up_bou,x_exist_list,cheapcon_function);
toc
disp(['dist_min_nomlz:',num2str(dist_min_nomlz)])

axis equal;
% line(X_sample(:,1),zeros(sample_number,1),'linestyle','none','Marker','o','color','b');axis([0,1,-0.5,0.5]);
scatter(X_sample(:,1),X_sample(:,2),'color','b');axis([0,1,0,1]);
% scatter3(X_sample(:,1),X_sample(:,2),X_sample(:,3),'color','b');axis([0,1,0,1,0,1]);view(3);
grid on;

function [X_sample,dist_min_nomlz,X_total] = getLatinHypercube...
    (sample_number,variable_number,...
    low_bou,up_bou,X_exist,cheapcon_function)
% generate latin hypercube desgin
%
% more uniform point distribution by simulating particle motion
%
% input:
% sample number(new point to sample),variable_number
% low_bou,up_bou,x_exist_list,cheapcon_function
%
% output:
% X_sample,dist_min_nomlz(min distance of normalize data)
% X_total,include all data in area
%
% Copyright 2023 3 Adel
%
if nargin < 6
    cheapcon_function = [];
    if nargin < 5
        X_exist = [];
        if nargin < 4
            up_bou = ones(1,variable_number);
            if nargin < 3
                low_bou = zeros(1,variable_number);
                if nargin < 2
                    error('getLatinHypercube: lack variable_number');
                end
            end
        end
    end
end

iteration_max = 100;

% check x_exist_list if meet boundary
if ~isempty(X_exist)
    if size(X_exist,2) ~= variable_number
        error('getLatinHypercube: x_exist_list variable_number error');
    end
    index = find(X_exist < low_bou);
    index = [index,find(X_exist > up_bou)];
    if ~isempty(index)
        error('getLatinHypercube: x_exist_list range error');
    end
    X_exist_nomlz = (X_exist-low_bou)./(up_bou-low_bou);
else
    X_exist_nomlz = [];
end

exist_number = size(X_exist,1);
total_number = sample_number+exist_number;
if sample_number < 0
    X_total = X_exist;
    X_sample = [];
    dist_min_nomlz = getMinDistance(X_exist_nomlz);
    return;
end

low_bou_nomlz = zeros(1,variable_number);
up_bou_nomlz = ones(1,variable_number);

% obtain initial point
if ~isempty(cheapcon_function)
    % obtian feasible point
    X_quasi_nomlz = [];

    % check if have enough X_supply_nomlz
    iteration = 0;
    while size(X_quasi_nomlz,1) < 10*sample_number && iteration < 500
        X_quasi_nomlz_initial = lhsdesign(10*sample_number,variable_number);

        qusai_index = [];
        for x_index = 1:size(X_quasi_nomlz_initial,1)
            if cheapcon_function(X_quasi_nomlz_initial(x_index,:).*(up_bou-low_bou)+low_bou) <= 0
                qusai_index = [qusai_index,x_index];
            end
        end
        X_quasi_nomlz = [X_quasi_nomlz;X_quasi_nomlz_initial(qusai_index,:)];

        iteration = iteration+1;
    end

    if iteration == 500 && size(X_quasi_nomlz,1) < sample_number
        error('getLatinHypercube: feasible quasi point cannot be found');
    end

    % use fuzzy clustering to get feasible point center
    X_sample_nomlz = clusteringFuzzy(X_quasi_nomlz,sample_number,2);
    X_feas_center_nomlz = X_sample_nomlz;

    scatter(X_quasi_nomlz(:,1),X_quasi_nomlz(:,2));
    hold on;
    scatter(X_sample_nomlz(:,1),X_sample_nomlz(:,2),'red');
    hold off;
else
    X_sample_nomlz = rand(sample_number,variable_number);
end

% pic_num = 1;

iteration = 0;
fval_list = zeros(sample_number,1);
gradient_list = zeros(sample_number,variable_number);
while iteration < iteration_max    
    % change each x place by newton methods
    for x_index = 1:sample_number
        
        % get gradient
        [fval_list(x_index,1),gradient_list(x_index,:)] = calParticleEnergy...
            (X_sample_nomlz(x_index,:),[X_sample_nomlz(1:x_index-1,:);X_sample_nomlz(x_index+1:end,:);X_exist_nomlz],...
            sample_number,variable_number,low_bou_nomlz-1/2/sample_number,up_bou_nomlz+1/2/sample_number);
        
%         energy_function = @(x) calParticleEnergy...
%             ([x],[X_sample_nomlz(1:x_index-1,:);X_sample_nomlz(x_index+1:end,:);X_exist_nomlz],...
%             sample_number,variable_number,low_bou_nomlz-1/2/sample_number,up_bou_nomlz+1/2/sample_number);
%         drawFunction(energy_function,low_bou_nomlz(1:2),up_bou_nomlz(1:2))
        
%         [fval,gradient] = differ(energy_function,X_sample_nomlz(x_index,:));
%         gradient'-gradient_list(x_index,:)
    end

    C = (1-iteration/iteration_max/2)*0.5;
    
    % updata partical location
    for x_index = 1:sample_number
        x = X_sample_nomlz(x_index,:);
        gradient = gradient_list(x_index,:);

        % check if feasible
        if ~isempty(cheapcon_function)
            con = cheapcon_function(x.*(up_bou-low_bou)+low_bou);
            % if no feasible,move point to close point
            if con > 0
                %                 % search closest point
                %                 dx_center = x-X_feas_center_nomlz;
                %                 [~,index] = min(norm(dx_center,"inf"));
                %                 gradient = dx_center(index(1),:);

                gradient = x-X_feas_center_nomlz(x_index,:);
            end
        end

        gradient = min(gradient,0.5);
        gradient = max(gradient,-0.5);
        x = x-gradient*C;

        boolean = x < low_bou_nomlz;
        x(boolean) = -x(boolean);
        boolean = x > up_bou_nomlz;
        x(boolean) = 2-x(boolean);
        X_sample_nomlz(x_index,:) = x;
    end
    
    iteration = iteration+1;

    scatter(X_sample_nomlz(:,1),X_sample_nomlz(:,2));
    bou = [low_bou_nomlz(1:2);up_bou_nomlz(1:2)];
    axis(bou(:));
    grid on;
%     
%     radius = 1;
%     hold on;
%     rectangle('Position',[-radius,-radius,2*radius,2*radius],'Curvature',[1 1])
%     hold off;
%     
    drawnow;
%     F = getframe(gcf);
%     I = frame2im(F);
%     [I,map] = rgb2ind(I,256);
%     if pic_num  =  =  1
%         imwrite(I,map,'show_trajectory_constrain.gif','gif','Loopcount',inf,'DelayTime',0.1);
%     else
%         imwrite(I,map,'show_trajectory_constrain.gif','gif','WriteMode','append','DelayTime',0.1);
%     end
%     pic_num = pic_num + 1;
end

% process out of boundary point
for x_index = 1:sample_number
    x = X_sample_nomlz(x_index,:);
    % check if feasible
    if ~isempty(cheapcon_function)
        con = cheapcon_function(x);
        % if no feasible,move point to close point
        if con > 0
            % search closest point
            dx_center = x-X_feas_center_nomlz;
            [~,index] = min(norm(dx_center,"inf"));

            gradient = dx_center(index(1),:);
        end
    end
    x = x-gradient*C;

    boolean = x < low_bou_nomlz;
    x(boolean) = -x(boolean);
    boolean = x > up_bou_nomlz;
    x(boolean) = 2-x(boolean);
    X_sample_nomlz(x_index,:) = x;
end

dist_min_nomlz = getMinDistance([X_sample_nomlz;X_exist_nomlz]);
X_sample = X_sample_nomlz.*(up_bou-low_bou)+low_bou;
X_total = [X_sample;X_exist];

    function [fval,gradient] = calParticleEnergy...
            (x,X_surplus,sample_number,variable_number,low_bou,up_bou)
        % function describe distance between X and X_supply
        % x is colume vector and X_surplus is matrix which is num-1 x var
        % low_bou_limit__ and up_bou_limit__ is colume vector
        % variable in colume
        %
        a__ = 10;
        a_bou__ = 10;
        
        sign__ = ((x > X_surplus)-0.5)*2;
        
        xi__ = -a__*(x-X_surplus).*sign__;
        psi__ = a_bou__*(low_bou-x);
        zeta__ = a_bou__*(x-up_bou);
        
        exp_psi__ = exp(psi__);
        exp_zeta__ = exp(zeta__);

%         sum_xi__ = sum(xi__,2)/variable_number;
%         exp_sum_xi__ = exp(sum_xi__);
%         % get fval
%         fval = sum(exp_sum_xi__,1)+...
%             sum(exp_psi__+exp_zeta__,2)/variable_number;

%         exp_xi__ = exp(xi__);
%         sum_exp_xi__ = sum(exp_xi__,2);
%         % get fval
%         fval = sum(sum_exp_xi__,1)/variable_number/sample_number+...
%             sum(exp_psi__+exp_zeta__,2)/variable_number;

        sum_xi__ = sum(xi__,2)/variable_number;
        exp_sum_xi__ = exp(sum_xi__);
        exp_xi__ = exp(xi__);
        sum_exp_xi__ = sum(exp_xi__,2)/variable_number;
        % get fval
        fval = (sum(sum_exp_xi__,1)+sum(exp_sum_xi__,1))/2/sample_number+...
            sum(exp_psi__+exp_zeta__,2)/variable_number*0.1;
        
%         % get gradient
%         gradient = sum(-a__*sign__.*exp_sum_xi__,1)/variable_number+...
%             (-a_bou__*exp_psi__+a_bou__*exp_zeta__)/variable_number;

%         % get gradient
%         gradient = sum(-a__*sign__.*exp_xi__,1)/variable_number/sample_number+...
%             (-a_bou__*exp_psi__+a_bou__*exp_zeta__)/variable_number;

        % get gradient
        gradient = (sum(-a__*sign__.*exp_sum_xi__,1)+sum(-a__*sign__.*exp_xi__,1))/2/variable_number/sample_number+...
            (-a_bou__*exp_psi__+a_bou__*exp_zeta__)/variable_number*0.1;
        
    end

    function distance_min__ = getMinDistance(x_list__)
        % get distance min from x_list
        %
        if isempty(x_list__)
            distance_min__ = [];
            return;
        end

        % sort x_supply_list_initial to decrese distance calculate times
        x_list__ = sortrows(x_list__,1);
        sample_number__ = size(x_list__,1);
        variable_number__ = size(x_list__,2);
        distance_min__ = variable_number__;
        for x_index__ = 1:sample_number__
            x_curr__ = x_list__(x_index__,:);
            x_next_index__ = x_index__ + 1;
            % first dimension only search in min_distance
            search_range__ = variable_number__;
            while x_next_index__ <= sample_number__ &&...
                    (x_list__(x_next_index__,1)-x_list__(x_index__,1))^2 ...
                    < search_range__
                x_next__ = x_list__(x_next_index__,:);
                distance_temp__ = sum((x_next__-x_curr__).^2);
                if distance_temp__ < distance_min__
                    distance_min__ = distance_temp__;
                end
                if distance_temp__ < search_range__
                    search_range__ = distance_temp__;
                end
                x_next_index__ = x_next_index__+1;
            end
        end
        distance_min__ = sqrt(distance_min__);
    end

end

function [center_list,FC_model] = clusteringMeanShift(X,bandwidth)
% mean shift clustering algorithm
% do not need to input clustering number
% clustering method is move point to value center
%
[x_number,variable_number] = size(X);

if nargin < 2
    bandwidth = sqrt(variable_number)/4;
end

% normalize data
aver_X = mean(X);
stdD_X = std(X);
index__ = find(stdD_X == 0);
if  ~isempty(index__), stdD_X(index__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;
X_nomlz_initial = X_nomlz;

damping_max = 1;
iteration_max = 1000;
torlance = 1e-4;
identify_torlance = 1e-2;

un_converage = true(1,x_number);
done = 0;
iteration = 0;
while ~done
%     kernal_function = @(x) calKernal(x,bandwidth,x_number,variable_number,X_nomlz_initial);
%     drawFunction(kernal_function,min(X_nomlz_initial),max(X_nomlz_initial));

    damping = damping_max*(1-iteration/iteration_max);
    for x_index = 1:x_number
        if un_converage(x_index)
            x = X_nomlz(x_index,:);
            [fval,gradient] = calKernal(x,bandwidth,x_number,variable_number,X_nomlz_initial);
            X_nomlz(x_index,:) = x+gradient*damping;

            if norm(gradient,"inf") < torlance
                un_converage(x_index) = false(1);
            end
        end
    end

    if iteration >= iteration_max || ~any(un_converage)
        done = 1;
    end

    iteration = iteration+1;

%     scatter(X_nomlz(:,1),X_nomlz(:,2));
%     drawnow;
end

% origin data in x_center index
index_list = zeros(x_number,1);

% identify center of data
X_center_nomlz = X_nomlz(1,:);
index_list(1) = 1;
repeat = 1; % the number of converage point in the center point

for x_index = 2:x_number
    x = X_nomlz(x_index,:);

    out_group_flag = 1;
    for x_center_index = 1:size(X_center_nomlz,1)
        x_center = X_center_nomlz(x_center_index,:);
        if norm(x-x_center,'inf') < identify_torlance
            index_list(x_index) = x_center_index;

            % updata x_center position
            X_center_nomlz(x_center_index,:) = (x_center*repeat(x_center_index)+x)/(repeat(x_center_index)+1);

            out_group_flag = 0;
            repeat(x_center_index) = repeat(x_center_index)+1;
        end
    end

    % if donot converage to exist center point,add it into center point list
    if out_group_flag
        X_center_nomlz = [X_center_nomlz;x];
        repeat = [repeat;1];
    end
end

% normalize data
center_list = X_center_nomlz.*stdD_X+aver_X;
X = X_nomlz.*stdD_X+aver_X;

FC_model.X = X;
FC_model.X_normalize = X_nomlz;
FC_model.center_list = center_list;
FC_model.index_list = index_list;

    function [fval,gradient] = calKernal(x,bandwidth,x_number,variable_number,x_list)
        x__x_list = x-x_list;
        bandwidth_sq = bandwidth^2;
        exp_band = exp(-sum((x__x_list).^2,2)/2/bandwidth_sq);
        fval = 1/x_number*sum(exp_band);

        if nargin > 2
            gradient = -1/x_number*...
                sum(x__x_list.*exp_band/bandwidth_sq,1);
        end
    end
end

function [center_list,FC_model] = clusteringFuzzy(X,classify_number,m)
% get fuzzy cluster model
% X is x_number x variable_number matrix
% center_list is classify_number x variable_number matrix
%
iteration_max = 100;
torlance = 1e-6;

[x_number,variable_number] = size(X);

% normalization data
aver_X = mean(X);
stdD_X = std(X);
index__ = find(stdD_X == 0);
if  ~isempty(index__),stdD_X(index__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;

% if x_number equal 1,clustering cannot done
if x_number == 1
    FC_model.X = X;
    FC_model.X_normalize = X_nomlz;
    FC_model.center_list = X;
    FC_model.fval_loss_list = [];
    return;
end

U = zeros(classify_number,x_number);
center_list = rand(classify_number,variable_number)*0.5;
iteration = 0;
done = 0;
fval_loss_list = zeros(iteration_max,1);

% get X_center_dis_sq
X_center_dis_sq = zeros(classify_number,x_number);
for classify_index = 1:classify_number
    for x_index = 1:x_number
        X_center_dis_sq(classify_index,x_index) = ...
            getSq((X_nomlz(x_index,:)-center_list(classify_index,:)));
    end
end

while ~done
    % updata classify matrix U
    for classify_index = 1:classify_number
        for x_index = 1:x_number
            U(classify_index,x_index) = ...
                1/sum((X_center_dis_sq(classify_index,x_index)./X_center_dis_sq(:,x_index)).^(1/(m-1)));
        end
    end
    
    % updata center_list
    center_list_old = center_list;
    for classify_index = 1:classify_number
        center_list(classify_index,:) = ...
            sum((U(classify_index,:)').^m.*X_nomlz,1)./...
            sum((U(classify_index,:)').^m,1);
    end
    
    % updata X_center_dis_sq
    X_center_dis_sq = zeros(classify_number,x_number);
    for classify_index = 1:classify_number
        for x_index = 1:x_number
            X_center_dis_sq(classify_index,x_index) = ...
                getSq((X_nomlz(x_index,:)-center_list(classify_index,:)));
        end
    end
    
%     plot(center_list(:,1),center_list(:,2));
    
    % forced interrupt
    if iteration > iteration_max
        done = 1;
    end
    
    % convergence judgment
    if sum(sum(center_list_old-center_list).^2)<torlance
        done = 1;
    end
    
    iteration = iteration+1;
    fval_loss_list(iteration) = sum(sum(U.^m.*X_center_dis_sq));
end
fval_loss_list(iteration+1:end) = [];
center_list = center_list.*stdD_X+aver_X;

FC_model.X = X;
FC_model.X_normalize = X_nomlz;
FC_model.center_list = center_list;
FC_model.fval_loss_list = fval_loss_list;

    function sq = getSq(dx)
        % dx is 1 x variable_number matrix
        %
        sq = dx*dx';
    end
end

function con = conHole(x)
if x(1) < 0.5 && x(2) < 0.5
    con = -1;
elseif x(1) > 0.5 && x(2) > 0.5
    con = -1;
else
    con = 1;
end

end
