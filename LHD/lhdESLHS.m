function [X_sample,dist_min_nomlz,X_total,dist_min_nomlz_result]=lhdESLHS(sample_num,vari_num,...
    low_bou,up_bou,X_exist,con_fcn_cheap)
% generate latin hypercube design
% ESLHS method is used(sample and iteration)
% election combination mode of point and find best combination
%
% input:
% sample_num(new point to sample), vari_num, ...
% low_bou, up_bou, X_exist(exist point in sample area), con_fcn_cheap
%
% output:
% X_sample, dist_min_nomlz(min distance of normalize data), ...
% X_total(include all data in area)
%
% reference: [1] LONG T, LI X, SHI R, et al., Gradient-Free
% Trust-Region-Based Adaptive Response Surface Method for Expensive
% Aircraft Optimization[J]. AIAA Journal, 2018, 56(2): 862-73.
%
% Copyright 2023 03 Adel
%
if nargin < 6
    con_fcn_cheap=[];
    if nargin < 5
        X_exist=[];
        if nargin < 4
            up_bou=[];
            if nargin < 3
                low_bou=[];
                if nargin < 2
                    error('lhdESLHS: lack vari_num');
                end
            end
        end
    end
end

if isempty(low_bou),low_bou=zeros(1,vari_num);end
if isempty(up_bou),up_bou=ones(1,vari_num);end

iter_max=min(10*sample_num,100);

% check x_exist_list if meet boundary
if ~isempty(X_exist)
    if size(X_exist,2) ~= vari_num
        error('lhdESLHS: vari_num of X_exist inequal to input vari_num');
    end
    X_exist_nomlz=(X_exist-low_bou)./(up_bou-low_bou);
else
    X_exist_nomlz=[];
end

exist_num=size(X_exist,1);
total_num=sample_num+exist_num;
if sample_num <= 0
    X_total=X_exist;
    X_sample=[];
    dist_min_nomlz=getMinDistance(X_exist_nomlz);
    return;
end

% get quasi-feasible point
x_initial_number=10*sample_num;
x_quasi_number=3*sample_num;
if ~isempty(con_fcn_cheap)
    X_supply_quasi_nomlz=[];

    % check if have enough X_supply_nomlz
    iter=0;
    while size(X_supply_quasi_nomlz,1) < x_quasi_number && iter < 100
        X_supply_initial_nomlz=rand(x_initial_number,vari_num);

        qusai_index=[];
        for x_index=1:size(X_supply_initial_nomlz,1)
            if con_fcn_cheap(X_supply_initial_nomlz(x_index,:).*(up_bou-low_bou)+low_bou) <= 0
                qusai_index=[qusai_index,x_index];
            end
        end
        X_supply_quasi_nomlz=[X_supply_quasi_nomlz;X_supply_initial_nomlz(qusai_index,:)];

        iter=iter+1;
    end

    if iter == 100 && isempty(X_supply_quasi_nomlz)
        error('lhdESLHS: feasible quasi point cannot be found');
    end
else
    X_supply_quasi_nomlz=rand(x_quasi_number,vari_num);
end

% iterate and get final x_supply_list
iter=0;
x_supply_quasi_num=size(X_supply_quasi_nomlz,1);
dist_min_nomlz=0;
X_sample_nomlz=[];

dist_min_nomlz_result=zeros(1,iter_max);
while iter <= iter_max
    % random select x_new_number X to X_trial_nomlz
    x_select_index=randperm(x_supply_quasi_num,sample_num);
    
    % get distance min itertion X_
    distance_min_iteration=getMinDistanceIter...
        (X_supply_quasi_nomlz(x_select_index,:),X_exist_nomlz);
    
    % if distance_min_iteration is large than last time
    if distance_min_iteration > dist_min_nomlz
        dist_min_nomlz=distance_min_iteration;
        X_sample_nomlz=X_supply_quasi_nomlz(x_select_index,:);
    end
    
    iter=iter+1;
    dist_min_nomlz_result(iter)=dist_min_nomlz;
end
dist_min_nomlz=getMinDistance([X_sample_nomlz;X_exist_nomlz]);
X_sample=X_sample_nomlz.*(up_bou-low_bou)+low_bou;
X_total=[X_exist;X_sample];

    function distance_min=getMinDistance(x_list__)
        % get distance min from x_list
        %
        if isempty(x_list__)
            distance_min=[];
            return;
        end

        % sort x_supply_list_initial to decrese distance calculate times
        x_list__=sortrows(x_list__,1);
        [sample_number__,variable_number__]=size(x_list__);
        distance_min=variable_number__;
        for x_index__=1:sample_number__
            x_curr__=x_list__(x_index__,:);
            x_next_index__=x_index__ + 1;
            % only search in min_distance(x_list had been sort)
            search_range__=variable_number__;
            while x_next_index__ <= sample_number__ &&...
                    (x_list__(x_next_index__,1)-x_list__(x_index__,1))^2 ...
                    < search_range__
                x_next__=x_list__(x_next_index__,:);
                distance_temp__=sum((x_next__-x_curr__).^2);
                if distance_temp__ < distance_min
                    distance_min=distance_temp__;
                end
                if distance_temp__ < search_range__
                    search_range__=distance_temp__;
                end
                x_next_index__=x_next_index__+1;
            end
        end
        distance_min=sqrt(distance_min);
    end
    function distance_min=getMinDistanceIter...
            (x_list__,x_exist_list__)
        % get distance min from x_list
        % this version do not consider distance between x exist
        %

        % sort x_supply_list_initial to decrese distance calculate times
        x_list__=sortrows(x_list__,1);
        [sample_number__,variable_number__]=size(x_list__);
        distance_min=variable_number__;
        for x_index__=1:sample_number__
            x_curr__=x_list__(x_index__,:);
            x_next_index__=x_index__ + 1;
            % only search in min_distance(x_list had been sort)
            search_range__=variable_number__;
            while x_next_index__ <= sample_number__ &&...
                    (x_list__(x_next_index__,1)-x_list__(x_index__,1))^2 ...
                    < search_range__
                x_next__=x_list__(x_next_index__,:);
                distance_temp__=sum((x_next__-x_curr__).^2);
                if distance_temp__ < distance_min
                    distance_min=distance_temp__;
                end
                if distance_temp__ < search_range__
                    search_range__=distance_temp__;
                end
                x_next_index__=x_next_index__+1;
            end
            for x_exist_index=1:size(x_exist_list__,1)
                x_next__=x_exist_list__(x_exist_index,:);
                distance_temp__=sum((x_next__-x_curr__).^2);
                if distance_temp__ < distance_min
                    distance_min=distance_temp__;
                end
            end
        end
        distance_min=sqrt(distance_min);
    end
end
