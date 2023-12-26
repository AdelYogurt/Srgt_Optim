function [X_sample,dist_min_nomlz,X_total,Idx_select]=lhdNSLE(X_base,sample_num,vari_num,...
    low_bou,up_bou,X_exist)
% generate nested latin hypercube design
% SLE method is used(sample and iteration, select max min distance group)
% election combination mode of point and find best combination
%
% input:
% X_base(which will be sample), sample number(new point to sample), ...
% variable_number, low_bou, up_bou, X_exist(exist point)
%
% output:
% X_sample, dist_min_nomlz(min distance of normalize data)
% X_total(include all data in area)
%
if nargin < 6
    X_exist=[];
    if nargin < 5
        up_bou=[];
        if nargin < 4
            low_bou=[];
            if nargin < 3
                error('lhdNSLE: lack input');
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
        error('lhdNSLE: vari_num of X_exist inequal to input vari_num');
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
X_base_nomlz=(X_base-low_bou)./(up_bou-low_bou);

% iterate and get final x_supply_list
iter=0;
x_supply_quasi_number=size(X_base_nomlz,1);
dist_min_nomlz=0;
X_sample_nomlz=[];

% dist_min_nomlz_result=zeros(1,iteration);
Idx_select=[];
while iter <= iter_max
    % random select x_new_number X to X_trial_nomlz
    X_index=randperm(x_supply_quasi_number,sample_num);
    
    % get distance min itertion X_
    distance_min_iteration=getMinDistanceIter...
        (X_base_nomlz(X_index,:),X_exist_nomlz);
    
    % if distance_min_iteration is large than last time
    if distance_min_iteration > dist_min_nomlz
        dist_min_nomlz=distance_min_iteration;
        X_sample_nomlz=X_base_nomlz(X_index,:);
        Idx_select=X_index;
    end
    
    iter=iter+1;
%     dist_min_nomlz_result(iteration)=dist_min_nomlz;
end
dist_min_nomlz=getMinDistance([X_sample_nomlz;X_exist_nomlz]);
X_sample=X_sample_nomlz.*(up_bou-low_bou)+low_bou;
X_total=[X_exist;X_sample];

    function distance_min__=getMinDistance(x_list__)
        % get distance min from x_list
        %
        if isempty(x_list__)
            distance_min__=[];
            return;
        end

        % sort x_supply_list_initial to decrese distance calculate times
        x_list__=sortrows(x_list__,1);
        [sample_number__,variable_number__]=size(x_list__);
        distance_min__=variable_number__;
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
                if distance_temp__ < distance_min__
                    distance_min__=distance_temp__;
                end
                if distance_temp__ < search_range__
                    search_range__=distance_temp__;
                end
                x_next_index__=x_next_index__+1;
            end
        end
        distance_min__=sqrt(distance_min__);
    end
    function distance_min__=getMinDistanceIter...
            (x_list__,x_exist_list__)
        % get distance min from x_list
        % this version do not consider distance between x exist
        %

        % sort x_supply_list_initial to decrese distance calculate times
        x_list__=sortrows(x_list__,1);
        [sample_number__,variable_number__]=size(x_list__);
        distance_min__=variable_number__;
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
                if distance_temp__ < distance_min__
                    distance_min__=distance_temp__;
                end
                if distance_temp__ < search_range__
                    search_range__=distance_temp__;
                end
                x_next_index__=x_next_index__+1;
            end
            for x_exist_index=1:size(x_exist_list__,1)
                x_next__=x_exist_list__(x_exist_index,:);
                distance_temp__=sum((x_next__-x_curr__).^2);
                if distance_temp__ < distance_min__
                    distance_min__=distance_temp__;
                end
            end
        end
        distance_min__=sqrt(distance_min__);
    end
end
