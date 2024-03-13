function [X_sample,dist_min_nomlz,X_total]=lhdSSLE(sample_num,vari_num,...
    low_bou,up_bou,X_exist)
% generate latin hypercube design
% S-SLE method is used
% each dimension will be pick up with first dimension as space
% find max point in each space, construct as final result
% exist point will base on low_bou and up_bou mapping to grid
%
% input:
% sample_number(new point to sample), variable_number, ...
% low_bou, up_bou, X_exist(exist point)
%
% output:
% X_sample, dist_min_nomlz(min distance of normalize data), ...
% X_total(include all point in area)
%
% reference: [1] LONG T, WU D, CHEN X, et al. A deterministic sequential
% maximin Latin hypercube design method using successive local enumeration
% for metamodel-based optimization [J]. Engineering Optimization, 2016,
% 48(6): 1019-36.
%
% Copyright 2022 03 Adel
%
if nargin < 5
    X_exist=[];
    if nargin < 4
        up_bou=[];
        if nargin < 3
            low_bou=[];
            if nargin < 2
                error('lhdSSLE: lack vari_num');
            end
        end
    end
end

if isempty(low_bou),low_bou=zeros(1,vari_num);end
if isempty(up_bou),up_bou=ones(1,vari_num);end

% check x_exist_list if meet boundary
if ~isempty(X_exist)
    if size(X_exist,2) ~= vari_num
        error('lhdSSLE: vari_num of X_exist inequal to input vari_num');
    end
    X_exist_nomlz=(X_exist-low_bou)./(up_bou-low_bou);
else
    X_exist_nomlz=[];
end

exist_num=size(X_exist,1);
total_num=sample_num+exist_num;
if sample_num < 0
    X_sample=[];
    X_total=X_exist;
    dist_min_nomlz=getMinDistance(X_exist_nomlz);
    return;
end

% initialize, feasible_list is usable grid list
% feasible_list is 1 x usable_grid_number*variable_number
% all feasible of variable sort in feasible_list
feasible_list=1:total_num;
feasible_list=repmat(feasible_list,1,vari_num);

% process x exist, x exist correspond grid exist
if ~isempty(X_exist)

    % change x_exist_list into grid
    grid_exist_list=zeros(size(X_exist_nomlz,1),vari_num);
    for x_exist_index=1:size(X_exist_nomlz,1)
        for variable_index=1:vari_num

            % mapping method
            feasible=round(X_exist_nomlz(x_exist_index,variable_index)*(total_num)+0.5);
            
            % if feasible have exist in feasible_list, choose another grid
            if sum(grid_exist_list(:,variable_index) == feasible)
                practicable=1:total_num;
                practicable(grid_exist_list(1:x_exist_index-1,variable_index))=[];
                distance=(practicable-feasible).^2;
                [~,index_min]=min(distance);
                feasible=practicable(index_min);
            end
            
            grid_exist_list(x_exist_index,variable_index)=feasible;
        end
    end
     
    % remove exist grid in feasiable range
    remove_index_list=grid_exist_list(:);
    place=repmat(0:(vari_num-1),exist_num,1);
    place=place(:)*total_num;
    feasible_list(remove_index_list+place)=[];
    
    if (length(feasible_list)/vari_num) ~= ...
            (total_num-size(X_exist_nomlz,1))
        save('matlab.mat');
        error('lhdSSLE: x_exist_list dimension is repeat');
    end
end

% choose gird by max range
grid_list=zeros(total_num,vari_num);

if ~isempty(X_exist)
   grid_list(1:size(X_exist,1),:)=grid_exist_list; 
end

for sample_new_index=exist_num+1:total_num
    % each sampling will decrease one gird of each variable
    grid_available_number=total_num-sample_new_index+1;

    % base place of each variable in feasible_list 
    place_base=0:vari_num-1;
    place_base=place_base*grid_available_number;
    
    if sample_new_index == 1
        % the first one gird can rand select
        index_list=[1,randsrc(1,vari_num-1,1:sample_num)];
        grid_list(sample_new_index,:)=index_list;
    else
        % first colume is constraint min to max, donot select
        % because multiple variable number, grid_list available have to
        % be row list which length is grid_number^(variable_number-1)
        % minimize the each grid to existing point distance
        % first find out each grid's min distance to exist point
        % then select min distance grid form list
        % index_list is the select grid index in feasible_list
        if vari_num == 1
            index_list=1;
            grid_list(sample_new_index,:)=feasible_list(index_list);
        else
            % get all grid list
            all_grid=getAllGrid...
                (feasible_list,grid_available_number,vari_num);
            
            % calculate all grid list to exist point distance
            grid_exit_dis=zeros(grid_available_number^(vari_num-1),sample_new_index-1);
            for x_exist_index=1:sample_new_index-1
                grid_exit_dis(:,x_exist_index)=sum((grid_list(x_exist_index,:)-all_grid).^2,2);
            end
            
            [~,grid_index]=max(min(grid_exit_dis,[],2));
            grid_index=grid_index(1);
            grid_list(sample_new_index,:)=all_grid(grid_index,:);
            
            % transform grid index to index_list
            index_list=ones(1,vari_num);
            grid_index=grid_index-1;
            bit_index=vari_num;
            while(grid_index > 0)
                index_list(bit_index)=mod(grid_index,grid_available_number)+1;
                grid_index=floor(grid_index/grid_available_number);
                bit_index=bit_index-1;
            end
            
        end
    end
    
    % remove index_list from feasiale list
    feasible_list(index_list+place_base)=[];
end

% remapping method
X_sample_nomlz=(grid_list(exist_num+1:end,:)-0.5)/(total_num);
dist_min_nomlz=getMinDistance([X_exist_nomlz;X_sample_nomlz]);

% normailze
X_sample=X_sample_nomlz.*(up_bou-low_bou)+low_bou;
X_total=[X_exist;X_sample];

    function all_grid=getAllGrid...
            (feasible_list,grid_available_number,variable_number)
        % get all grid list
        %
        all_grid=zeros(grid_available_number^(variable_number-1),variable_number);
        for variable_index__=2:variable_number
            g=ones(grid_available_number^(variable_number-variable_index__),1)*...
                feasible_list(1,((variable_index__-1)*grid_available_number+1):(variable_index__*grid_available_number));
            g=g(:);
            t=g*ones(1,grid_available_number^(variable_index__-2));
            all_grid(:,variable_index__)=t(:);
        end
        all_grid(:,1)=feasible_list(1);
    end

    function distance_min__=getMinDistance(x_list__)
        % get distance min from x_list
        %
        
        % sort x_supply_list_initial to decrese distance calculate times
        x_list__=sortrows(x_list__,1);
        sample_number__=size(x_list__,1);
        variable_number__=size(x_list__,2);
        distance_min__=variable_number__;
        for x_index__=1:sample_number__
            x_curr__=x_list__(x_index__,:);
            x_next_index__=x_index__ + 1;
            % first dimension only search in min_distance
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

end