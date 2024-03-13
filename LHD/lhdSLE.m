function [X_sample,dist_min_nomlz]=lhdSLE(sample_num,vari_num,...
    low_bou,up_bou)
% generate latin hypercube design
% successive local enumeration(SLE) method
% each dimension will be pick up with first dimension as space
% find max point in each space, construct as final result
%
% input:
% sample number(new point to sample),variable_number
%
% output:
% X_sample,dist_min_nomlz(min distance of normalize data)
%
% reference: [1] ZHU H, LIU L, LONG T, et al. A novel algorithm of maximin
% Latin hypercube design using successive local enumeration [J].
% Engineering Optimization, 2012, 44(5): 551-64.
%
% Copyright 2022 Adel
%
if nargin < 4
    up_bou=[];
    if nargin < 3
        low_bou=[];
        if nargin < 2
            error('lhdSLE: lack vari_num');
        end
    end
end

if isempty(low_bou),low_bou=zeros(1,vari_num);end
if isempty(up_bou),up_bou=ones(1,vari_num);end

if sample_num < 0
    X_sample_nomlz=[];
    dist_min_nomlz=getMinDistance(X_sample_nomlz);
    return;
end

% initialize, feasible_list is usable grid list
% feasible_list is 1 x usable_grid_number*variable_number
% all feasible of variable sort in feasible_list
feasible_list=1:sample_num;
feasible_list=repmat(feasible_list,1,vari_num);

% choose gird by max range
grid_list=zeros(sample_num,vari_num);
for sample_new_index=1:sample_num
    % each sampling will decrease one gird of each variable
    grid_available_number=sample_num-sample_new_index+1;

    % base place of each variable in feasible_list 
    place_base=0:(vari_num-1);
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
X_sample_nomlz=(grid_list-0.5)/(sample_num);

dist_min_nomlz=getMinDistance(X_sample_nomlz);

% normailze
X_sample=X_sample_nomlz.*(up_bou-low_bou)+low_bou;

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
        % all x will be calculate
        %
        if isempty(x_list__)
            distance_min__=[];
            return;
        end

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