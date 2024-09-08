function ND_idx_list=getNoDominate(data_list)
% distinguish no dominate front of data list
% data_list is x_number x data_number matrix
% notice if all data of x1 is less than x2,x1 domain x2
%
% notice: there are any way to find pareto base on followed
% rule: point of no dominate will no be dominate by all point
%
x_number=size(data_list,1);
ND_idx_list=[]; % sort all idx of filter point list

% select no domain filter
for x_idx=1:x_number
    data=data_list(x_idx,:);
    pareto_idx=1;
    add_filter_flag=1;
    while pareto_idx <= length(ND_idx_list)
        % compare x with exit pareto front point
        x_pareto_idx=ND_idx_list(pareto_idx,:);

        % contain constraint of x_filter
        data_pareto=data_list(x_pareto_idx,:);

        % compare x with x_pareto
        judge=data > data_pareto;
        if all(judge)
            add_filter_flag=0;
            break;
        end

        % if better than exit pareto point,reject pareto point
        judge=data < data_pareto;
        if all(judge)
            ND_idx_list(pareto_idx)=[];
            pareto_idx=pareto_idx-1;
        end

        pareto_idx=pareto_idx+1;
    end

    % add into pareto list if possible
    if add_filter_flag
        ND_idx_list=[ND_idx_list;x_idx];
    end
end
end
