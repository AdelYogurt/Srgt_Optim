function [center_list,FC_model] = clusteringFuzzyFeatureSpace...
    (X,classify_number,m,kernel_function)
% get fuzzy cluster model with feature space
% kernal function recommend kernal_function = @(sq) exp(-sq/2*1000);
% X is x_number x variable_number matrix
% center_list is classify_number x variable_number matrix
%
if nargin < 4
    kernel_function = [];
end

iteration_max = 100;
torlance = 1e-6;

[x_number,variable_number] = size(X);

% normaliz data
aver_X = mean(X);
stdD_X = std(X);
index__ = find(stdD_X == 0);
if  ~isempty(index__),stdD_X(index__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;

% default kernal function
if isempty(kernel_function)
    sigma = 1/variable_number;
    kernel_function = @(dis_sq) exp(-dis_sq*sigma);
end

% if x_number equal 1, clustering cannot done
if x_number == 1
    center_list = X;
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
                1/sum(((2-2*kernel_function(X_center_dis_sq(classify_index,x_index)))./...
                (2-2*kernel_function(X_center_dis_sq(:,x_index)))).^(1/(m-1)));
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

%     scatter(X_nomlz(:,1),X_nomlz(:,2));
%     line(center_list(:,1),center_list(:,2),'Marker','o','LineStyle','None','Color','r');
    
    % forced interrupt
    if iteration > iteration_max
        done = 1;
    end
    
    % convergence judgment
    if sum(sum(center_list_old-center_list).^2) < torlance
        done = 1;
    end
    
    iteration = iteration+1;
    fval_loss_list(iteration) = sum(sum(U.^m.*(2-2*kernel_function(X_center_dis_sq))));
end
fval_loss_list(iteration+1:end) = [];

% normalize
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
