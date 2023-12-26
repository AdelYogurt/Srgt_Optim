clc;
clear;
close all hidden;

X = [rand(10,2)*0.5;rand(10,2)*0.5+0.5;(rand(10,2)*0.5+[0,0.5])];

[center_list,MS_model] = clusteringMeanShift(X);
clusteringVisualization(MS_model);

function [center_list,MS_model] = clusteringMeanShift(X,bandwidth)
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
X_move = X_nomlz.*stdD_X+aver_X;

MS_model.X = X;
MS_model.X_move = X_move;
MS_model.X_normalize = X_nomlz_initial;
MS_model.center_list = center_list;
MS_model.index_list = index_list;

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
