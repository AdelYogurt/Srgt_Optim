clc;
clear;
close all hidden;

data_library_name = 'optimal_data_library';

benchmark = BenchmarkFunction();

% variable_number = 2;
% object_function = @(x) benchmark.singleGPObject(x);
% A = [];
% B = [];
% Aeq = [];
% Beq = [];
% low_bou = [-2, -2];
% up_bou = [2, 2];
% nonlcon_function = [];
% cheapcon_function = [];

variable_number = 2;
object_function = @(x) benchmark.singlePKObject(x);
A = [];
B = [];
Aeq = [];
Beq = [];
low_bou = [-3, -3];
up_bou = [3, 3];
nonlcon_function = [];
cheapcon_function = [];

% variable_number = 4;
% object_function = @(x) benchmark.singleROSObject(x);
% object_function_LF = @(x) benchmark.singleROSObjectLow(x);
% A = [];
% B = [];
% Aeq = [];
% Beq = [];
% low_bou = [-2, -2, -2, -2];
% up_bou = [2, 2, 2, 2];
% nonlcon_function = [];
% nonlcon_function_LF = [];
% cheapcon_function = [];

% variable_number = 6;
% object_function = @(x) benchmark.singleHNObject(x);
% A = [];
% B = [];
% Aeq = [];
% Beq = [];
% low_bou = zeros(1, variable_number);
% up_bou = ones(1, variable_number);
% nonlcon_function = [];
% cheapcon_function = [];
% model_function = [];

% variable_number = 20;
% object_function = @(x) benchmark.singleDP20Object(x);
% object_function_LF = @(x) benchmark.singleDP20ObjectLow(x);
% A = [];
% B = [];
% Aeq = [];
% Beq = [];
% low_bou = ones(1, variable_number)*-30;
% up_bou = ones(1, variable_number)*30;
% nonlcon_function = [];
% nonlcon_function_LF = [];
% cheapcon_function = [];

% variable_number = 20;
% object_function = @(x) benchmark.singleEP20Object(x);
% object_function_LF = @(x) benchmark.singleEP20ObjectLow(x);
% A = [];
% B = [];
% Aeq = [];
% Beq = [];
% low_bou = ones(1, variable_number)*-30;
% up_bou = ones(1, variable_number)*30;
% nonlcon_function = [];
% nonlcon_function_LF = [];
% cheapcon_function = [];

% variable_number = 2;
% object_function = @(x) benchmark.singleG06Object(x);
% object_function_LF = @(x) benchmark.singleG06ObjectLow(x);
% A = [];
% B = [];
% Aeq = [];
% Beq = [];
% low_bou = [13, 0];
% up_bou = [100, 100];
% nonlcon_function = @(x) benchmark.singleG06Nonlcon(x);
% nonlcon_function_LF = @(x) benchmark.singleG06NonlconLow(x);
% cheapcon_function = [];
% model_function = [];

% variable_number = 4;
% object_function = @(x) benchmark.singlePVD4Object(x);
% object_function_low = @(x) benchmark.singlePVD4ObjecttLow(x);
% A = [-1, 0, 0.0193, 0;
%     0, -1, 0.00954, 0;];
% B = [0;0];
% Aeq = [];
% Beq = [];
% low_bou = [0, 0, 0, 0];
% up_bou = [1, 1, 50, 240];
% nonlcon_function = @(x) cheapconFunction(x, A, B, Aeq, Beq, @(x) benchmark.singlePVD4Nonlcon(x));
% cheapcon_function = [];
% model_function = [];

% variable_number = 13;
% object_function = @benchmark.singleG01Object;
% object_function_low = @benchmark.singleG01ObjectLow;
% A = [ 2   2   0   0   0   0   0   0   0   1   1   0   0;
%     2   0   2   0   0   0   0   0   0   1   0   1   0;
%     0   2   2   0   0   0   0   0   0   0   1   1   0;
%     -8  0   0   0   0   0   0   0   0   1   0   0   0;
%     0   -8  0   0   0   0   0   0   0   0   1   0   0;
%     0   0   0   -2  -1  0   0   0   0   1   0   0   0;
%     0   0   0   0   0   -2  -1  0   0   0   1   0   0;
%     0   0   0   0   0   0   0   -2  -1  0   0   1   0;
%     ];
% B = [10;10;10;0;0;0;0;0];
% Aeq = [];
% Beq = [];
% low_bou = zeros(1, 13);
% up_bou = ones(1, 13);
% up_bou(10:12) = 100;
% nonlcon_function = @(x) cheapconFunction(x, A, B, Aeq, Beq, []);
% nonlcon_function_LF = @(x) cheapconFunction(x, A, B, Aeq, Beq, []);
% cheapcon_function = [];

% x_initial = rand(1, variable_number).*(up_bou-low_bou)+low_bou;
% fmincon_option = optimoptions('fmincon', 'Algorithm', 'sqp');
% [x_best, fval_best, ~, output, lambda, grad, hessian] = fmincon(object_function, x_initial, A, B, Aeq, Beq, low_bou, up_bou, nonlcon_function, fmincon_option)

%% single run

% delete([data_library_name, '.txt']);
% delete('result_total.txt');
% 
% [x_best, fval_best, NFE, output] = optimalSurrogateKGPC...
%     (object_function, variable_number, low_bou, up_bou, nonlcon_function, ...
%     cheapcon_function, [], 200)
% 
% result_x_best = output.result_x_best;
% result_fval_best = output.result_fval_best;
% 
% figure(1);
% plot(result_fval_best);
% 
% figure(2);
% [x_list, fval_list, con_list, coneq_list] = dataLibraryLoad...
%     (data_library_name, low_bou, up_bou);
% scatter3(x_list(:, 1), x_list(:, 2), fval_list);
% xlabel('X');
% ylabel('Y');
% zlabel('Z');

%% repeat run

repeat_number = 10;
result_fval = zeros(repeat_number, 1);
Max_NFE = 40;
for repeat_index = 1:repeat_number
    delete([data_library_name, '.txt']);
    delete('result_total.txt');

    [x_best, fval_best, NFE, output] = optimalSurrogateKGPC...
        (object_function, variable_number, low_bou, up_bou, nonlcon_function, ...
        cheapcon_function, [], Max_NFE);

    result_fval(repeat_index) = fval_best;
end

fprintf('Fval     : lowest = %4.4f, mean = %4.4f, worst = %4.4f, std = %4.4f \n', min(result_fval), mean(result_fval), max(result_fval), std(result_fval));
object_function_name=char(object_function);
save([object_function_name(15:end-3),'_',num2str(Max_NFE),'_K_GPC','.mat']);

%% main
function [x_best, fval_best, NFE, output] = optimalSurrogateKGPC...
    (object_function, variable_number, low_bou, up_bou, nonlcon_function, ...
    cheapcon_function, model_function, ....
    NFE_max, iteration_max, torlance, nonlcon_torlance)
% surrogate base optimal use radias base function method version 0
% use SVM to get interest point
% FS_FCM to get interest point center point
% and updata interest space
% all function exchange x should be colume vector
% x_list is x_number x variable_number matrix
% both nonlcon_function and cheapcon_function format is [con, coneq]
% model_function should output fval, format is [fval, con, coneq]
% con or coneq can be colume vector if there was more than one constrain
%
% referance: [1] SHI R, LIU L, LONG T, et al. Sequential Radial Basis
% Function Using Support Vector Machine for Expensive Design Optimization
% [J]. AIAA Journal, 2017, 55(1): 214-27.
%
% Copyright Adel 2022.10
%
if nargin < 11 || isempty(nonlcon_torlance)
    nonlcon_torlance = 1e-3;
    if nargin < 10 || isempty(torlance)
        torlance = 1e-3;
        if nargin < 9
            iteration_max = [];
            if nargin < 8
                NFE_max = [];
            end
        end
    end
end

if nargin < 7
    model_function = [];
    if nargin < 6
        cheapcon_function = [];
        if nargin < 5
            nonlcon_function = [];
        end
    end
end

DRAW_FIGURE_FLAG = 0; % whether draw data
INFORMATION_FLAG = 1; % whether print data
CONVERGENCE_JUDGMENT_FLAG = 0; % whether judgment convergence

if isempty(iteration_max)
    iteration_max = 100;
end

% hyper parameter
sample_number_initial = 5*variable_number;
if variable_number <= 5
    sample_number_iteration = 2;
else
    sample_number_iteration = 3;
end
sample_number_data = 10*sample_number_initial;
RBF_number = min(100, (variable_number + 1)*(variable_number + 2)/2);

% max fval when normalize fval, con, coneq
nomlz_fval = 10;

filter_torlance = 1e-3;
protect_range = 1e-4;

data_library_name = 'optimal_data_library';
file_result = fopen('result_total.txt', 'a');
fprintf(file_result, '%s\n', datetime);
fclose(file_result);
clear('file_result');

done = 0;NFE = 0;iteration = 0;

result_x_best = zeros(iteration_max, variable_number);
result_fval_best = zeros(iteration_max, 1);
x_data_list = lhsdesign(sample_number_data, variable_number, 'iterations', 50).*...
    (up_bou-low_bou)+low_bou;

% if do not input model_function, generate model_function
if isempty(model_function)
    model_function = @(x) modelFunction(x, object_function, nonlcon_function);
end

iteration = iteration+1;

% step 2
% use latin hypercube method to get initial sample x_list
% [~, x_updata_list, ~] = getLatinHypercube...
%     (sample_number_initial, variable_number, [], low_bou, up_bou, cheapcon_function);
x_updata_list = lhsdesign(sample_number_initial, variable_number, 'iterations', 50, 'criterion', 'maximin').*(up_bou-low_bou)+low_bou;

% detech expensive constraints
if ~isempty(x_updata_list)
    [~, con, coneq] = dataLibraryUpdata...
        (data_library_name, model_function, x_updata_list(1, :));NFE = NFE+1;
    x_updata_list = x_updata_list(2:end, :);
else
    [~, con, coneq] = dataLibraryLoad(data_library_name, low_bou, up_bou);
end
if ~isempty(con) || ~isempty(coneq)
    expensive_nonlcon_flag = 1;
else
    expensive_nonlcon_flag = 0;
end

% NFE and iteration setting
if isempty(NFE_max)
    if expensive_nonlcon_flag
        NFE_max = 50*variable_number;
    else
        NFE_max = 20*variable_number;
    end
end

if isempty(iteration_max)
    if expensive_nonlcon_flag
        iteration_max = 50*variable_number;
    else
        iteration_max = 20*variable_number;
    end
end

% import data from data library
[x_list, fval_list, con_list, coneq_list] = dataLibraryLoad...
    (data_library_name, low_bou, up_bou);

kriging_model_fval = [];
kriging_model_con = [];
kriging_model_coneq = [];
GPC_hyp.mean = 0;
GPC_hyp.cov = zeros(1, variable_number+1);
while ~done
    % step 3
    % updata data library by x_list
    [fval_updata_list, con_updata_list, coneq_updata_list] = dataLibraryUpdata...
        (data_library_name, model_function, x_updata_list);NFE = NFE+size(x_updata_list, 1);
    x_list = [x_list;x_updata_list];
    fval_list = [fval_list;fval_updata_list];
    if ~isempty(con_updata_list)
        con_list = [con_list;con_updata_list];
    end
    if ~isempty(coneq_updata_list)
        coneq_list = [coneq_list;coneq_updata_list];
    end

    % nomalization all data
    fval_max = max(abs(fval_list), [], 1);
    fval_nomlz_list = fval_list./fval_max*nomlz_fval;
%     fval_nomlz_list = fval_list;
    if ~isempty(con_list)
        con_max_list = max(abs(con_list), [], 1);
        con_nomlz_list = con_list./con_max_list*nomlz_fval;
    else
        con_max_list = [];
        con_nomlz_list = [];
    end
    if ~isempty(coneq_list)
        coneq_max_list = max(abs(coneq_list), [], 1);
        coneq_nomlz_list = coneq_list./coneq_max_list*nomlz_fval;
    else
        coneq_max_list = [];
        coneq_nomlz_list = [];
    end

    % step 4
    % select nearest point to construct RBF
    %     [~, index] = min(fval_nomlz_list);
    index = randi(size(x_list,1));
    x_initial = x_list(index, :);
    RBF_model_number = min(RBF_number, size(x_list, 1));

    distance = sum(((x_initial-x_list)./(up_bou-low_bou)).^2, 2);
    [~, dis_index_list] = sort(distance);
    index_list = [dis_index_list(1:RBF_model_number)];
    x_list_model = x_list(index_list, :);
    fval_nomlz_list_model = fval_nomlz_list(index_list, :);
    if ~isempty(con_list)
        con_nomlz_list_model = con_nomlz_list(index_list, :);
    else
        con_nomlz_list_model = [];
    end
    if ~isempty(coneq_list)
        coneq_nomlz_list_model = coneq_nomlz_list(index_list, :);
    else
        coneq_nomlz_list_model = [];
    end

    % get RBF model and function
    [RBF_model_fval, RBF_model_con, RBF_model_coneq, output_RBF] = getRadialBasisModel...
        (x_list_model, fval_nomlz_list_model, con_nomlz_list_model, coneq_nomlz_list_model);
    object_function_surrogate = output_RBF.object_function_surrogate;
    nonlcon_function_surrogate = output_RBF.nonlcon_function_surrogate;

    % if surrogate not convex
    if object_function_surrogate(sum(x_list_model,1)/RBF_model_number)...
            > sum(fval_nomlz_list_model)/RBF_model_number
        low_bou_RBF = min(x_list_model, [], 1);
        up_bou_RBF = max(x_list_model, [], 1);
    else
        low_bou_RBF=low_bou;
        up_bou_RBF=up_bou;
    end

    % search local
    [x_potential,fval_potential_predict]=fmincon(object_function_surrogate,x_initial,[],[],[],[],low_bou_RBF,up_bou_RBF,nonlcon_function_surrogate,...
        optimoptions('fmincon','Display','none','Algorithm','sqp'));

    %     % step 5
    %     % MSP guideline to obtain x_adapt
    %     [x_potential, ~, exitflag, ~] = findMinMSP...
    %         (object_function_surrogate, variable_number, low_bou, up_bou, nonlcon_function_surrogate, ...
    %         cheapcon_function, nonlcon_torlance);
    %
    %     if exitflag  ==  -2
    %         % optimal feasiblilty if do not exist feasible point
    %         object_nonlcon_function_surrogate = @(x) objectNonlconFunctionSurrogate(x, nonlcon_function_surrogate);
    %         [x_potential, ~, exitflag, ~] = findMinMSP...
    %             (object_nonlcon_function_surrogate, variable_number, low_bou, up_bou, [], ...
    %             cheapcon_function, nonlcon_torlance);
    %     end

    % check x_potential if exist in data library
    % if not, updata data libraray
    [x_potential, fval_potential, con_potential, coneq_potential, NFE_p, repeat_index] = dataLibraryUpdataProtect...
        (data_library_name, model_function, x_potential, x_list, low_bou, up_bou, protect_range);NFE = NFE+NFE_p;
    x_list = [x_list;x_potential];
    fval_list = [fval_list;fval_potential];
    if ~isempty(con_list)
        con_list = [con_list;con_potential];
    end
    if ~isempty(coneq_list)
        coneq_list = [coneq_list;coneq_potential];
    end

    % normalization fval updata
    if ~isempty(fval_potential)
        fval_potential_nomlz = fval_potential/fval_max*nomlz_fval;
        fval_nomlz_list = [fval_nomlz_list;fval_potential_nomlz];
    end
    if ~isempty(con_potential)
        con_potential_nomlz = (con_potential./con_max_list)*nomlz_fval;
        con_nomlz_list = [con_nomlz_list;con_potential_nomlz];
    end
    if ~isempty(coneq_potential)
        coneq_potential_nomlz = (coneq_potential./coneq_max_list)*nomlz_fval;
        coneq_nomlz_list = [coneq_nomlz_list;coneq_potential_nomlz];
    end

    % when x_potential is exist in data library, x_potential_add will be
    % empty, this times we will use origin point data
    if isempty(x_potential)
        x_potential = x_list(repeat_index, :);
        fval_potential = fval_list(repeat_index, :);
        if ~isempty(con_list)
            con_potential = con_list(repeat_index, :);
        else
            con_potential = [];
        end
        if ~isempty(coneq_list)
            coneq_potential = coneq_list(repeat_index, :);
        else
            coneq_potential = [];
        end
    end

    if DRAW_FIGURE_FLAG && variable_number < 3
        interpVisualize(RBF_model_fval, low_bou, up_bou);
        line(x_potential(1), x_potential(2), fval_potential/fval_max*nomlz_fval, 'Marker', 'o', 'color', 'r', 'LineStyle', 'none')
    end

    % step 6
    % find best result to record
    [x_best, fval_best, con_best, coneq_best] = findMinRaw...
        (x_list, fval_list, con_list, coneq_list, ...
        cheapcon_function, nonlcon_torlance);

    if INFORMATION_FLAG
        fprintf('iteration:          %-3d    NFE:    %-3d\n', iteration, NFE);
        fprintf('current x:          %s\n', num2str(x_potential));
        fprintf('current value:      %f\n', fval_potential);
        fprintf('current violation:  %s  %s\n', num2str(con_potential), num2str(coneq_potential));
        fprintf('\n');
    end

    result_x_best(iteration, :) = x_best;
    result_fval_best(iteration, :) = fval_best;
    iteration = iteration+1;

    % forced interrupt
    if iteration > iteration_max || NFE >= NFE_max
        done = 1;
    end

    % convergence judgment
    if CONVERGENCE_JUDGMENT_FLAG
        if (iteration > 2 && ...
                abs((fval_potential-fval_potential_old)/fval_potential_old) < torlance)
            done = 1;
            if ~isempty(con_best)
                if sum(con_best > nonlcon_torlance)
                    done = 0;
                end
            end
            if ~isempty(coneq_best)
                if sum(abs(coneq_best) > nonlcon_torlance)
                    done = 0;
                end
            end
        end
    end

    % Interest space sampling
    if ~done
        [quantile, normal_index, small_index, large_index] = findUnusual(fval_nomlz_list);
        %     model_index = [normal_index, small_index];
        %     x_list_model = x_list(model_index, :);
        %     fval_nomlz_list_model = fval_nomlz_list(model_index, :);
        %     if ~isempty(con_list)
        %         con_nomlz_list_model = con_nomlz_list(model_index, :);
        %     else
        %         con_nomlz_list_model = [];
        %     end
        %     if ~isempty(coneq_list)
        %         coneq_nomlz_list_model = coneq_nomlz_list(model_index, :);
        %     else
        %         coneq_nomlz_list_model = [];
        %     end
        x_list_model = x_list;
        fval_nomlz_list_model = fval_nomlz_list; fval_nomlz_list_model(large_index) = quantile(end);
        if ~isempty(con_list)
            con_nomlz_list_model = con_nomlz_list(model_index, :);
        else
            con_nomlz_list_model = [];
        end
        if ~isempty(coneq_list)
            coneq_nomlz_list_model = coneq_nomlz_list(model_index, :);
        else
            coneq_nomlz_list_model = [];
        end

        % get kriging model and function
        [kriging_model_fval, ~, ~, output_kriging] = getKrigingModel...
            (x_list_model, fval_nomlz_list_model, [], [], ...
            kriging_model_fval);
        object_function_surrogate = output_kriging.object_function_surrogate;

        % step 7
        % using SVM to identify area which is interesting
        if expensive_nonlcon_flag
            % because data prefer getting better
            filter_index_list = [];% filter point list
            feasible_index_list = [];% feasible point list

            % generate filter
            for x_index = 1:size(x_list, 1)
                con_x_nomlz = [];
                if ~isempty(con_nomlz_list)
                    con_x_nomlz = max(con_nomlz_list(x_index, :));
                end
                coneq_x_nomlz = [];
                if ~isempty(coneq_nomlz_list)
                    coneq_x_nomlz = max(abs(coneq_nomlz_list(x_index, :)));
                end
                total_con_x_nomlz = max([con_x_nomlz;coneq_x_nomlz]);

                % only with constraint will add into filter
                if (total_con_x_nomlz > filter_torlance)
                    add_filter_flag = 1;

                    filter_index_list_unit = 1;
                    while filter_index_list_unit <=  length(filter_index_list)
                        x_filter_index = filter_index_list(filter_index_list_unit, :);

                        % get h(x) of x and x_filter
                        con_filter_nomlz = [];
                        if ~isempty(con_nomlz_list)
                            con_filter_nomlz = max(con_nomlz_list(x_filter_index, :));
                        end
                        coneq_filter_nomlz = [];
                        if ~isempty(coneq_nomlz_list)
                            coneq_filter_nomlz = max(coneq_nomlz_list(x_filter_index, :));
                        end
                        total_con_filter_nomlz = max([con_filter_nomlz;coneq_filter_nomlz]);

                        % if cannot improve filter, reject it
                        if (fval_nomlz_list(x_index) > fval_nomlz_list(x_filter_index)) && ...
                                (total_con_x_nomlz > total_con_filter_nomlz)
                            add_filter_flag = 0;
                            break;
                        end

                        % if better than filter, reject filter
                        if (fval_nomlz_list(x_index) < fval_nomlz_list(x_filter_index)) && ...
                                (total_con_x_nomlz < total_con_filter_nomlz)
                            filter_index_list(filter_index_list_unit) = [];
                            filter_index_list_unit = filter_index_list_unit-1;
                        end

                        filter_index_list_unit = filter_index_list_unit+1;
                    end
                    % add into filter list if possible
                    if add_filter_flag
                        filter_index_list = [filter_index_list;x_index];
                    end
                else
                    feasible_index_list = [feasible_index_list;x_index];
                end
            end
            %         last_end_index = size(x_list, 1);
            if length(feasible_index_list) > 0.2*size(x_list, 1)
                filter_torlance = filter_torlance/2;
            end

            fval_label = -ones(size(x_list, 1), 1);
            fval_label(filter_index_list) = 1;

            % feasible point set label 1
            fval_label(feasible_index_list) = 1;

            % use filter and train SVM
            [predict_function_GPC, model_GPC] = classifyGaussProcess...
                (x_list, fval_label, GPC_hyp);
            GPC_hyp = model_GPC.hyp;
            if DRAW_FIGURE_FLAG && variable_number < 3
                classifyVisualization...
                    (model_GPC, low_bou, up_bou);
            end

            % get data to obtain clustering center
            x_sup_list = x_data_list(predict_function_GPC(x_data_list)  ==  1, :);

            if isempty(x_sup_list)
                % no sup point found use filter point
                if isempty(feasible_index_list)
                    if ~isempty(con_nomlz_list)
                        con_filter_nomlz_list = con_nomlz_list(filter_index_list);
                    else
                        con_filter_nomlz_list = [];
                    end
                    if ~isempty(coneq_nomlz_list)
                        coneq_filter_nomlz_list = coneq_nomlz_list(filter_index_list);
                    else
                        coneq_filter_nomlz_list = [];
                    end
                    max_totalcon_list = max([con_filter_nomlz_list, coneq_filter_nomlz_list], [], 2);
                    [~, filter_min_index] = min(max_totalcon_list);
                    x_center = x_list(filter_index_list(filter_min_index), :);
                else
                    [~, min_fval_index] = min(fval_list(feasible_index_list));
                    x_center = x_list(feasible_index_list(min_fval_index), :);
                end
            end
        
            object_function_PF=@(X) PFFunction(object_function_surrogate_KS,X);
        else
            
            % interset sampling
            %             [fval_sort, index_list] = sort(fval_list);
            %             x_list_sort = x_list(index_list, :);
            %             index = floor(size(fval_list, 1)/2);
            %             fval_thresh = fval_sort(index);
            fval_thresh = prctile(fval_list, 25);

            % step 7-1
            % classify exist data
            fval_label = -ones(size(x_list, 1), 1);
            boolean_list = fval_list <= fval_thresh;
            fval_label(boolean_list) = 1;
            x_positive_list = x_list(boolean_list,:);

            % step 7-2
            % get a large number of x point, use SVM to predict x point
            [predict_function_GPC, model_GPC] = classifyGaussProcess...
                (x_list, fval_label, GPC_hyp);
            GPC_hyp = model_GPC.hyp;
            if DRAW_FIGURE_FLAG && variable_number < 3
                classifyVisualization...
                    (model_GPC, low_bou, up_bou);
            end

            EI_function = @(X) EIFunction(object_function_surrogate, X, min(fval_nomlz_list));
            IF_function = @(X) IFFunction(x_list, X, exp(kriging_model_fval.hyp), variable_number);
            LCB_function = @(X) -LCBFunction(object_function_surrogate, X, 10);
            PGPC_function = @(X) PGPCFunction(predict_function_GPC, X);

            x_data_list = lhsdesign(sample_number_initial*10, variable_number)...
                .*(up_bou - low_bou) + low_bou;

            % new function
            PGPCEIF_function = @(X) LCB_function(X).*PGPC_function(X);

%             % mulit start search
%             fval_PGPCEIF = 1;
%             for x_index = 1:size(x_positive_list,1)
%                 [x_potential_iter, fval_PGPCEIF_iter] = fmincon(PGPCEIF_function,x_positive_list(x_index,:), [], [], [], [], low_bou, up_bou,...
%                     cheapcon_function, optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'none', 'OptimalityTolerance', 1e-2));
%                 if fval_PGPCEIF_iter < fval_PGPCEIF
%                     x_potential = x_potential_iter;
%                     fval_PGPCEIF = fval_PGPCEIF_iter;
%                 end
%             end
            
            % start in min fval
            [~, index] = min(PGPC_function(x_positive_list));
            [x_potential, fval_PGPCEIF] = fmincon(PGPCEIF_function, x_positive_list(index,:), [], [], [], [], low_bou, up_bou,...
                cheapcon_function, optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'none'));

            if DRAW_FIGURE_FLAG && variable_number < 3
                interpVisualize(kriging_model_fval, low_bou, up_bou);
                line(x_potential(1), x_potential(2), object_function_surrogate(x_potential), 'Marker', 'o', 'color', 'r', 'LineStyle', 'none')
            end

            x_updata_list = x_potential;
        end

%         % step 7-3
%         % calculate clustering center
%         if ~isempty(x_sup_list)
%             FC_model = classifyFuzzyClustering...
%                 (x_sup_list, 1, m);
%             x_center = FC_model.center_list;
%         end

%         % updata ISR
%         x_potential_nomlz = (x_potential-low_bou)./(up_bou-low_bou);
%         x_center_nomlz = (x_center-low_bou)./(up_bou-low_bou);
%         bou_range_nomlz = eta*norm(x_potential_nomlz-x_center_nomlz, 2);
%         if bou_range_nomlz < 1e-2
%             bou_range_nomlz = 1e-2;
%         end
%         bou_range = bou_range_nomlz.*(up_bou-low_bou);
%         low_bou_ISR = x_potential-bou_range;
%         low_bou_ISR = max(low_bou_ISR, low_bou);
%         up_bou_ISR = x_potential+bou_range;
%         up_bou_ISR = min(up_bou_ISR, up_bou);

%         if DRAW_FIGURE_FLAG && variable_number < 3
%             bou_line = [low_bou_ISR;[low_bou_ISR(1), up_bou_ISR(2)];up_bou_ISR;[up_bou_ISR(1), low_bou_ISR(2)];low_bou_ISR];
%             line(bou_line(:, 1), bou_line(:, 2));
%             line(x_potential(1), x_potential(2), 'Marker', 'x')
%         end

%         % sampling in ISR
%         [x_list_exist, ~, ~, ~] = dataLibraryLoad...
%             (data_library_name, low_bou_ISR, up_bou_ISR);
%         %     [~, x_updata_list, ~] = getLatinHypercube...
%         %         (sample_number_iteration+size(x_list_exist, 1), variable_number, x_list_exist, ...
%         %         low_bou_ISR, up_bou_ISR, cheapcon_function);
%         x_updata_list = lhsdesign(sample_number_iteration, variable_number)...
%             .*(up_bou_ISR-low_bou_ISR)+low_bou_ISR;

    end

    x_potential_old = x_potential;
    fval_potential_old = fval_potential;
    fval_best_old = fval_best;
end
result_x_best = result_x_best(1:iteration-1, :);
result_fval_best = result_fval_best(1:iteration-1);

output.result_x_best = result_x_best;
output.result_fval_best = result_fval_best;

    function [fval, con, coneq] = modelFunction...
            (x, object_function, nonlcon_function)
        % model function, concertrate fval, con, coneq into one function
        %
        if nargin < 3 || isempty(nonlcon_function)
            con = [];
            coneq = [];
        else
            [con, coneq] = nonlcon_function(x);
        end
        fval = object_function(x);
    end

    function fval = objectNonlconFunctionSurrogate(x, nonlcon_function_surrogate)
        [con__, coneq__] = nonlcon_function_surrogate(x);
        fval = 0;
        if ~isempty(con__)
            fval = fval+sum(max(con__, 0).^2);
        end
        if ~isempty(coneq__)
            fval = fval+sum(max(con__, 0).^2);
        end
    end

    function [x_updata_list, fval_updata_list, con_updata_list, coneq_updata_list, NFE_updata, repeat_index] = dataLibraryUpdataProtect...
            (data_library_name, model_function, x_add_list, ...
            x_list, low_bou, up_bou, protect_range)
        % function updata data with same_point_avoid protect
        % return fval
        % all list is x_number x variable_number matrix
        % notice if x_add is exist in library, point will be delete
        %
        variable_number__ = size(x_list, 2);
        NFE_updata = 0;
        x_updata_list = [];fval_updata_list = [];con_updata_list = [];coneq_updata_list = [];repeat_index = [];
        for x_index__ = 1:size(x_add_list, 1)
            x_updata__ = x_add_list(x_index__, :);

            % check x_potential if exist in data library
            % if not, updata data libraray
            distance__ = sum((abs(x_updata__-x_list)./(up_bou-low_bou)), 2);
            [~, min_index__] = min(distance__);
            if distance__(min_index__) < variable_number__*protect_range
                % distance to exist point of point to add is small than protect_range
                repeat_index = [repeat_index;min_index__];
            else
                [fval_updata__, con_updata__, coneq_updata__] = dataLibraryUpdata...
                    (data_library_name, model_function, x_updata__);NFE_updata = NFE_updata+1;
                x_updata_list = [x_updata_list;x_updata__];
                fval_updata_list = [fval_updata_list;fval_updata__];
                con_updata_list = [con_updata_list;con_updata__];
                coneq_updata_list = [coneq_updata_list;coneq_updata__];
            end
        end
    end

    function fval = EIFunction(object_function_surrogate, X, fval_min)
        % EI function
        %
        [Fval_pred, Fval_var] = object_function_surrogate(X);
        normal_fval = (fval_min - Fval_pred)./sqrt(Fval_var);
        EI_l = (fval_min - Fval_pred).*normcdf(normal_fval);
        EI_g = Fval_var.*normpdf(normal_fval);
        fval = EI_l + EI_g;
    end

    function fval = PFFunction(object_function_surrogate, X)
        % PF function
        [Con_pred,Con_var]=object_function_surrogate(X);
        fval=normcdf(-Con_pred./sqrt(Con_var));
    end

    function fval = IFFunction(x_list, X, theta, variable_number)
        fval = zeros(size(X, 1), size(x_list, 1));
        for variable_index = 1:variable_number
            fval = fval + (X(:, variable_index) - x_list(:, variable_index)').^2*theta(variable_index);
        end
        fval = min(1 - exp(-fval), [], 2);
    end

    function fval = LCBFunction(object_function_surrogate, X, w)
        [fval_pred,fval_var]=object_function_surrogate(X);
        fval=fval_pred-w*fval_var;
    end

    function fval = PGPCFunction(predict_function_GPC, X)
        [~, fval] = predict_function_GPC(X);
    end
end

%% auxiliary function
function [x_best, fval_best, exitflag, output] = findMinMSP...
    (object_function_surrogate, variable_number, low_bou, up_bou, nonlcon_function_surrogate, ...
    cheapcon_function, nonlcon_torlance)
% find min fval use MSP guideline
% MSP: object_funtion is object_function (generate by surrogate model)
% nonlcon_function generate by surrogate model
% use ga as optimal method
%
if nargin < 6
    cheapcon_function = [];
    if nargin < 5
        nonlcon_function_surrogate = [];
        if nargin < 4
            up_bou = [];
            if nargin < 3
                low_bou = [];
            end
        end
    end
end

% obtian total constraint function
if ~isempty(nonlcon_function_surrogate) || ~isempty(cheapcon_function)
    constraint_function = @(x) totalconFunction...
        (x, nonlcon_function_surrogate, cheapcon_function, nonlcon_torlance);
else
    constraint_function = [];
end

% generate initial population for ga
population_matrix = zeros(max(10, 2*variable_number), variable_number);
for population_index = 1:size(population_matrix, 1)
    x = rand(1, variable_number).*(up_bou-low_bou)+low_bou;
    if ~isempty(cheapcon_function)
        [con, coneq] = cheapcon_function(x);
        while sum([~(con < 0);abs(coneq) < 0])
            x = rand(1, variable_number).*(up_bou-low_bou)+low_bou;
            [con, coneq] = cheapcon_function(x);
        end
    end
    population_matrix(population_index, :) = x;
end

% optiaml
ga_option = optimoptions('ga', 'FunctionTolerance', 1e-2, 'ConstraintTolerance', 1e-2, ...
    'PopulationSize', max(10, 2*variable_number), ...
    'MaxGenerations', 100, 'InitialPopulationMatrix', population_matrix, ...
    'display', 'none');
[x_best, fval_best, exitflag, output] = ga...
    (object_function_surrogate, variable_number, [], [], [], [], low_bou', up_bou', constraint_function, ga_option);
fmincon_option = optimoptions('fmincon', 'FunctionTolerance', 1e-6, 'ConstraintTolerance', 1e-6, ...
    'algorithm', 'sqp', ....
    'display', 'none');
[x_best, fval_best, exitflag, output] = fmincon...
    (object_function_surrogate, x_best, [], [], [], [], low_bou, up_bou, constraint_function, fmincon_option);

    function [con, coneq] = totalconFunction...
            (x, nonlcon_function, cheapcon_function, nonlcon_torlance)
        con = [];
        coneq = [];
        if ~isempty(nonlcon_function)
            [expencon, expenconeq] = nonlcon_function(x);
            con = [con;expencon-nonlcon_torlance];
            coneq = [coneq;expenconeq-nonlcon_torlance];
        end
        if ~isempty(cheapcon_function)
            [expencon, expenconeq] = cheapcon_function(x);
            con = [con;expencon];
            coneq = [coneq;expenconeq];
        end
    end
end

function [x_best, fval_best, exitflag, output, LCB_function] = findMinLCB...
    (object_function_surrogate, variable_number, low_bou, up_bou, nonlcon_function_surrogate, ...
    cheapcon_function, nonlcon_torlance)
% find min fval use LCB guideline
% LCB: object_funtion is ...
% object_function (generate by surrogate model)...
% subtract sqrt(object_var_function(x)) (generate by surrogate model)
% nonlcon_function generate by surrogate model
% use ga as optimal method
%
if nargin < 6
    cheapcon_function = [];
    if nargin < 5
        nonlcon_function_surrogate = [];
        if nargin < 4
            up_bou = [];
            if nargin < 3
                low_bou = [];
            end
        end
    end
end

% obtian total constraint function
if ~isempty(nonlcon_function_surrogate) || ~isempty(cheapcon_function)
    constraint_function = @(x) totalconFunction...
        (x, nonlcon_function_surrogate, cheapcon_function, nonlcon_torlance);
else
    constraint_function = [];
end

% generate initial population for ga
population_matrix = zeros(max(10, 2*variable_number), variable_number);
for population_index = 1:size(population_matrix, 1)
    x = rand(1, variable_number).*(up_bou-low_bou)+low_bou;
    if ~isempty(cheapcon_function)
        [con, coneq] = cheapcon_function(x);
        while sum([~(con < 0);abs(coneq) < 0])
            x = rand(1, variable_number).*(up_bou-low_bou)+low_bou;
            [con, coneq] = cheapcon_function(x);
        end
    end
    population_matrix(population_index, :) = x;
end

% generate initial population for ga
population_matrix = zeros(2*variable_number, variable_number);
for population_index = 1:2*variable_number
    x = rand(1, variable_number).*(up_bou-low_bou)+low_bou;
    if ~isempty(cheapcon_function)
        [con, coneq] = cheapcon_function(x);
        while sum(~(con < 0))
            x = rand(1, variable_number).*(up_bou-low_bou)+low_bou;
            [con, coneq] = cheapcon_function(x);
        end
    end
    population_matrix(population_index, :) = x;
end

LCB_function = @(x) LCBFunction(object_function_surrogate, x);

% optiaml
ga_option = optimoptions('ga', 'FunctionTolerance', 1e-2, 'ConstraintTolerance', 1e-2, ...
    'PopulationSize', max(10, 2*variable_number), ...
    'MaxGenerations', 100, 'InitialPopulationMatrix', population_matrix, ...
    'display', 'none');
[x_best, fval_best, exitflag, output] = ga...
    (LCB_function, variable_number, [], [], [], [], low_bou, up_bou, constraint_function, ga_option);
fmincon_option = optimoptions('fmincon', 'FunctionTolerance', 1e-6, 'ConstraintTolerance', 1e-6, ...
    'algorithm', 'sqp', ....
    'display', 'none');
[x_best, fval_best, exitflag, output] = fmincon...
    (LCB_function, x_best, [], [], [], [], low_bou, up_bou, constraint_function, fmincon_option);

    function LCB = LCBFunction(object_function_surrogate, x)
        [pred, var] = object_function_surrogate(x);
        LCB = pred-var;
    end
    function [con, coneq] = totalconFunction...
            (x, nonlcon_function, cheapcon_function, nonlcon_torlance)
        con = [];
        coneq = [];
        if ~isempty(nonlcon_function)
            [expencon, expenconeq] = nonlcon_function(x);
            con = [con;expencon-nonlcon_torlance];
            coneq = [coneq;expenconeq-nonlcon_torlance];
        end
        if ~isempty(cheapcon_function)
            [expencon, expenconeq] = cheapcon_function(x);
            con = [con;expencon];
            coneq = [coneq;expenconeq];
        end
    end

end

function [x_best, fval_best, con_best, coneq_best] = findMinRaw...
    (x_list, fval_list, con_list, coneq_list, ...
    cheapcon_function, nonlcon_torlance)
% find min fval in raw data
% x_list, rank is variable
% con_list, rank is con
% coneq_list, rank is coneq
% function will find min fval in con == 0
% if there was not feasible x, find min consum
%
con_best = [];
coneq_best = [];
max_nonlcon_list = zeros(size(x_list, 1), 1);
max_cheapcon_list = zeros(size(x_list, 1), 1);
% process expendsive con
if ~isempty(con_list)
    max_nonlcon_list = max(con_list, [], 2);
end
if ~isempty(coneq_list)
    max_nonlcon_list = max(abs(coneq_list), [], 2);
end

% add cheap con
for x_index = 1:size(x_list, 1)
    if ~isempty(cheapcon_function)
        [con, coneq] = cheapcon_function(x_list(x_index, :));
        max_cheapcon_list(x_index) = max_cheapcon_list(x_index)+...
            sum(max(con, 0))+sum(coneq.*coneq);
    end
end

con_judge_list = (max_nonlcon_list > nonlcon_torlance)+...
    (max_cheapcon_list > 0);
index = find(con_judge_list  ==  0);
if ~isempty(index)
    % feasible x
    x_list = x_list(index, :);
    fval_list = fval_list(index);
    if ~isempty(con_list)
        con_list = con_list(index, :);
    end
    if ~isempty(coneq_list)
        coneq_list = coneq_list(index, :);
    end

    % min fval
    [fval_best, index_best] = min(fval_list);
    x_best = x_list(index_best, :);
    if ~isempty(con_list)
        con_best = con_list(index_best, :);
    end
    if ~isempty(coneq_list)
        coneq_best = coneq_list(index_best, :);
    end
else
    % min consum
    [~, index_best] = min(max_nonlcon_list);
    fval_best = fval_list(index_best);
    x_best = x_list(index_best, :);
    if ~isempty(con_list)
        con_best = con_list(index_best, :);
    end
    if ~isempty(coneq_list)
        coneq_best = coneq_list(index_best, :);
    end
end
end

function [quantile, normal_index, small_index, large_index] = findUnusual(data)
% base on Box diagram to search unusual value
%
x_number = length(data);
[data, index_list] = sort(data);

Q1 = getQuantile(data, 0.25);
Q3 = getQuantile(data, 0.75);
IQR = Q3-Q1;

normal_index = 1:x_number;
small_index = normal_index(data < (Q1-1.5*IQR));
large_index = normal_index(data > (Q3+1.5*IQR));
normal_index([large_index, small_index]) = [];

quantile = [min(data(normal_index));
    getQuantile(data(normal_index), 0.25);
    getQuantile(data(normal_index), 0.5);
    getQuantile(data(normal_index), 0.75);
    max(data(normal_index))];

[~, index_list] = sort(index_list);
normal_index = index_list(normal_index);
small_index = index_list(small_index);
large_index = index_list(large_index);

    function quantile = getQuantile(data, percent)
        index = length(data)*percent;
        if (index  ==  fix(index))
            % index is integer
            quantile = 0.5*(data(index)+data(index+1));
        else
            quantile = 0.5*data(ceil(index));
        end
    end
end

%% FCM
function FC_model = classifyFuzzyClustering...
    (X, classify_number, m)
% get fuzzy cluster model
% X is x_number x variable_number matrix
% center_list is classify_number x variable_number matrix
%
iteration_max = 100;
torlance = 1e-6;

[x_number, variable_number] = size(X);

% normalization data
aver_X = mean(X);
stdD_X = std(X);
index__ = find(stdD_X  ==  0);
if  ~isempty(index__), stdD_X(index__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;

% if x_number equal 1, clustering cannot done
if x_number  ==  1
    FC_model.X = X;
    FC_model.X_normalize = X_nomlz;
    FC_model.center_list = X;
    FC_model.fval_loss_list = [];
    return;
end

U = zeros(classify_number, x_number);
center_list = rand(classify_number, variable_number)*0.5;
iteration = 0;
done = 0;
fval_loss_list = zeros(iteration_max, 1);

% get X_center_dis_sq
X_center_dis_sq = zeros(classify_number, x_number);
for classify_index = 1:classify_number
    for x_index = 1:x_number
        X_center_dis_sq(classify_index, x_index) = ...
            getSq((X_nomlz(x_index, :)-center_list(classify_index, :)));
    end
end

while ~done
    % updata classify matrix U
    for classify_index = 1:classify_number
        for x_index = 1:x_number
            U(classify_index, x_index) = ...
                1/sum((X_center_dis_sq(classify_index, x_index)./X_center_dis_sq(:, x_index)).^(1/(m-1)));
        end
    end
    
    % updata center_list
    center_list_old = center_list;
    for classify_index = 1:classify_number
        center_list(classify_index, :) = ...
            sum((U(classify_index, :)').^m.*X_nomlz, 1)./...
            sum((U(classify_index, :)').^m, 1);
    end
    
    % updata X_center_dis_sq
    X_center_dis_sq = zeros(classify_number, x_number);
    for classify_index = 1:classify_number
        for x_index = 1:x_number
            X_center_dis_sq(classify_index, x_index) = ...
                getSq((X_nomlz(x_index, :)-center_list(classify_index, :)));
        end
    end
    
%     plot(center_list(:, 1), center_list(:, 2));
    
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

%% GPC
function [predict_function, CGP_model] = classifyGaussProcess...
    (X, Class, hyp)
% generate gaussian process classifier model
% version 6, this version is assembly of gpml-3.6 EP method
% X is x_number x variable_number matirx, Y is x_number x 1 matrix
% low_bou, up_bou is 1 x variable_number matrix
% only support binary classification, -1 and 1
%
% input:
% X, Class, hyp(mean, cov(len, eta))
%
% abbreviation:
% pred: predicted, nomlz: normalization, num: number
% var: variance
%
[x_number, variable_number] = size(X);
if nargin < 5
    hyp.mean = 0;
    hyp.cov = zeros(1, 2);
end

% normalization data
aver_X = mean(X);
stdD_X = std(X);
index__ = find(stdD_X == 0);
if  ~isempty(index__), stdD_X(index__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;

object_function = @(x) objectNLLGPC(x, {@infEP}, {@meanConst}, {@calCov}, {@likErf}, X_nomlz, Class);
hyp_x = [hyp.mean, hyp.cov];

% [fval, gradient] = object_function(hyp_x)
% [fval_differ, gradient_differ] = differ(object_function, hyp_x)

hyp_low_bou = -3*ones(1, 3);
hyp_up_bou = 3*ones(1, 3);
hyp_x = fmincon(object_function, hyp_x, [], [], [], [], hyp_low_bou, hyp_up_bou, [], ...
    optimoptions('fmincon', 'Display', 'none', 'SpecifyObjectiveGradient', true, ...
    'MaxFunctionEvaluations', 20, 'OptimalityTolerance', 1e-3));

hyp.mean = hyp_x(1);
hyp.cov = hyp_x(2:end);
hyp.lik = [];
post = infEP(hyp, {@meanConst}, {@calCov}, {@likErf}, X_nomlz, Class);
predict_function = @(x_pred) classifyGaussPredictor...
    (x_pred, hyp, {@meanConst}, {@calCov}, {@likErf}, post, X_nomlz, aver_X, stdD_X);

% output model
CGP_model.X = X;
CGP_model.Class = Class;
CGP_model.X_nomlz = X_nomlz;
CGP_model.aver_X = aver_X;
CGP_model.stdD_X = stdD_X;
CGP_model.predict_function = predict_function;
CGP_model.hyp = hyp;

    function [fval, gradient] = objectNLLGPC(x, inf, mean, cov, lik, X, Y)
        hyp_iter.mean = x(1);
        hyp_iter.cov = x(2:end);
        hyp_iter.lik = [];

        if nargout < 2
            [~, nlZ] = feval(inf{:}, hyp_iter, mean, cov, lik, X, Y);
            fval = nlZ;
        elseif nargout < 3
            [~, nlZ, dnlZ] = feval(inf{:}, hyp_iter, mean, cov, lik, X, Y);
            fval = nlZ;
            gradient = [dnlZ.mean, dnlZ.cov];
        end
    end

    function [class, possibility, miu_pre, var_pre] = classifyGaussPredictor...
            (X_pred, hyp, mean, cov, lik, post, X, aver_X, stdD_X)
        % predict function
        %
        X_pred_nomlz = (X_pred-aver_X)./stdD_X;
        pred_num = size(X_pred_nomlz, 1);
        ys = ones(pred_num, 1);

        alpha = post.alpha; L = post.L; sW = post.sW;
        nz = true(size(alpha, 1), 1);               % non-sparse representation
        %verify whether L contains valid Cholesky decomposition or something different
        Lchol = isnumeric(L) && all(all(tril(L, -1)==0)&diag(L)'>0&isreal(diag(L))');
        ns = size(X_pred_nomlz, 1);                                       % number of data points
        nperbatch = 1000;                       % number of data points per mini batch
        nact = 0;                       % number of already processed test data points
        ymu = zeros(ns, 1); ys2 = ymu; miu_pre = ymu; var_pre = ymu; possibility = ymu;   % allocate mem
        while nact<ns               % process minibatches of test cases to save memory
            id = (nact+1):min(nact+nperbatch, ns);               % data points to process
            kss = feval(cov{:}, hyp.cov, X_pred_nomlz(id, :), 'diag');              % self-variance
            Ks = feval(cov{:}, hyp.cov, X(nz, :), X_pred_nomlz(id, :));        % avoid computation
            ms = feval(mean{:}, hyp.mean, X_pred_nomlz(id, :));
            N = size(alpha, 2);  % number of alphas (usually 1; more in case of sampling)
            Fmu = repmat(ms, 1, N) + Ks'*full(alpha(nz, :));        % conditional mean fs|f
            miu_pre(id) = sum(Fmu, 2)/N;                                   % predictive means
            if Lchol    % L contains chol decomp => use Cholesky parameters (alpha, sW, L)
                V  = L'\(repmat(sW, 1, length(id)).*Ks);
                var_pre(id) = kss - sum(V.*V, 1)';                       % predictive variances
            else                % L is not triangular => use alternative parametrisation
                if isnumeric(L), LKs = L*Ks; else LKs = L(Ks); end    % matrix or callback
                var_pre(id) = kss + sum(Ks.*LKs, 1)';                    % predictive variances
            end
            var_pre(id) = max(var_pre(id), 0);   % remove numerical noise i.e. negative variances
            Fs2 = repmat(var_pre(id), 1, N);     % we have multiple values in case of sampling
            if nargin<9
                [Lp, Ymu, Ys2] = feval(lik{:}, hyp.lik, [], Fmu(:), Fs2(:));
            else
                Ys = repmat(ys(id), 1, N);
                [Lp, Ymu, Ys2] = feval(lik{:}, hyp.lik, Ys(:), Fmu(:), Fs2(:));
            end
            possibility(id)  = sum(reshape(Lp, [], N), 2)/N;    % log probability; sample averaging
            ymu(id) = sum(reshape(Ymu, [], N), 2)/N;          % predictive mean ys|y and ..
            ys2(id) = sum(reshape(Ys2, [], N), 2)/N;                          % .. variance
            nact = id(end);          % set counter to index of last processed data point
        end

        possibility = exp(possibility);
        class = ones(pred_num, 1);
        index_list = find(possibility < 0.5);
        class(index_list) = -1;
    end

    function [K, dK_dcov] = calCov(cov, X, Z)
        % obtain covariance of x
        % cov: eta, len(equal to 1/len.^2)
        %
        % k = eta*exp(-sum(x_dis*len)/vari_num);
        %
        [x_num, vari_num] = size(X);

        len = exp(cov(1));
        eta = exp(cov(2));

        % predict
        if nargin > 2 && nargout < 2 && ~isempty(Z)
            if strcmp(Z, 'diag')
                K = eta;
            else
                [z_number, vari_num] = size(Z);
                % initializate square of X inner distance/ vari_num
                K = zeros(x_num, z_number);
                for len_index = 1:vari_num
                    K = K+(X(:, len_index)-Z(:, len_index)').^2*len/vari_num;
                end
                K = eta*exp(-K);
            end
        else
            % initializate square of X inner distance sq
            sq_dis_v = zeros(x_num, x_num, vari_num);
            for len_index = 1:vari_num
                sq_dis_v(:, :, len_index) = (X(:, len_index)-X(:, len_index)').^2/vari_num;
            end

            % exp of x__x with theta
            exp_dis = zeros(x_num);
            for len_index = 1:vari_num
                exp_dis = exp_dis+sq_dis_v(:, :, len_index)*len;
            end
            exp_dis = exp(-exp_dis);
            K = exp_dis*eta;

            if nargout >= 2
                dK_dcov = cell(1, 2);
                dK_dlen = zeros(x_num, x_num);
                for len_index = 1:vari_num
                    dK_dlen = dK_dlen + sq_dis_v(:, :, len_index);
                end
                dK_dlen = -dK_dlen.*K*len;
                dK_dcov{1} = dK_dlen;

                dK_dcov{2} = K;
            end
        end

    end

    function [post nlZ dnlZ] = infEP(hyp, mean, cov, lik, x, y)
        % Expectation Propagation approximation to the posterior Gaussian Process.
        % The function takes a specified covariance function (see covFunctions.m) and
        % likelihood function (see likFunctions.m), and is designed to be used with
        % gp.m. See also infMethods.m. In the EP algorithm, the sites are
        % updated in random order, for better performance when cases are ordered
        % according to the targets.
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch 2013-09-13.
        %
        % See also INFMETHODS.M.
        %
        persistent last_ttau last_tnu              % keep tilde parameters between calls
        tol = 1e-4; max_sweep = 10; min_sweep = 2;     % tolerance to stop EP iterations

        inf = 'infEP';
        n = size(x, 1);
        if isnumeric(cov), K = cov;                    % use provided covariance matrix
        else K = feval(cov{:}, hyp.cov, x); end       % evaluate the covariance matrix
        if isnumeric(mean), m = mean;                         % use provided mean vector
        else m = feval(mean{:}, hyp.mean, x); end             % evaluate the mean vector

        % A note on naming: variables are given short but descriptive names in
        % accordance with Rasmussen & Williams "GPs for Machine Learning" (2006): mu
        % and s2 are mean and variance, nu and tau are natural parameters. A leading t
        % means tilde, a subscript _ni means "not i" (for cavity parameters), or _n
        % for a vector of cavity parameters. N(f|mu, Sigma) is the posterior.

        % marginal likelihood for ttau = tnu = zeros(n, 1); equals n*log(2) for likCum*
        nlZ0 = -sum(feval(lik{:}, hyp.lik, y, m, diag(K), inf));
        if any(size(last_ttau) ~= [n 1])      % find starting point for tilde parameters
            ttau = zeros(n, 1); tnu  = zeros(n, 1);        % init to zero if no better guess
            Sigma = K;                     % initialize Sigma and mu, the parameters of ..
            mu = m; nlZ = nlZ0;                  % .. the Gaussian posterior approximation
        else
            ttau = last_ttau; tnu  = last_tnu;   % try the tilde values from previous call
            [Sigma, mu, L, alpha, nlZ] = epComputeParams(K, y, ttau, tnu, lik, hyp, m, inf);
            if nlZ > nlZ0                                           % if zero is better ..
                ttau = zeros(n, 1); tnu  = zeros(n, 1);       % .. then init with zero instead
                Sigma = K;                   % initialize Sigma and mu, the parameters of ..
                mu = m; nlZ = nlZ0;                % .. the Gaussian posterior approximation
            end
        end

        nlZ_old = Inf; sweep = 0;               % converged, max. sweeps or min. sweeps?
        while (abs(nlZ-nlZ_old) > tol && sweep < max_sweep) || sweep<min_sweep
            nlZ_old = nlZ; sweep = sweep+1;
            for i = randperm(n)       % iterate EP updates (in random order) over examples
                tau_ni = 1/Sigma(i, i)-ttau(i);      %  first find the cavity distribution ..
                nu_ni = mu(i)/Sigma(i, i)-tnu(i);                % .. params tau_ni and nu_ni

                % compute the desired derivatives of the indivdual log partition function
                [lZ, dlZ, d2lZ] = feval(lik{:}, hyp.lik, y(i), nu_ni/tau_ni, 1/tau_ni, inf);
                ttau_old = ttau(i); tnu_old = tnu(i);  % find the new tilde params, keep old
                ttau(i) =                     -d2lZ  /(1+d2lZ/tau_ni);
                ttau(i) = max(ttau(i), 0); % enforce positivity i.e. lower bound ttau by zero
                tnu(i)  = ( dlZ - nu_ni/tau_ni*d2lZ )/(1+d2lZ/tau_ni);

                dtt = ttau(i)-ttau_old; dtn = tnu(i)-tnu_old;      % rank-1 update Sigma ..
                si = Sigma(:, i); ci = dtt/(1+dtt*si(i));
                Sigma = Sigma - ci*si*si';                         % takes 70% of total time
                mu = mu - (ci*(mu(i)+si(i)*dtn)-dtn)*si;               % .. and recompute mu
            end
            % recompute since repeated rank-one updates can destroy numerical precision
            [Sigma, mu, L, alpha, nlZ] = epComputeParams(K, y, ttau, tnu, lik, hyp, m, inf);
        end

        if sweep == max_sweep && abs(nlZ-nlZ_old) > tol
            error('maximum number of sweeps exceeded in function infEP')
        end

        last_ttau = ttau; last_tnu = tnu;                       % remember for next call
        post.alpha = alpha; post.sW = sqrt(ttau); post.L = L;  % return posterior params

        if nargout>2                                           % do we want derivatives?
            dnlZ = hyp;                                   % allocate space for derivatives
            tau_n = 1./diag(Sigma)-ttau;             % compute the log marginal likelihood
            nu_n  = mu./diag(Sigma)-tnu;                    % vectors of cavity parameters
            sW = sqrt(ttau);
            F = alpha*alpha'-repmat(sW, 1, n).*(L\(L'\diag(sW)));   % covariance hypers
            [K, dK] = feval(cov{:}, hyp.cov, x, []);
            for i = 1:length(hyp.cov)
                dnlZ.cov(i) = -sum(sum(F.*dK{i}))/2;
            end
            for i = 1:numel(hyp.lik)                                   % likelihood hypers
                dlik = feval(lik{:}, hyp.lik, y, nu_n./tau_n, 1./tau_n, inf, i);
                dnlZ.lik(i) = -sum(dlik);
            end
            [junk, dlZ] = feval(lik{:}, hyp.lik, y, nu_n./tau_n, 1./tau_n, inf);% mean hyps
            for i = 1:numel(hyp.mean)
                dm = feval(mean{:}, hyp.mean, x, i);
                dnlZ.mean(i) = -dlZ'*dm;
            end
        end
    end

    function [Sigma, mu, L, alpha, nlZ] = epComputeParams(K, y, ttau, tnu, lik, hyp, m, inf)
        % function to compute the parameters of the Gaussian approximation, Sigma and
        % mu, and the negative log marginal likelihood, nlZ, from the current site
        % parameters, ttau and tnu. Also returns L (useful for predictions).
        %
        n = length(y);                                      % number of training cases
        sW = sqrt(ttau);                                        % compute Sigma and mu
        L = chol(eye(n)+sW*sW'.*K);                            % L'*L = B = eye(n)+sW*K*sW
        V = L'\(repmat(sW, 1, n).*K);
        Sigma = K - V'*V;
        alpha = tnu-sW.*(L\(L'\(sW.*(K*tnu+m))));
        mu = K*alpha+m; v = diag(Sigma);

        tau_n = 1./diag(Sigma)-ttau;             % compute the log marginal likelihood
        nu_n  = mu./diag(Sigma)-tnu;                    % vectors of cavity parameters
        lZ = feval(lik{:}, hyp.lik, y, nu_n./tau_n, 1./tau_n, inf);
        p = tnu-m.*ttau; q = nu_n-m.*tau_n;                        % auxiliary vectors
        nlZ = sum(log(diag(L))) - sum(lZ) - p'*Sigma*p/2 + (v'*p.^2)/2 ...
            - q'*((ttau./tau_n.*q-2*p).*v)/2 - sum(log(1+ttau./tau_n))/2;
    end

    function A = meanConst(hyp, x, i)

        % Constant mean function. The mean function is parameterized as:
        %
        % m(x) = c
        %
        % The hyperparameter is:
        %
        % hyp = [ c ]
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-08-04.
        %
        % See also MEANFUNCTIONS.M.

        if nargin<2, A = '1'; return; end             % report number of hyperparameters
        if numel(hyp)~=1, error('Exactly one hyperparameter needed.'), end
        c = hyp;
        if nargin==2
            A = c*ones(size(x, 1), 1);                                       % evaluate mean
        else
            if i==1
                A = ones(size(x, 1), 1);                                          % derivative
            else
                A = zeros(size(x, 1), 1);
            end
        end
    end

    function [varargout] = likErf(hyp, y, mu, s2, inf, i)
        % likErf - Error function or cumulative Gaussian likelihood function for binary
        % classification or probit regression. The expression for the likelihood is
        %   likErf(t) = (1+erf(t/sqrt(2)))/2 = normcdf(t).
        %
        % Several modes are provided, for computing likelihoods, derivatives and moments
        % respectively, see likFunctions.m for the details. In general, care is taken
        % to avoid numerical issues when the arguments are extreme.
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2014-03-19.
        %
        % See also LIKFUNCTIONS.M.
        %
        if nargin<3, varargout = {'0'}; return; end   % report number of hyperparameters
        if nargin>1, y = sign(y); y(y==0) = 1; else y = 1; end % allow only +/- 1 values
        if numel(y)==0, y = 1; end

        if nargin<5                              % prediction mode if inf is not present
            y = y.*ones(size(mu));                                       % make y a vector
            s2zero = 1; if nargin>3&&numel(s2)>0&&norm(s2)>eps, s2zero = 0; end  % s2==0 ?
            if s2zero                                         % log probability evaluation
                lp = logphi(y.*mu);
            else                                                              % prediction
                lp = likErf(hyp, y, mu, s2, 'infEP');
            end
            p = exp(lp); ymu = {}; ys2 = {};
            if nargout>1
                ymu = 2*p-1;                                                % first y moment
                if nargout>2
                    ys2 = 4*p.*(1-p);                                        % second y moment
                end
            end
            varargout = {lp, ymu, ys2};
        else                                                            % inference mode
            switch inf
                case 'infLaplace'
                    if nargin<6                                             % no derivative mode
                        f = mu; yf = y.*f;                            % product latents and labels
                        varargout = cell(nargout, 1); [varargout{:}] = logphi(yf);   % query logphi
                        if nargout>1
                            varargout{2} = y.*varargout{2};
                            if nargout>3, varargout{4} = y.*varargout{4}; end
                        end
                    else                                                       % derivative mode
                        varargout = {[], [], []};                         % derivative w.r.t. hypers
                    end

                case 'infEP'
                    if nargin<6                                             % no derivative mode
                        z = mu./sqrt(1+s2); dlZ = {}; d2lZ = {};
                        if numel(y)>0, z = z.*y; end
                        if nargout <= 1, lZ = logphi(z);                         % log part function
                        else          [lZ, n_p] = logphi(z); end
                        if nargout > 1
                            if numel(y)==0, y = 1; end
                            dlZ = y.*n_p./sqrt(1+s2);                      % 1st derivative wrt mean
                            if nargout>2, d2lZ = -n_p.*(z+n_p)./(1+s2); end         % 2nd derivative
                        end
                        varargout = {lZ, dlZ, d2lZ};
                    else                                                       % derivative mode
                        varargout = {[]};                                     % deriv. wrt hyp.lik
                    end
            end
        end
    end

    function [lp, dlp, d2lp, d3lp] = logphi(z)
        % Safe computation of logphi(z) = log(normcdf(z)) and its derivatives
        %                    dlogphi(z) = normpdf(x)/normcdf(x).
        % The function is based on index 5725 in Hart et al. and gsl_sf_log_erfc_e.
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2013-11-13.
        %
        z = real(z);                                 % support for real arguments only
        lp = zeros(size(z));                                         % allocate memory
        id1 = z.*z<0.0492;                                 % first case: close to zero
        lp0 = -z(id1)/sqrt(2*pi);
        c = [ 0.00048204; -0.00142906; 0.0013200243174; 0.0009461589032;
            -0.0045563339802; 0.00556964649138; 0.00125993961762116;
            -0.01621575378835404; 0.02629651521057465; -0.001829764677455021;
            2*(1-pi/3); (4-pi)/3; 1; 1];
        f = 0; for i = 1:14, f = lp0.*(c(i)+f); end, lp(id1) = -2*f-log(2);
        id2 = z<-11.3137;                                    % second case: very small
        r = [ 1.2753666447299659525; 5.019049726784267463450;
            6.1602098531096305441; 7.409740605964741794425;
            2.9788656263939928886 ];
        q = [ 2.260528520767326969592;  9.3960340162350541504;
            12.048951927855129036034; 17.081440747466004316;
            9.608965327192787870698;  3.3690752069827527677 ];
        num = 0.5641895835477550741; for i = 1:5, num = -z(id2).*num/sqrt(2) + r(i); end
        den = 1.0;                   for i = 1:6, den = -z(id2).*den/sqrt(2) + q(i); end
        e = num./den; lp(id2) = log(e/2) - z(id2).^2/2;
        id3 = ~id2 & ~id1; lp(id3) = log(erfc(-z(id3)/sqrt(2))/2);  % third case: rest
        if nargout>1                                        % compute first derivative
            dlp = zeros(size(z));                                      % allocate memory
            dlp( id2) = abs(den./num) * sqrt(2/pi); % strictly positive first derivative
            dlp(~id2) = exp(-z(~id2).*z(~id2)/2-lp(~id2))/sqrt(2*pi); % safe computation
            if nargout>2                                     % compute second derivative
                d2lp = -dlp.*abs(z+dlp);             % strictly negative second derivative
                if nargout>3                                    % compute third derivative
                    d3lp = -d2lp.*abs(z+2*dlp)-dlp;     % strictly positive third derivative
                end
            end
        end
    end

end

%% surrogate model
function [kriging_model_fval, kriging_model_con, kriging_model_coneq, output] = getKrigingModel...
    (x_list, fval_list, con_list, coneq_list, ...
    kriging_model_fval, kriging_model_con, kriging_model_coneq)
% base on library_data to create kriging model and function
% if input model, function will updata model
% object_function is multi fval output
% nonlcon_function is normal nonlcon_function which include con, coneq
% con is colume vector, coneq is colume vector
% var_function is same
%
if nargin < 5
    kriging_model_fval = [];
end

if size(x_list, 1) ~= size(fval_list, 1)
    error('getKrigingModel: x_list size no equal fval_list size')
end

if isempty(kriging_model_fval)
    [predict_function_fval, kriging_model_fval] = interpKrigingPreModel...
        (x_list, fval_list);
else
    [predict_function_fval, kriging_model_fval] = interpKrigingPreModel...
        (x_list, fval_list, kriging_model_fval.hyp);
end

if ~isempty(con_list)
    predict_function_con = cell(size(con_list, 2), 1);
    if size(x_list, 1) ~= size(con_list, 1)
        error('getKrigingModel: x_list size no equal con_list size')
    end
    if isempty(kriging_model_con)
        kriging_model_con = struct('X', [], 'Y', [], ...
            'fval_regression', [], 'covariance', [], 'inv_covariance', [], ...
            'hyp', [], 'beta', [], 'gama', [], 'sigma_sq', [], ...
            'aver_X', [], 'stdD_X', [], 'aver_Y', [], 'stdD_Y', [], ...
            'predict_function', []);
        kriging_model_con = repmat(kriging_model_con, 1, [size(con_list, 2)]);
        for con_index = 1:size(con_list, 2)
            [predict_function_con{con_index}, kriging_model_con(con_index)] = interpKrigingPreModel...
                (x_list, con_list(:, con_index));
        end
    else
        for con_index = 1:size(con_list, 2)
            [predict_function_con{con_index}, kriging_model_con(con_index)] = interpKrigingPreModel...
                (x_list, con_list(:, con_index), kriging_model_con(con_index).hyp);
        end
    end
else
    predict_function_con = [];
    kriging_model_con = [];
end

if ~isempty(coneq_list)
    predict_function_coneq = cell(size(coneq_list, 2), 1);
    if size(x_list, 1) ~= size(coneq_list, 1)
        error('getKrigingModel: x_list size no equal coneq_list size')
    end
    if isempty(kriging_model_coneq)
        kriging_model_coneq = struct('X', [], 'Y', [], ...
            'fval_regression', [], 'covariance', [], 'inv_covariance', [], ...
            'hyp', [], 'beta', [], 'gama', [], 'sigma_sq', [], ...
            'aver_X', [], 'stdD_X', [], 'aver_Y', [], 'stdD_Y', [], ...
            'predict_function', []);
        kriging_model_coneq = repmat(kriging_model_coneq, 1, [size(coneq_list, 2)]);
        for coneq_index = 1:size(coneq_list, 2)
            [predict_function_coneq{coneq_index}, kriging_model_coneq(coneq_index)] = interpKrigingPreModel...
                (x_list, coneq_list(:, coneq_index));
        end
    else
        for coneq_index = 1:size(coneq_list, 2)
            [predict_function_coneq{coneq_index}, kriging_model_coneq(coneq_index)] = interpKrigingPreModel...
                (x_list, coneq_list(:, coneq_index), kriging_model_coneq(coneq_index).hyp);
        end
    end
else
    predict_function_coneq = [];
    kriging_model_coneq = [];
end

object_function_surrogate = @(X_predict) objectFunctionSurrogate(X_predict, predict_function_fval);
if isempty(predict_function_con) && isempty(predict_function_coneq)
    nonlcon_function_surrogate = [];
else
    nonlcon_function_surrogate = @(X_predict) nonlconFunctionSurrogate(X_predict, predict_function_con, predict_function_coneq);
end

output.object_function_surrogate = object_function_surrogate;
output.nonlcon_function_surrogate = nonlcon_function_surrogate;

    function [fval, fval_var] = objectFunctionSurrogate...
            (X_predict, predict_function_fval)
        % connect all predict favl
        %
        [fval, fval_var] = predict_function_fval(X_predict);
    end

    function [con, con_var, coneq, coneq_var] = nonlconFunctionSurrogate...
            (X_predict, predict_function_con, predict_function_coneq)
        % connect all predict con and coneq
        %
        if isempty(predict_function_con)
            con = [];
            con_var = [];
        else
            con = zeros(size(X_predict, 1), length(predict_function_con));
            con_var = zeros(size(X_predict, 1), length(predict_function_con));
            for con_index__ = 1:length(predict_function_con)
                [con(:, con_index__), con_var(:, con_index__)] = ....
                    predict_function_con{con_index__}(X_predict);
            end
        end
        if isempty(predict_function_coneq)
            coneq = [];
            coneq_var = [];
        else
            coneq = zeros(size(X_predict, 1), length(predict_function_coneq));
            coneq_var = zeros(size(X_predict, 1), length(predict_function_coneq));
            for coneq_index__ = 1:length(predict_function_coneq)
                [coneq(:, coneq_index__), coneq_var(:, coneq_index__)] = ...
                    predict_function_coneq{coneq_index__}(X_predict);
            end
        end
    end
end

function [predict_function, kriging_model] = interpKrigingPreModel...
    (X, Y, hyp)
% nomalization method is grassian
% add multi x_predict input support
% prepare model, optimal theta and calculation parameter
% X, Y are x_number x variable_number matrix
% aver_X, stdD_X is 1 x x_number matrix
% theta beta gama sigma_sq is normalizede, so predict y is normalize
% theta = exp(hyp)
%
% input initial data X, Y, which are real data
%
% output is a kriging model, include predict_function...
% X, Y, base_function_list
%
% Copyright 2023.2 Adel
%
[x_number, variable_number] = size(X);
if nargin < 3
    hyp = 0;
end

% normalize data
aver_X = mean(X);
stdD_X = std(X);
aver_Y = mean(Y);
stdD_Y = std(Y);
index__ = find(stdD_X == 0);
if  ~isempty(index__), stdD_X(index__) = 1; end
index__ = find(stdD_Y == 0);
if  ~isempty(index__), stdD_Y(index__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;
Y_nomlz = (Y-aver_Y)./stdD_Y;

% initial X_dis_sq
X_dis_sq = zeros(x_number, x_number, variable_number);
for variable_index = 1:variable_number
    X_dis_sq(:, :, variable_index) = ...
        (X_nomlz(:, variable_index)-X_nomlz(:, variable_index)').^2;
end

% regression function define
% notice reg_function process no normalization data
% reg_function = @(X) regZero(X);
reg_function = @(X) regLinear(X);

% calculate reg
fval_reg_nomlz = (reg_function(X)-aver_Y)./stdD_Y;

% optimal to get hyperparameter
fmincon_option = optimoptions('fmincon', 'Display', 'none', ...
    'OptimalityTolerance', 1e-2, ...
    'FiniteDifferenceStepSize', 1e-5, ...,
    'MaxIterations', 10, 'SpecifyObjectiveGradient', false);
low_bou_hyp = -3;
up_bou_hyp = 3;
object_function_hyp = @(hyp) objectNLLKriging...
    (X_dis_sq, Y_nomlz, x_number, variable_number, hyp, fval_reg_nomlz);

% [fval, gradient] = object_function_hyp(hyp)
% [~, gradient_differ] = differ(object_function_hyp, hyp)

% drawFunction(object_function_hyp, low_bou_hyp, up_bou_hyp);

hyp = fmincon...
    (object_function_hyp, hyp, [], [], [], [], low_bou_hyp, up_bou_hyp, [], fmincon_option);

% get parameter
[covariance, inv_covariance, beta, sigma_sq] = interpKriging...
    (X_dis_sq, Y_nomlz, x_number, variable_number, exp(hyp), fval_reg_nomlz);
gama = inv_covariance*(Y_nomlz-fval_reg_nomlz*beta);
FTRF = fval_reg_nomlz'*inv_covariance*fval_reg_nomlz;

% initialization predict function
predict_function = @(X_predict) interpKrigingPredictor...
    (X_predict, X_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
    x_number, variable_number, exp(hyp), beta, gama, sigma_sq, ...
    inv_covariance, fval_reg_nomlz, FTRF, reg_function);

kriging_model.X = X;
kriging_model.Y = Y;
kriging_model.fval_regression = fval_reg_nomlz;
kriging_model.covariance = covariance;
kriging_model.inv_covariance = inv_covariance;

kriging_model.hyp = hyp;
kriging_model.beta = beta;
kriging_model.gama = gama;
kriging_model.sigma_sq = sigma_sq;
kriging_model.aver_X = aver_X;
kriging_model.stdD_X = stdD_X;
kriging_model.aver_Y = aver_Y;
kriging_model.stdD_Y = stdD_Y;

kriging_model.predict_function = predict_function;

% abbreviation:
% num: number, pred: predict, vari: variable, hyp: hyper parameter
% NLL: negative log likelihood
    function [fval, gradient] = objectNLLKriging...
            (X_dis_sq, Y, x_num, vari_num, hyp, F_reg)
        % function to minimize sigma_sq
        %
        theta = exp(hyp);
        [cov, inv_cov, ~, sigma2, inv_FTRF, Y_Fmiu] = interpKriging...
            (X_dis_sq, Y, x_num, vari_num, theta, F_reg);

        % calculation negative log likelihood
        L = chol(cov)';
        fval = x_num/2*log(sigma2)+sum(log(diag(L)));

        % calculate gradient
        if nargout > 1
            % gradient
            dcov_dtheta = zeros(x_num, x_num);
            for vari_index = 1:vari_num
                dcov_dtheta = dcov_dtheta + X_dis_sq(:, :, vari_index);
            end
            dcov_dtheta = -dcov_dtheta.*cov*theta/vari_num;

            dinv_cov_dtheta = ...
                -inv_cov*dcov_dtheta*inv_cov;

            dinv_FTRF_dtheta = -inv_FTRF*...
                (F_reg'*dinv_cov_dtheta*F_reg)*...
                inv_FTRF;

            dmiu_dtheta = dinv_FTRF_dtheta*(F_reg'*inv_cov*Y)+...
                inv_FTRF*(F_reg'*dinv_cov_dtheta*Y);

            dY_Fmiu_dtheta = -F_reg*dmiu_dtheta;

            dsigma2_dtheta = (dY_Fmiu_dtheta'*inv_cov*Y_Fmiu+...
                Y_Fmiu'*dinv_cov_dtheta*Y_Fmiu+...
                Y_Fmiu'*inv_cov*dY_Fmiu_dtheta)/x_num;

            dlnsigma2_dtheta = 1/sigma2*dsigma2_dtheta;

            dlndetR = trace(inv_cov*dcov_dtheta);

            gradient = x_num/2*dlnsigma2_dtheta+0.5*dlndetR;

        end
    end

    function [cov, inv_cov, beta, sigma_sq, inv_FTRF, Y_Fmiu] = interpKriging...
            (X_dis_sq, Y, x_num, vari_num, theta, F_reg)
        % kriging interpolation kernel function
        % Y(x) = beta+Z(x)
        %
        cov = zeros(x_num, x_num);
        for vari_index = 1:vari_num
            cov = cov+X_dis_sq(:, :, vari_index)*theta;
        end
        cov = exp(-cov/vari_num)+eye(x_num)*1e-3;

        % coefficient calculation
        inv_cov = cov\eye(x_num);
        inv_FTRF = (F_reg'*inv_cov*F_reg)\eye(size(F_reg, 2));

        % basical bias
        beta = inv_FTRF*(F_reg'*inv_cov*Y);
        Y_Fmiu = Y-F_reg*beta;
        sigma_sq = (Y_Fmiu'*inv_cov*Y_Fmiu)/x_num;

    end

    function [Y_pred, Var_pred] = interpKrigingPredictor...
            (X_pred, X_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
            x_num, vari_num, theta, beta, gama, sigma_sq, ...
            inv_cov, fval_reg_nomlz, FTRF, reg_function)
        % kriging interpolation predict function
        % input predict_x and kriging model
        % predict_x is row vector
        % output the predict value
        %
        [x_pred_num, ~] = size(X_pred);
        fval_reg_pred = reg_function(X_pred);

        % normalize data
        X_pred_nomlz = (X_pred-aver_X)./stdD_X;
        fval_reg_pred_nomlz = (fval_reg_pred-aver_Y)./stdD_Y;

        % predict covariance
        predict_cov = zeros(x_num, x_pred_num);
        for vari_index = 1:vari_num
            predict_cov = predict_cov+...
                (X_nomlz(:, vari_index)-X_pred_nomlz(:, vari_index)').^2*theta;
        end
        predict_cov = exp(-predict_cov/vari_num);

        % predict base fval

        Y_pred = fval_reg_pred_nomlz*beta+predict_cov'*gama;

        % predict variance
        u__ = fval_reg_nomlz'*inv_cov*predict_cov-fval_reg_pred_nomlz';
        Var_pred = sigma_sq*...
            (1+u__'/FTRF*u__+...
            -predict_cov'*inv_cov*predict_cov);

        % normalize data
        Y_pred = Y_pred*stdD_Y+aver_Y;
        Var_pred = diag(Var_pred)*stdD_Y*stdD_Y;
    end

    function F_reg = regZero(X)
        % zero order base funcion
        %
        F_reg = ones(size(X, 1), 1); % zero
    end

    function F_reg = regLinear(X)
        % first order base funcion
        %
        F_reg = [ones(size(X, 1), 1), X]; % linear
    end
end

function [radialbasis_model_fval, radialbasis_model_con, radialbasis_model_coneq, output] = getRadialBasisModel...
    (x_list, fval_list, con_list, coneq_list)
% base on library_data to create radialbasis model and function
% if input model, function will updata model
% object_function is single fval output
% nonlcon_function is normal nonlcon_function which include con, coneq
% con is colume vector, coneq is colume vector
% var_function is same
%
% basis_function = @(r) r.^3;
basis_function = @(r) r.^3;

[predict_function_fval, radialbasis_model_fval] = interpRadialBasisPreModel...
    (x_list, fval_list, basis_function);

if ~isempty(con_list)
    predict_function_con = cell(size(con_list, 2), 1);
    radialbasis_model_con = struct('X', [], 'Y', [], ...
        'radialbasis_matrix', [], 'beta', [], ...
        'aver_X', [], 'stdD_X', [], 'aver_Y', [], 'stdD_Y', [], 'basis_function', [], ...
        'predict_function', []);
    radialbasis_model_con = repmat(radialbasis_model_con, [size(con_list, 2), 1]);
    for con_index = 1:size(con_list, 2)
        [predict_function_con{con_index}, radialbasis_model_con(con_index)] = interpRadialBasisPreModel...
            (x_list, con_list(:, con_index));
    end
else
    predict_function_con = [];
    radialbasis_model_con = [];
end

if ~isempty(coneq_list)
    predict_function_coneq = cell(size(coneq_list, 2), 1);
    radialbasis_model_coneq = struct('X', [], 'Y', [], ...
        'radialbasis_matrix', [], [], 'beta', [], ...
        'aver_X', [], 'stdD_X', [], 'aver_Y', [], 'stdD_Y', [], 'basis_function', [], ...
        'predict_function', []);
    radialbasis_model_coneq = repmat(radialbasis_model_coneq, [size(coneq_list, 2), 1]);
    for coneq_index = 1:size(coneq_list, 2)
        [predict_function_coneq{coneq_index}, radialbasis_model_con(con_index)] = interpRadialBasisPreModel...
            (x_list, coneq_list(:, coneq_index));
    end
else
    predict_function_coneq = [];
    radialbasis_model_coneq = [];
end

object_function_surrogate = @(X_predict) objectFunctionSurrogate(X_predict, predict_function_fval);
if isempty(radialbasis_model_con) && isempty(radialbasis_model_coneq)
    nonlcon_function_surrogate = [];
else
    nonlcon_function_surrogate = @(X_predict) nonlconFunctionSurrogate(X_predict, predict_function_con, predict_function_coneq);
end

output.object_function_surrogate = object_function_surrogate;
output.nonlcon_function_surrogate = nonlcon_function_surrogate;

    function fval = objectFunctionSurrogate...
            (X_predict, predict_function_fval)
        % connect all predict favl
        %
        fval = predict_function_fval(X_predict);
    end
    function [con, coneq] = nonlconFunctionSurrogate...
            (X_predict, predict_function_con, predict_function_coneq)
        % connect all predict con and coneq
        %
        if isempty(predict_function_con)
            con = [];
        else
            con = zeros(size(X_predict, 1), length(predict_function_con));
            for con_index__ = 1:length(predict_function_con)
                con(:, con_index__) = ....
                    predict_function_con{con_index__}(X_predict);
            end
        end
        if isempty(predict_function_coneq)
            coneq = [];
        else
            coneq = zeros(size(X_predict, 1), length(predict_function_coneq));
            for coneq_index__ = 1:length(predict_function_coneq)
                coneq(:, coneq_index__) = ...
                    predict_function_coneq{coneq_index__}(X_predict);
            end
        end
    end
end

function [predict_function, radialbasis_model] = interpRadialBasisPreModel...
    (X, Y, basis_function)
% radial basis function interp pre model function
% input initial data X, Y, which are real data
% X, Y are x_number x variable_number matrix
% aver_X, stdD_X is 1 x x_number matrix
% output is a radial basis model, include X, Y, base_function
% and predict_function
%
% Copyright 2023 Adel
%
if nargin < 3
    basis_function = [];
end

[x_number, variable_number] = size(X);

% normalize data
aver_X = mean(X);
stdD_X = std(X);
aver_Y = mean(Y);
stdD_Y = std(Y);
index__ = find(stdD_X == 0);
if ~isempty(index__), stdD_X(index__) = 1;end
index__ = find(stdD_Y == 0);
if ~isempty(index__), stdD_Y(index__) = 1;end
X_nomlz = (X-aver_X)./stdD_X;
Y_nomlz = (Y-aver_Y)./stdD_Y;

if isempty(basis_function)
    c = (prod(max(X_nomlz)-min(X_nomlz))/x_number)^(1/variable_number);
    basis_function = @(r) exp(-(r.^2)/c);
end

% initialization distance of all X
X_dis = zeros(x_number, x_number);
for variable_index = 1:variable_number
    X_dis = X_dis+(X_nomlz(:, variable_index)-X_nomlz(:, variable_index)').^2;
end
X_dis = sqrt(X_dis);

[beta, rdibas_matrix] = interpRadialBasis...
    (X_dis, Y_nomlz, basis_function, x_number);

% initialization predict function
predict_function = @(X_predict) interpRadialBasisPredictor...
    (X_predict, X_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
    x_number, variable_number, beta, basis_function);

radialbasis_model.X = X;
radialbasis_model.Y = Y;
radialbasis_model.radialbasis_matrix = rdibas_matrix;
radialbasis_model.beta = beta;

radialbasis_model.aver_X = aver_X;
radialbasis_model.stdD_X = stdD_X;
radialbasis_model.aver_Y = aver_Y;
radialbasis_model.stdD_Y = stdD_Y;
radialbasis_model.basis_function = basis_function;

radialbasis_model.predict_function = predict_function;

% abbreviation:
% num: number, pred: predict, vari: variable
    function [beta, rdibas_matrix] = interpRadialBasis...
            (X_dis, Y, basis_function, x_number)
        % interp polynomial responed surface core function
        % calculation beta
        %
        % Copyright 2022 Adel
        %
        rdibas_matrix = basis_function(X_dis);
        
        % stabilize matrix
        rdibas_matrix = rdibas_matrix+eye(x_number)*1e-6;
        
        % solve beta
        beta = rdibas_matrix\Y;
    end

    function [Y_pred] = interpRadialBasisPredictor...
            (X_pred, X_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
            x_num, vari_num, beta, basis_function)
        % radial basis function interpolation predict function
        %
        [x_pred_num, ~] = size(X_pred);

        % normalize data
        X_pred_nomlz = (X_pred-aver_X)./stdD_X;
        
        % calculate distance
        X_dis_pred = zeros(x_pred_num, x_num);
        for vari_index = 1:vari_num
            X_dis_pred = X_dis_pred+...
                (X_pred_nomlz(:, vari_index)-X_nomlz(:, vari_index)').^2;
        end
        X_dis_pred = sqrt(X_dis_pred);
        
        % predict variance
        Y_pred = basis_function(X_dis_pred)*beta;
        
        % normalize data
        Y_pred = Y_pred*stdD_Y + aver_Y;
    end

end

%% data library
function [fval_list, con_list, coneq_list] = dataLibraryUpdata...
    (data_library_name, model_function, x_list)
% updata data library
% updata format:
% variable_number, fval_number, con_number, coneq_number
% x, fval, con, coneq
%
fval_list = [];
con_list = [];
coneq_list = [];
[x_number, variable_number] = size(x_list);

if ~strcmp(data_library_name(end-3:end), '.txt')
    data_library_name = [data_library_name, '.txt'];
end

% store format
x_format_base = '%.8e ';
fval_format_base = '%.8e ';
x_format = repmat(x_format_base, 1, variable_number);
file_optimalSurrogate_output = fopen(data_library_name, 'a');
file_result = fopen('result_total.txt', 'a');

% updata format:
% variable_number, fval_number, con_number, coneq_number
% x, fval, con, coneq
for x_index = 1:x_number
    x = x_list(x_index, :);
    [fval, con, coneq] = model_function(x);
    fval = fval(:);
    con = con(:);
    coneq = coneq(:);
    fval_list = [fval_list;fval(:)'];
    con_list = [con_list;con(:)'];
    coneq_list = [coneq_list;coneq(:)'];

    % write data to txt_optimalSurrogateSADEKTS
    fprintf(file_optimalSurrogate_output, '%d ', variable_number);
    fprintf(file_optimalSurrogate_output, '%d ', length(fval));
    fprintf(file_optimalSurrogate_output, '%d ', length(con));
    fprintf(file_optimalSurrogate_output, '%d ', length(coneq));

    fprintf(file_optimalSurrogate_output, x_format, x);
    fval_format = repmat(fval_format_base, 1, length(fval));
    fprintf(file_optimalSurrogate_output, fval_format, fval);
    fval_format = repmat(fval_format_base, 1, length(con));
    fprintf(file_optimalSurrogate_output, fval_format, con);
    fval_format = repmat(fval_format_base, 1, length(coneq));
    fprintf(file_optimalSurrogate_output, fval_format, coneq);
    fprintf(file_optimalSurrogate_output, '\n');

    % write data to txt_result
    fprintf(file_result, '%d ', variable_number);
    fprintf(file_result, '%d ', length(fval));
    fprintf(file_result, '%d ', length(con));
    fprintf(file_result, '%d ', length(coneq));

    fprintf(file_result, x_format, x);
    fval_format = repmat(fval_format_base, 1, length(fval));
    fprintf(file_result, fval_format, fval);
    fval_format = repmat(fval_format_base, 1, length(con));
    fprintf(file_result, fval_format, con);
    fval_format = repmat(fval_format_base, 1, length(coneq));
    fprintf(file_result, fval_format, coneq);
    fprintf(file_result, '\n');
end

fclose(file_optimalSurrogate_output);
clear('file_optimalSurrogate_output');
fclose(file_result);
clear('file_result');
end

function [x_list, fval_list, con_list, coneq_list] = dataLibraryLoad...
    (data_library_name, low_bou, up_bou)
% load data from data library
% low_bou, up_bou is range of data
% updata format:
% variable_number, fval_number, con_number, coneq_number
% x, fval, con, coneq
%
if nargin < 3
    up_bou = inf;
    if nargin < 2
        low_bou = -inf;
        if nargin < 1
            error('dataLibraryLoad: lack data_library_name');
        end
    end
end

if ~strcmp(data_library_name(end-3:end), '.txt')
    data_library_name = [data_library_name, '.txt'];
end

% updata format:
% variable_number, fval_number, con_number, coneq_number
% x, fval, con, coneq
if exist(data_library_name, 'file') == 2
    data_list = importdata(data_library_name);
    if ~isempty(data_list)
        % search whether exist point
        x_list = [];
        fval_list = [];
        con_list = [];
        coneq_list = [];

        for data_index = 1:size(data_list, 1)
            data = data_list(data_index, :);

            variable_number = data(1);
            fval_number = data(2);
            con_number = data(3);
            coneq_number = data(4);

            base = 5;
            x = data(base:base+variable_number-1);
            judge = sum(x < low_bou)+sum(x > up_bou);
            if ~judge
                x_list = [x_list;x];
                base = base+variable_number;
                fval_list = [fval_list;data(base:base+fval_number-1)];
                base = base+fval_number;
                con = data(base:base+con_number-1);
                if ~isempty(con)
                    con_list = [con_list;con];
                end
                base = base+con_number;
                coneq = data(base:base+coneq_number-1);
                if ~isempty(coneq)
                    coneq_list = [coneq_list;coneq];
                end
            end
        end
    else
        x_list = [];
        fval_list = [];
        con_list = [];
        coneq_list = [];
    end
else
    x_list = [];
    fval_list = [];
    con_list = [];
    coneq_list = [];
end
end

%% LHD
function [X, X_new, distance_min_nomlz] = getLatinHypercube...
    (sample_number, variable_number, X_exist, ...
    low_bou, up_bou, cheapcon_function)
% generate sample sequence latin hypercube
% more uniform point distribution by simulating particle motion
% sample number is total point in area
% default low_bou is 0, up_bou is 1, cheapcon_function is []
% low_bou and up_bou is colume vector
% x in x_exist_list, x_list, supply_x_list is row vector
% x_exist_list should meet bou
%
% Copyright 2022 Adel
%
if nargin < 6
    cheapcon_function = [];
    if nargin < 5
        if nargin < 3
            X_exist = [];
            if nargin < 2
                error('getLatinHypercube: lack variable_number');
            end
        end
        low_bou = zeros(1, variable_number);
        up_bou = ones(1, variable_number);
    end
end
iteration_max = 100;

% check x_exist_list if meet boundary
if ~isempty(X_exist)
    index = find(X_exist < low_bou);
    index = [index, find(X_exist > up_bou)];
    if ~isempty(index)
        error('getLatinHypercube: x_exist_list range error');
    end
    if size(X_exist, 2) ~= variable_number
        error('getLatinHypercube: x_exist_list variable_number error');
    end
    X_exist_nomlz = (X_exist-low_bou)./(up_bou-low_bou);
else
    X_exist_nomlz = [];
end

% check input
if sample_number < 0
    X = [];
    X_new = [];
    distance_min_nomlz = [];
    return;
end

% check x_new_number
x_new_number = sample_number-size(X_exist, 1);
if x_new_number < 0
    X = X_exist;
    X_new = [];
    distance_min_nomlz = getMinDistance(X_exist_nomlz);
    return;
end

low_bou_nomlz = zeros(1, variable_number);
up_bou_nomlz = ones(1, variable_number);

% get initial X_new_nomalize by lhsdesign
X_new_nomlz = rand(x_new_number, variable_number);
distance_min_nomlz = getMinDistance([X_new_nomlz;X_exist_nomlz]);

% x is nomalize, so constraint function should change
if ~isempty(cheapcon_function)
    cheapcon_function = @(x) ...
        max(max(sample_number*cheapcon_function(x.*(up_bou-low_bou)+low_bou)+1, 0), [], 1);
end

iteration = 0;
fval_list = zeros(x_new_number, 1);
gradient_list = zeros(x_new_number, variable_number);
while iteration < iteration_max
    % change each x place by newton methods
    for x_index = 1:x_new_number
        % get gradient
        [fval_list(x_index, 1), gradient_list(x_index, :)] = objectFunctionXPlace...
            (X_new_nomlz(x_index, :), [X_new_nomlz(1:x_index-1, :);X_new_nomlz(x_index+1:end, :);X_exist_nomlz], ...
            sample_number, variable_number, low_bou_nomlz-0.1/variable_number, up_bou_nomlz+0.1/variable_number, cheapcon_function);
    end

    % normalize fval
    fval_list = fval_list/max(fval_list);
    for x_index = 1:x_new_number
        C = fval_list(x_index, 1)*distance_min_nomlz*(1-iteration/iteration_max);
        x = X_new_nomlz(x_index, :)+...
            -gradient_list(x_index, :)/...
            norm(gradient_list(x_index, :))*C;
        x = min(x, up_bou_nomlz);
        x = max(x, low_bou_nomlz);
        X_new_nomlz(x_index, :) = x;
    end

    iteration = iteration+1;
end
distance_min_nomlz = getMinDistance([X_new_nomlz;X_exist_nomlz]);
X_new = X_new_nomlz.*(up_bou-low_bou)+low_bou;
X = [X_new;X_exist];

    function [fval, gradient] = objectFunctionXPlace...
            (x, X_surplus, sample_number, variable_number, low_bou, up_bou, cheapcon_function)
        % function describe distance between X and X_supply
        % x is colume vector and X_surplus is matrix which is num-1 x var
        % low_bou_limit__ and up_bou_limit__ is colume vector
        % variable in colume
        %
        a__ = 10/variable_number;
        a_bou__ = 30/sample_number;

        sign__ = ((x > X_surplus)-0.5)*2;

        xi__ = -a__*(x-X_surplus).*sign__;
        sum_xi__ = sum(xi__, 2);
        psi__ = a__*(low_bou-x)*a_bou__;
        zeta__ = a__*(x-up_bou)*a_bou__;

        %         exp_xi__ = exp(xi__);
        exp_sum_xi__ = exp(sum_xi__);
        exp_psi__ = exp(psi__);
        exp_zeta__ = exp(zeta__);

        % get fval
        fval = sum(exp_sum_xi__, 1)+...
            sum(exp_psi__+exp_zeta__, 2);

        % get gradient
        gradient = sum(-a__*sign__.*exp_sum_xi__, 1)+...
            -a__*exp_psi__*a_bou__+...
            a__*exp_zeta__*a_bou__;

        if ~isempty(cheapcon_function)
            fval_con = cheapcon_function(x);
            fval = fval+fval_con;
            [gradient_con] = differ...
                (cheapcon_function, x, fval_con, variable_number);
            gradient = gradient+gradient_con;
        end

        function [gradient] = differ(differ_function, x, fval, variable_number, step)
            % differ function to get gradient and hessian
            % gradient is rank vector
            %
            if nargin < 5
                step = 1e-6;
            end
            fval__ = zeros(variable_number, 2); % backward is 1, forward is 2
            gradient = zeros(1, variable_number);

            % fval and gradient
            for variable_index__ = 1:variable_number
                x_forward__ = x;
                x_backward__ = x;
                x_backward__(variable_index__) = x_backward__(variable_index__)-step;
                fval__(variable_index__, 1) = differ_function(x_backward__);

                x_forward__(variable_index__) = x_forward__(variable_index__)+step;
                fval__(variable_index__, 2) = differ_function(x_forward__);

                gradient(variable_index__) = ...
                    (fval__(variable_index__, 2)-fval__(variable_index__, 1))/2/step;
            end
        end
    end
    function distance_min__ = getMinDistance(x_list__)
        % get distance min from x_list
        %

        % sort x_supply_list_initial to decrese distance calculate times
        x_list__ = sortrows(x_list__, 1);
        sample_number__ = size(x_list__, 1);
        variable_number__ = size(x_list__, 2);
        distance_min__ = variable_number__;
        for x_index__ = 1:sample_number__
            x_curr__ = x_list__(x_index__, :);
            x_next_index__ = x_index__ + 1;
            % first dimension only search in min_distance
            search_range__ = variable_number__;
            while x_next_index__ <=  sample_number__ &&...
                    (x_list__(x_next_index__, 1)-x_list__(x_index__, 1))^2 ...
                    < search_range__
                x_next__ = x_list__(x_next_index__, :);
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
