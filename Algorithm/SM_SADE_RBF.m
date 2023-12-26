clc;
clear;
close all hidden;

benchmark = BenchmarkFunction();

benchmark_type = 'single';
% benchmark_name = 'GP';
% benchmark_name = 'Wei';
% benchmark_name = 'PK';
% benchmark_name = 'EP20';
% benchmark_name = 'Forrester';
% benchmark_name = 'PVD4';
benchmark_name = 'G01';
% benchmark_name = 'G06';

[model_function,variable_number,low_bou,up_bou,...
    object_function,A,B,Aeq,Beq,nonlcon_function] = benchmark.getBenchmark(benchmark_type,benchmark_name);

% x_initial = rand(1,variable_number).*(up_bou-low_bou)+low_bou;
% [x_best,obj_best,~,output] = fmincon(object_function,x_initial,A,B,Aeq,Beq,low_bou,up_bou,[],optimoptions('fmincon','Algorithm','sqp','MaxFunctionEvaluations',10000,'Display','iter-detailed'))

cheapcon_function = [];

%% single run

% [x_best,obj_best,NFE,output] = optimalSurrogateSADERBF...
%     (model_function,variable_number,low_bou,up_bou,...
%     cheapcon_function,200,500)
% result_x_best = output.result_x_best;
% result_obj_best = output.result_obj_best;
% 
% figure(1);
% plot(result_obj_best);

%% repeat run

repeat_number = 10;
result_obj = zeros(repeat_number,1);
result_NFE = zeros(repeat_number,1);
max_NFE = 300;
for repeat_index = 1:repeat_number
    [x_best,obj_best,NFE,output] = optimalSurrogateSADERBF...
        (model_function,variable_number,low_bou,up_bou,...
        cheapcon_function,max_NFE,300,1e-6,0);

    result_obj(repeat_index) = obj_best;
    result_NFE(repeat_index) = NFE;
end

fprintf('Fval     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_obj),mean(result_obj),max(result_obj),std(result_obj));
fprintf('NFE     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_NFE),mean(result_NFE),max(result_NFE),std(result_NFE));
save([benchmark_name,'_SADE_RBF_',num2str(200),'.mat']);

%% main
function [x_best,obj_best,NFE,output] = optimalSurrogateSADERBF...
    (model_function,variable_number,low_bou,up_bou,...
    cheapcon_function,....
    NFE_max,iteration_max,torlance,nonlcon_torlance,x_initial_list)
% RBF-CDE optimization algorithm
% use RBF model instead of origin kriging model
%
% referance: [1] 叶年辉,龙腾,武宇飞,et al.
% 基于Kriging代理模型的约束差分进化算法 [J]. 航空学报,2021,42(6): 13.
%
% Copyright 2022 Adel
%
if nargin < 10
    x_initial_list = [];
    if nargin < 9 || isempty(nonlcon_torlance)
        nonlcon_torlance = 1e-3;
        if nargin < 8 || isempty(torlance)
            torlance = 1e-3;
            if nargin < 7
                iteration_max = [];
                if nargin < 6
                    NFE_max = [];
                end
            end
        end
    end
end

if nargin < 5
    cheapcon_function = [];
end

DRAW_FIGURE_FLAG = 0; % whether draw data
INFORMATION_FLAG = 1; % whether print data
CONVERGENCE_JUDGMENT_FLAG = 0; % whether judgment convergence
WRIRE_FILE_FLAG = 0; % whether write to file

% hyper parameter
population_number = min(100,10*variable_number);
RBF_number = max(100,(variable_number+1)*(variable_number+2)/2);
scaling_factor = 0.8; % F
cross_rate = 0.8;

% max obj when normalize obj,con,coneq
nomlz_obj = 10;

% surrogate add point protect range
protect_range = 1e-8;

% NFE and iteration setting
if isempty(NFE_max)
    NFE_max = 10+10*variable_number;
end

if isempty(iteration_max)
    iteration_max = 20+20*variable_number;
end

done = 0;NFE = 0;iteration = 0;

% step 2
% generate initial sample X
if isempty(x_initial_list)
    %     [~,x_updata_list,~] = getLatinHypercube...
    %         (population_number,variable_number,[],low_bou,up_bou,cheapcon_function);
    X_updata = lhsdesign(population_number,variable_number).*(up_bou-low_bou)+low_bou;
else
    X_updata = x_initial_list;
end

% detech expensive constraints and initializa data library
data_library = DataLibrary(model_function,variable_number,low_bou,up_bou,...
    nonlcon_torlance,[],WRIRE_FILE_FLAG);

% detech expensive constraints and initialize data library
[~,~,~,~,~,~,~,NFE_updata]=data_library.dataUpdata(X_updata(1,:),0);
NFE = NFE+NFE_updata;
if ~isempty(data_library.vio_list)
    expensive_nonlcon_flag = 1;
else
    expensive_nonlcon_flag = 0;
end

% updata data library by X
[~,~,~,~,~,~,~,NFE_updata]=data_library.dataUpdata(X_updata(2:end,:),0);
NFE = NFE+NFE_updata;

data_library=copy(data_library);
% find fesiable data in current data library
if expensive_nonlcon_flag
    feasi_boolean_list = data_library.vio_list == 0;
end

result_x_best = zeros(iteration_max,variable_number);
result_obj_best = zeros(iteration_max,1);

iteration = iteration+1;

next_search_mode = 'G'; % 'G' is global search,'l' is local search
while ~done
    search_mode = next_search_mode;
    % nomalization all data by max obj and to create surrogate model
    [X,Fval,Con,Coneq,Vio,Ks]=data_library.dataLoad;

    X_model = X;
    obj_max = max(abs(Fval),[],1);
    Fval_model=Fval/obj_max*nomlz_obj;
    if ~isempty(Con)
        con_maX = max(abs(Con),[],1);
        Con_model = Con./con_maX*nomlz_obj;
    else
        Con_model = [];
    end
    if ~isempty(Coneq)
        coneq_maX = max(abs(Coneq),[],1);
        Coneq_model = Coneq./coneq_maX*nomlz_obj;
    else
        Coneq_model = [];
    end
    if ~isempty(Ks)
        ks_maX = max(abs(Ks),[],1);
        Ks_model = Ks./ks_maX*nomlz_obj;
    else
        Ks_model = [];
    end

    if search_mode == 'G'
        % global search
        [x_infill,...
            RBF_model_obj,RBF_model_con,RBF_model_coneq] = searchGlobal...
            (X_model,Fval_model,Con_model,Coneq_model,...
            variable_number,low_bou,up_bou,cheapcon_function,nonlcon_torlance,...
            population_number,scaling_factor,cross_rate,...
            expensive_nonlcon_flag);
    else
        % local search
        [x_infill,...
            RBF_model_obj,RBF_model_con,RBF_model_coneq] = searchLocal...
            (X_model,Fval_model,Con_model,Coneq_model,...
            variable_number,low_bou,up_bou,cheapcon_function,nonlcon_torlance,...
            population_number,RBF_number,...
            expensive_nonlcon_flag);
    end

    % updata infill point
    [x_infill,obj_infill,con_infill,coneq_infill,vio_infill,ks_infill,repeat_index,NFE_updata] = ...
        data_library.dataUpdata(x_infill,protect_range);
    NFE = NFE+NFE_updata;

    % process error
    if isempty(x_infill)
        % continue;
        x_infill = X(repeat_index,:);
        obj_infill = Fval(repeat_index,:);
        if ~isempty(Con)
            con_infill = Con(repeat_index,:);
        end
        if ~isempty(Coneq)
            coneq_infill = Coneq(repeat_index,:);
        end
        if ~isempty(Vio)
            vio_local_infill = Vio(repeat_index,:);
        end
    else
        if expensive_nonlcon_flag
            feasi_boolean_list = [feasi_boolean_list;vio_infill == 0];
        end
    end

    improve_flag = false(1);
    if expensive_nonlcon_flag
        min_vio = min(Vio);
        min_obj = min(Fval([feasi_boolean_list(1:end-1);false(1)]),[],1);

        % if all point is infeasible,violation of point infilled is
        % less than min violation of all point means improve.if
        % feasible point exist,obj of point infilled is less than min
        % obj means improve
        if vio_infill < min_vio
            if ~isempty(min_obj)
                if obj_infill < min_obj
                    % improve, continue local search
                    improve_flag = true(1);
                end
            else
                % improve, continue local search
                improve_flag = true(1);
            end
        end
    else
        min_obj = min(Fval(1:end-1));

        % obj of point infilled is less than min obj means improve
        if obj_infill < min_obj
            % imporve, continue local search
            improve_flag = true(1);
        end
    end

    % if no imporve begin local search or global search
    if ~improve_flag
        next_search_mode = flipSearchMode(next_search_mode);
    end

    if DRAW_FIGURE_FLAG && variable_number < 3
        interpVisualize(RBF_model_obj,low_bou,up_bou);
        line(x_infill(1),x_infill(2),obj_infill./obj_max*nomlz_obj,'Marker','o','color','r');
    end

    % find best result to record
    [x_best,obj_best,con_best,coneq_best] = findMinRaw...
        (X,Fval,Con,Coneq,...
        cheapcon_function,nonlcon_torlance);
    vio_best = calViolation(con_best,coneq_best,nonlcon_torlance);

    if INFORMATION_FLAG
        fprintf('model: %s obj:    %f    violation:    %f    NFE:    %-3d\n',search_mode,obj_best,vio_best,NFE);
        %         fprintf('iteration:          %-3d    NFE:    %-3d\n',iteration,NFE);
        %         fprintf('x:          %s\n',num2str(x_infill));
        %         fprintf('value:      %f\n',obj_infill);
        %         fprintf('violation:  %s  %s\n',num2str(con_infill),num2str(coneq_infill));
        %         fprintf('\n');
    end

    result_x_best(iteration,:) = x_best;
    result_obj_best(iteration,:) = obj_best;
    iteration = iteration+1;

    % forced interrupt
    if iteration > iteration_max || NFE >= NFE_max
        done = 1;
    end

    % convergence judgment
    if CONVERGENCE_JUDGMENT_FLAG
        if (iteration > 2 && ...
                abs((obj_best-obj_best_old)/obj_best_old) < torlance)
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

    obj_best_old = obj_best;
end

result_x_best = result_x_best(1:iteration-1,:);
result_obj_best = result_obj_best(1:iteration-1);

output.result_x_best = result_x_best;
output.result_obj_best = result_obj_best;
output.data_library = data_library;

    function next_search_flag=flipSearchMode(next_search_flag)
        switch next_search_flag
            case 'L'
                next_search_flag = 'G';
            case 'G'
                next_search_flag = 'L';
        end
    end
end

%% auxiliary function
function vio_list = calViolation(con_list,coneq_list,nonlcon_torlance)
% calculate violation of data
%
if isempty(con_list) && isempty(coneq_list)
    vio_list = [];
else
    vio_list = zeros(max(size(con_list,1),size(coneq_list,1)),1);
    if ~isempty(con_list)
        vio_list = vio_list+sum(max(con_list-nonlcon_torlance,0),2);
    end
    if ~isempty(coneq_list)
        vio_list = vio_list+sum((abs(coneq_list)-nonlcon_torlance),2);
    end

end
end

function [x_best,obj_best,con_best,coneq_best] = findMinRaw...
    (x_list,obj_list,con_list,coneq_list,...
    cheapcon_function,nonlcon_torlance)
% find min obj in raw data
% x_list,rank is variable
% con_list,rank is con
% coneq_list,rank is coneq
% function will find min obj in con == 0
% if there was not feasible x,find min consum
%
con_best = [];
coneq_best = [];
max_nonlcon_list = zeros(size(x_list,1),1);
max_cheapcon_list = zeros(size(x_list,1),1);
% process expendsive con
if ~isempty(con_list)
    max_nonlcon_list = max(con_list,[],2);
end
if ~isempty(coneq_list)
    max_nonlcon_list = max(abs(coneq_list),[],2);
end

% add cheap con
for x_index = 1:size(x_list,1)
    if ~isempty(cheapcon_function)
        [con,coneq] = cheapcon_function(x_list(x_index,:));
        max_cheapcon_list(x_index) = max_cheapcon_list(x_index)+...
            sum(max(con,0))+sum(coneq.*coneq);
    end
end

con_judge_list = (max_nonlcon_list > nonlcon_torlance)+...
    (max_cheapcon_list > 0);
index = find(con_judge_list == 0);
if ~isempty(index)
    % feasible x
    x_list = x_list(index,:);
    obj_list = obj_list(index);
    if ~isempty(con_list)
        con_list = con_list(index,:);
    end
    if ~isempty(coneq_list)
        coneq_list = coneq_list(index,:);
    end

    % min obj
    [obj_best,index_best] = min(obj_list);
    x_best = x_list(index_best,:);
    if ~isempty(con_list)
        con_best = con_list(index_best,:);
    end
    if ~isempty(coneq_list)
        coneq_best = coneq_list(index_best,:);
    end
else
    % min consum
    [~,index_best] = min(max_nonlcon_list);
    obj_best = obj_list(index_best);
    x_best = x_list(index_best,:);
    if ~isempty(con_list)
        con_best = con_list(index_best,:);
    end
    if ~isempty(coneq_list)
        coneq_best = coneq_list(index_best,:);
    end
end
end

function [x_list,obj_list,con_list,coneq_list,vio_list] = rankData...
    (x_list,obj_list,con_list,coneq_list,...
    cheapcon_function,nonlcon_torlance)
% rank data base on feasibility rule
% infeasible is rank by sum of constraint
% torlance to cheapcon_function is 0
%
if nargin < 6 || isempty(nonlcon_torlance)
    nonlcon_torlance = 0;
end
if nargin < 5
    cheapcon_function = [];
end

[x_number,~] = size(x_list);
vio_list = zeros(x_number,1);
if ~isempty(con_list)
    vio_list = vio_list+sum(max(con_list-nonlcon_torlance,0),2);
end
if ~isempty(coneq_list)
    vio_list = vio_list+sum((abs(coneq_list)-nonlcon_torlance),2);
end

% add cheap con
for x_index = 1:size(x_list,1)
    if ~isempty(cheapcon_function)
        [con,coneq] = cheapcon_function(x_list(x_index,:));
        vio_list(x_index) = vio_list(x_index)+...
            sum(max(con,0))+sum(max(abs(coneq),0));
    end
end

% rank data
% infeasible data rank by violation, feasible data rank by obj
feasi_boolean_list = vio_list <= 0;
all = 1:x_number;
feasi_index_list = all(feasi_boolean_list);
infeasi_index_list = all(~feasi_boolean_list);
[~,index_list] = sort(obj_list(feasi_index_list));
feasi_index_list = feasi_index_list(index_list);
[~,index_list] = sort(vio_list(infeasi_index_list));
infeasi_index_list = infeasi_index_list(index_list);
index_list = [feasi_index_list,infeasi_index_list];

% rank by index_list
x_list = x_list(index_list,:);
obj_list = obj_list(index_list);
if ~isempty(con_list)
    con_list = con_list(index_list,:);
end
if ~isempty(coneq_list)
    coneq_list = coneq_list(index_list,:);
end
vio_list = vio_list(index_list);

end

function [x_global_infill,...
    RBF_model_obj,RBF_model_con,RBF_model_coneq] = searchGlobal...
    (x_list,obj_nomlz_list,con_nomlz_list,coneq_nomlz_list,...
    variable_number,low_bou,up_bou,cheapcon_function,nonlcon_torlance,...
    population_number,scaling_factor,cross_rate,...
    expensive_nonlcon_flag)
% find global infill point function
%

% step 5
% rank x_list data
[x_rank_list,~,~,~] = rankData...
    (x_list,obj_nomlz_list,con_nomlz_list,coneq_nomlz_list,...
    cheapcon_function,nonlcon_torlance);

% step 6
% only the first population_number will be use
x_best_popu_list = x_rank_list(1:population_number,:);

% differ evolution mutations
X_new_R1 = differEvolutionRand...
    (low_bou,up_bou,x_best_popu_list,scaling_factor,population_number,1);
X_new_R2 = differEvolutionRand...
    (low_bou,up_bou,x_best_popu_list,scaling_factor,population_number,2);
X_new_CR = differEvolutionCurrentRand...
    (low_bou,up_bou,x_best_popu_list,scaling_factor);
X_new_CB = differEvolutionCurrentBest...
    (low_bou,up_bou,x_best_popu_list,scaling_factor,1);

% differ evolution crossover
X_new_R1 = differEvolutionCrossover...
    (low_bou,up_bou,x_best_popu_list,X_new_R1,cross_rate);
X_new_R2 = differEvolutionCrossover...
    (low_bou,up_bou,x_best_popu_list,X_new_R2,cross_rate);
X_new_CR = differEvolutionCrossover...
    (low_bou,up_bou,x_best_popu_list,X_new_CR,cross_rate);
X_new_CB = differEvolutionCrossover...
    (low_bou,up_bou,x_best_popu_list,X_new_CB,cross_rate);

% find global infill point base kriging model from offspring X
x_DE_list = [X_new_R1;X_new_R2;X_new_CR;X_new_CB];

% step 4
% construct RBF model
% base on distance to x_list to repace predict variance
[RBF_model_obj,RBF_model_con,RBF_model_coneq,output_RBF] = getRadialBasisModel...
    (x_list,obj_nomlz_list,con_nomlz_list,coneq_nomlz_list);
object_function_surrogate = output_RBF.object_function_surrogate;
nonlcon_function_surrogate = output_RBF.nonlcon_function_surrogate;

%         obj_list = meritFunction...
%             (x_trial_list,object_function_surrogate,x_list,variable_number,0.5,0.5);

% evaluate each x_offspring obj and constraints
[obj_pred_DE_list] = object_function_surrogate(x_DE_list);

% variance were replaced by x distance to exist x
% distance scale
dis = zeros(size(x_DE_list,1),size(x_list,1));
for vari_index = 1:variable_number
    dis = dis+(x_DE_list(:,vari_index)-x_list(:,vari_index)').^2;
end
D = min(sqrt(dis),[],2);
D_min = min(D); D_max = max(D);
obj_var_DE_list = (D_max-D)./(D_max-D_min);
if expensive_nonlcon_flag

    if ~isempty(nonlcon_function_surrogate)
        [con_pred_DE_list,coneq_pred_DE_list] = nonlcon_function_surrogate(x_DE_list);
        con_var_DE_list = obj_var_DE_list;
        coneq_var_DE_list = obj_var_DE_list;
    end

    vio_DE_list = zeros(4*population_number,1);
    if ~isempty(con_nomlz_list)
        vio_DE_list = vio_DE_list+sum(max(con_pred_DE_list-nonlcon_torlance,0),2);
    end
    if ~isempty(coneq_nomlz_list)
        vio_DE_list = vio_DE_list+sum((abs(con_pred_DE_list)-nonlcon_torlance),2);
    end

    feasi_boolean_DE_list = vio_DE_list <= nonlcon_torlance;
else
    feasi_boolean_DE_list = true(ones(1,4*population_number));
end

% if have feasiable_index_list,only use feasiable to choose
if all(~feasi_boolean_DE_list)
    % base on constaints improve select global infill
    % lack process of equal constraints
    con_nomlz_base = max(min(con_nomlz_list,[],1),0);
    con_impove_probability_list = sum(...
        normcdf((con_nomlz_base-con_pred_DE_list)./sqrt(con_var_DE_list)),2);
    [~,con_best_index] = max(con_impove_probability_list);
    con_best_index = con_best_index(1);
    x_global_infill = x_DE_list(con_best_index,:);
else
    % base on fitness DE point to select global infill
    if expensive_nonlcon_flag
        x_DE_list = x_DE_list(feasi_boolean_DE_list,:);
        obj_pred_DE_list = obj_pred_DE_list(feasi_boolean_DE_list);
        obj_var_DE_list = obj_var_DE_list(feasi_boolean_DE_list);
    end

    obj_DE_min = min(obj_pred_DE_list,[],1);
    obj_DE_max = max(obj_pred_DE_list,[],1);

    DE_fitness_list = -(obj_pred_DE_list-obj_DE_min)/(obj_DE_max-obj_DE_min)+...
        obj_var_DE_list;
    [~,fitness_best_index] = max(DE_fitness_list);
    fitness_best_index = fitness_best_index(1);
    x_global_infill = x_DE_list(fitness_best_index,:);
end
    function X_new = differEvolutionRand(low_bou,up_bou,X,F,x_number,rand_number)
        if nargin < 4
            rand_number = 1;
            if nargin < 3
                x_number = 1;
                if nargin < 2
                    error('differEvolutionRand: lack scaling factor F');
                end
            end
        end
        [x_number__,variable_number__] = size(X);
        X_new = zeros(x_number,variable_number__);
        for x_index__ = 1:x_number
            index__ = randi(x_number__,2*rand_number+1,1);
            X_new(x_index__,:) = X(index__(1),:);
            for rand_index__ = 1:rand_number
                X_new(x_index__,:) = X_new(x_index__,:)+...
                    F*(X(index__(2*rand_index__),:)-X(index__(2*rand_index__+1),:));
                X_new(x_index__,:) = max(X_new(x_index__,:),low_bou);
                X_new(x_index__,:) = min(X_new(x_index__,:),up_bou);
            end
        end
    end
    function X_new = differEvolutionCurrentRand(low_bou,up_bou,X,F)
        [x_number__,variable_number__] = size(X);
        X_new = zeros(x_number__,variable_number__);
        for x_index__ = 1:x_number__
            index__ = randi(x_number__,3,1);
            X_new(x_index__,:) = X(x_index__,:)+...
                F*(X(index__(1),:)-X(x_index__,:)+...
                X(index__(2),:)-X(index__(3),:));
            X_new(x_index__,:) = max(X_new(x_index__,:),low_bou);
            X_new(x_index__,:) = min(X_new(x_index__,:),up_bou);
        end
    end
    function X_new = differEvolutionCurrentBest(low_bou,up_bou,X,F,x_best_index)
        [x_number__,variable_number__] = size(X);
        X_new = zeros(x_number__,variable_number__);
        for x_index__ = 1:x_number__
            index__ = randi(x_number__,2,1);
            X_new(x_index__,:) = X(x_index__,:)+...
                F*(X(x_best_index,:)-X(x_index__,:)+...
                X(index__(1),:)-X(index__(2),:));
            X_new(x_index__,:) = max(X_new(x_index__,:),low_bou);
            X_new(x_index__,:) = min(X_new(x_index__,:),up_bou);
        end
    end
    function X_new = differEvolutionCrossover(low_bou,up_bou,X,V,C_R)
        if size(X,1) ~= size(V,1)
            error('differEvolutionOffspring: size incorrect');
        end
        [x_number__,variable_number__] = size(X);
        X_new = X;
        rand_number = rand(x_number__,variable_number__);
        index__ = find(rand_number < C_R);
        X_new(index__) = V(index__);
        for x_index__ = 1:x_number__
            X_new(x_index__,:) = max(X_new(x_index__,:),low_bou);
            X_new(x_index__,:) = min(X_new(x_index__,:),up_bou);
        end
    end

end

function [x_local_infill,...
    RBF_model_obj,RBF_model_con,RBF_model_coneq] = searchLocal...
    (x_list,obj_nomlz_list,con_nomlz_list,coneq_nomlz_list,...
    variable_number,low_bou,up_bou,cheapcon_function,nonlcon_torlance,...
    population_number,RBF_number,...
    expensive_nonlcon_flag)
% find local infill point function
%

[x_list,obj_nomlz_list,con_nomlz_list,coneq_nomlz_list,vio_nomlz_list] = rankData...
    (x_list,obj_nomlz_list,con_nomlz_list,coneq_nomlz_list,...
    cheapcon_function,nonlcon_torlance);

% step 8
% rand select initial local point from x_list
x_index = randi(population_number);
%         x_index = find(vio_nomlz_list == 0);
%         x_index = x_index(end)+1;
x_initial = x_list(x_index,:);

%         % rank x_list data and select best point as initial local point
%         [x_list_rank,~,~,~] = rankData...
%             (x_list,obj_nomlz_list,con_nomlz_list,coneq_nomlz_list,...
%             cheapcon_function,nonlcon_torlance);
%         x_initial = x_list_rank(1,:);

% select nearest point to construct RBF
RBF_number = min(RBF_number,size(x_list,1));
distance = sum(((x_initial-x_list)./(up_bou-low_bou)).^2,2);
[~,index_list] = sort(distance);
index_list = index_list(1:RBF_number);
x_RBF_list = x_list(index_list,:);
obj_RBF_nomlz_list = obj_nomlz_list(index_list,:);
if ~isempty(con_nomlz_list)
    con_RBF_nomlz_list = con_nomlz_list(index_list,:);
else
    con_RBF_nomlz_list = [];
end
if ~isempty(coneq_nomlz_list)
    coneq_RBF_nomlz_list = coneq_nomlz_list(index_list,:);
else
    coneq_RBF_nomlz_list = [];
end

% modify
% get RBF model and function
[RBF_model_obj,RBF_model_con,RBF_model_coneq,output_RBF] = getRadialBasisModel...
    (x_RBF_list,obj_RBF_nomlz_list,con_RBF_nomlz_list,coneq_RBF_nomlz_list);
object_function_surrogate = output_RBF.object_function_surrogate;
nonlcon_function_surrogate = output_RBF.nonlcon_function_surrogate;
low_bou_local = min(x_RBF_list,[],1);
up_bou_local = max(x_RBF_list,[],1);

% get local infill point
% obtian total constraint function
if ~isempty(nonlcon_function_surrogate) || ~isempty(cheapcon_function)
    constraint_function = @(x) totalconFunction...
        (x,nonlcon_function_surrogate,cheapcon_function);
else
    constraint_function = [];
end
fmincon_options = optimoptions('fmincon','Display','none','Algorithm','sqp','MaxIterations',50);
x_local_infill = fmincon(object_function_surrogate,x_initial,[],[],[],[],...
    low_bou_local,up_bou_local,constraint_function,fmincon_options);

    function Obj = SrgtObj(SRGTRBF_structObj,x)
        [Obj, ~] = RBFPredict(x, SRGTRBF_structObj);
        % Obj = Obj + pred;
    end

    function [g, h] = SrgtCon(SRGTRBF_structCong,x)
        h = [];
        g = RBFPredict(x, SRGTRBF_structCong);
    end

end

function [con,coneq] = totalconFunction...
    (x,nonlcon_function,cheapcon_function)
con = [];
coneq = [];
if ~isempty(nonlcon_function)
    [expencon,expenconeq] = nonlcon_function(x);
    con = [con;expencon];
    coneq = [coneq;expenconeq];
end
if ~isempty(cheapcon_function)
    [expencon,expenconeq] = cheapcon_function(x);
    con = [con;expencon];
    coneq = [coneq;expenconeq];
end
end

%% surrogate model
function [radialbasis_model_obj,radialbasis_model_con,radialbasis_model_coneq,output] = getRadialBasisModel...
    (x_list,obj_list,con_list,coneq_list)
% base on library_data to create radialbasis model and function
% if input model,function will updata model
% object_function is single obj output
% nonlcon_function is normal nonlcon_function which include con,coneq
% con is colume vector,coneq is colume vector
% var_function is same
%
basis_function = @(r) r.^3;

[predict_function_obj,radialbasis_model_obj] = interpRadialBasisPreModel...
    (x_list,obj_list,basis_function);

if ~isempty(con_list)
    predict_function_con = cell(size(con_list,2),1);
    radialbasis_model_con = struct('X',[],'Y',[],...
        'radialbasis_matrix',[],'beta',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],'basis_function',[],...
        'predict_function',[]);
    radialbasis_model_con = repmat(radialbasis_model_con,[size(con_list,2),1]);
    for con_index = 1:size(con_list,2)
        [predict_function_con{con_index},radialbasis_model_con(con_index)] = interpRadialBasisPreModel...
            (x_list,con_list(:,con_index),basis_function);
    end
else
    predict_function_con = [];
    radialbasis_model_con = [];
end

if ~isempty(coneq_list)
    predict_function_coneq = cell(size(coneq_list,2),1);
    radialbasis_model_coneq = struct('X',[],'Y',[],...
        'radialbasis_matrix',[],[],'beta',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],'basis_function',[],...
        'predict_function',[]);
    radialbasis_model_coneq = repmat(radialbasis_model_coneq,[size(coneq_list,2),1]);
    for coneq_index = 1:size(coneq_list,2)
        [predict_function_coneq{coneq_index},radialbasis_model_con(coneq_index)] = interpRadialBasisPreModel...
            (x_list,coneq_list(:,coneq_index),basis_function);
    end
else
    predict_function_coneq = [];
    radialbasis_model_coneq = [];
end

object_function_surrogate = @(X_predict) objectFunctionSurrogate(X_predict,predict_function_obj);
if isempty(radialbasis_model_con) && isempty(radialbasis_model_coneq)
    nonlcon_function_surrogate = [];
else
    nonlcon_function_surrogate = @(X_predict) nonlconFunctionSurrogate(X_predict,predict_function_con,predict_function_coneq);
end

output.object_function_surrogate = object_function_surrogate;
output.nonlcon_function_surrogate = nonlcon_function_surrogate;

    function obj = objectFunctionSurrogate...
            (X_predict,predict_function_obj)
        % connect all predict favl
        %
        obj = predict_function_obj(X_predict);
    end
    function [con,coneq] = nonlconFunctionSurrogate...
            (X_predict,predict_function_con,predict_function_coneq)
        % connect all predict con and coneq
        %
        if isempty(predict_function_con)
            con = [];
        else
            con = zeros(size(X_predict,1),length(predict_function_con));
            for con_index__ = 1:length(predict_function_con)
                con(:,con_index__) = ....
                    predict_function_con{con_index__}(X_predict);
            end
        end
        if isempty(predict_function_coneq)
            coneq = [];
        else
            coneq = zeros(size(X_predict,1),length(predict_function_coneq));
            for coneq_index__ = 1:length(predict_function_coneq)
                coneq(:,coneq_index__) = ...
                    predict_function_coneq{coneq_index__}(X_predict);
            end
        end
    end
end

function [predict_function,radialbasis_model] = interpRadialBasisPreModel...
    (X,Y,basis_function)
% radial basis function interp pre model function
% input initial data X,Y,which are real data
% X,Y are x_number x variable_number matrix
% aver_X,stdD_X is 1 x x_number matrix
% output is a radial basis model,include X,Y,base_function
% and predict_function
%
% Copyright 2023 Adel
%
if nargin < 3
    basis_function = [];
end

[x_number,variable_number] = size(X);

% normalize data
aver_X = mean(X);
stdD_X = std(X);
aver_Y = mean(Y);
stdD_Y = std(Y);
index__ = find(stdD_X == 0);
if ~isempty(index__),stdD_X(index__) = 1;end
index__ = find(stdD_Y == 0);
if ~isempty(index__),stdD_Y(index__) = 1;end
X_nomlz = (X-aver_X)./stdD_X;
Y_nomlz = (Y-aver_Y)./stdD_Y;

if isempty(basis_function)
    c = (prod(max(X_nomlz)-min(X_nomlz))/x_number)^(1/variable_number);
    basis_function = @(r) exp(-(r.^2)/c);
end

% initialization distance of all X
X_dis = zeros(x_number,x_number);
for variable_index = 1:variable_number
    X_dis = X_dis+(X_nomlz(:,variable_index)-X_nomlz(:,variable_index)').^2;
end
X_dis = sqrt(X_dis);

[beta,rdibas_matrix] = interpRadialBasis...
    (X_dis,Y_nomlz,basis_function,x_number);

% initialization predict function
predict_function = @(X_predict) interpRadialBasisPredictor...
    (X_predict,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_number,variable_number,beta,basis_function);

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
% num: number,pred: predict,vari: variable
    function [beta,rdibas_matrix] = interpRadialBasis...
            (X_dis,Y,basis_function,x_number)
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
            (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,beta,basis_function)
        % radial basis function interpolation predict function
        %
        [x_pred_num,~] = size(X_pred);

        % normalize data
        X_pred_nomlz = (X_pred-aver_X)./stdD_X;

        % calculate distance
        X_dis_pred = zeros(x_pred_num,x_num);
        for vari_index = 1:vari_num
            X_dis_pred = X_dis_pred+...
                (X_pred_nomlz(:,vari_index)-X_nomlz(:,vari_index)').^2;
        end
        X_dis_pred = sqrt(X_dis_pred);

        % predict variance
        Y_pred = basis_function(X_dis_pred)*beta;

        % normalize data
        Y_pred = Y_pred*stdD_Y+aver_Y;
    end

end

%% LHD
function [X,X_new,distance_min_nomlz] = getLatinHypercube...
    (sample_number,variable_number,X_exist,...
    low_bou,up_bou,cheapcon_function)
% generate sample sequence latin hypercube
% election sequential method is used(sample and iteration)
%
% sample number is total point in area
% default low_bou is 0,up_bou is 1,cheapcon_function is []
% low_bou and up_bou is colume vector
% X_exist,X,supply_X is x_number x variable_number matrix
% X_exist should meet bou
%
% reference:
% [1]LONG T,LI X,SHI R,et al.,Gradient-Free Trust-Region-Based
% Adaptive Response Surface Method for Expensive Aircraft Optimization[J].
% AIAA Journal,2018,56(2): 862-73.
%
% Copyright 2022 Adel
%
if nargin < 6
    cheapcon_function = [];
    if nargin < 5
        if nargin < 3
            if nargin < 2
                error('getLatinHypercube: lack variable_number');
            end
        end
        low_bou = zeros(variable_number,1);
        up_bou = ones(variable_number,1);
    end
end

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

if sample_number <= 0
    X = [];
    X_new = [];
    distance_min_nomlz = [];
    return;
end

iteration_max = 1000*variable_number;
x_new_number = sample_number-size(X_exist,1);
if x_new_number <= 0
    X = X_exist;
    X_new = [];
    distance_min_nomlz = getMinDistance(X_exist_nomlz);
    return;
end

% get quasi-feasible point
x_initial_number = 100*x_new_number;
if ~isempty(cheapcon_function)
    X_supply_quasi_nomlz = [];
    % check if have enough X_supply_nomlz
    while size(X_supply_quasi_nomlz,1) < 100*x_new_number
        X_supply_initial_nomlz = rand(x_initial_number,variable_number);
        x_index = 1;
        while x_index <= size(X_supply_initial_nomlz,1)
            x_supply = X_supply_initial_nomlz(x_index,:).*(up_bou-low_bou)+low_bou;
            if cheapcon_function(x_supply) > 0
                X_supply_initial_nomlz(x_index,:) = [];
            else
                x_index = x_index+1;
            end
        end
        X_supply_quasi_nomlz = [X_supply_quasi_nomlz;X_supply_initial_nomlz];
    end
else
    X_supply_quasi_nomlz = rand(x_initial_number,variable_number);
end

% iterate and get final x_supply_list
iteration = 0;
x_supply_quasi_number = size(X_supply_quasi_nomlz,1);
distance_min_nomlz = 0;
X_new_nomlz = [];
while iteration <= iteration_max
    % random select x_new_number X to X_trial_nomlz
    x_select_index = randperm(x_supply_quasi_number,x_new_number);

    % get distance min itertion X_
    distance_min_iteration = getMinDistanceIter...
        (X_supply_quasi_nomlz(x_select_index,:),X_exist_nomlz);

    % if distance_min_iteration is large than last time
    if distance_min_iteration > distance_min_nomlz
        distance_min_nomlz = distance_min_iteration;
        X_new_nomlz = X_supply_quasi_nomlz(x_select_index,:);
    end

    iteration = iteration+1;
end
X_new = X_new_nomlz.*(up_bou-low_bou)+low_bou;
X = [X_new;X_exist];
distance_min_nomlz = getMinDistance([X_new_nomlz;X_exist_nomlz]);

    function distance_min__ = getMinDistance(x_list__)
        % get distance min from x_list
        %
        % sort x_supply_list_initial to decrese distance calculate times
        x_list__ = sortrows(x_list__,1);
        [sample_number__,variable_number__] = size(x_list__);
        distance_min__ = variable_number__;
        for x_index__ = 1:sample_number__
            x_curr__ = x_list__(x_index__,:);
            x_next_index__ = x_index__ + 1;
            % only search in min_distance(x_list had been sort)
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
    function distance_min__ = getMinDistanceIter...
            (x_list__,x_exist_list__)
        % get distance min from x_list
        %
        % sort x_supply_list_initial to decrese distance calculate times
        x_list__ = sortrows(x_list__,1);
        [sample_number__,variable_number__] = size(x_list__);
        distance_min__ = variable_number__;
        for x_index__ = 1:sample_number__
            x_curr__ = x_list__(x_index__,:);
            x_next_index__ = x_index__ + 1;
            % only search in min_distance(x_list had been sort)
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
            for x_exist_index = 1:size(x_exist_list__,1)
                x_next__ = x_exist_list__(x_exist_index,:);
                distance_temp__ = sum((x_next__-x_curr__).^2);
                if distance_temp__ < distance_min__
                    distance_min__ = distance_temp__;
                end
            end
        end
        distance_min__ = sqrt(distance_min__);
    end
end
