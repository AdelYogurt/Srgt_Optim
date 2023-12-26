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
% benchmark_name = 'G01';
benchmark_name = 'G06';

[model_function,variable_number,low_bou,up_bou,...
    object_function,A,B,Aeq,Beq,nonlcon_function] = benchmark.getBenchmarkFunction(benchmark_type,benchmark_name);

% x_initial = rand(1,variable_number).*(up_bou-low_bou)+low_bou;
% [x_best,fval_best,~,output] = fmincon(object_function,x_initial,A,B,Aeq,Beq,low_bou,up_bou,[],optimoptions('fmincon','Algorithm','sqp','MaxFunctionEvaluations',10000,'Display','iter-detailed'))

cheapcon_function = [];

%% single run

[x_best,fval_best,NFE,output] = optimalSurrogateSRBFSVM...
    (model_function,variable_number,low_bou,up_bou,...
    cheapcon_function,200,500)
result_x_best = output.result_x_best;
result_fval_best = output.result_fval_best;

figure(1);
plot(result_fval_best);

%% repeat run

% repeat_number = 10;
% result_fval = zeros(repeat_number,1);
% result_NFE = zeros(repeat_number,1);
% max_NFE = 200;
% for repeat_index = 1:repeat_number
%     [x_best,fval_best,NFE,output] = optimalSurrogateSRBFSVM...
%         (model_function,variable_number,low_bou,up_bou,...
%         cheapcon_function,max_NFE,300,1e-6,1e-3);
% 
%     result_fval(repeat_index) = fval_best;
%     result_NFE(repeat_index) = NFE;
% end
% 
% fprintf('Fval     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_fval),mean(result_fval),max(result_fval),std(result_fval));
% fprintf('NFE     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_NFE),mean(result_NFE),max(result_NFE),std(result_NFE));
% save([benchmark_name,'_SRBF_SVM','.mat']);

%% main
function [x_best,fval_best,NFE,output] = optimalSurrogateSRBFSVM...
    (model_function,variable_number,low_bou,up_bou,...
    cheapcon_function,....
    NFE_max,iteration_max,torlance,nonlcon_torlance)
% surrogate base optimal use radias base function method version 0
% use SVM to get interest point
% FS_FCM to get interest point center point
% and updata interest space
% all function exchange x should be colume vector
% x_list is x_number x variable_number matrix
% both nonlcon_function and cheapcon_function format is [con,coneq]
% model_function should output fval,format is [fval,con,coneq]
% con or coneq can be colume vector if there was more than one constrain
%
% referance: [1] SHI R,LIU L,LONG T,et al. Sequential Radial Basis
% Function Using Support Vector Machine for Expensive Design Optimization
% [J]. AIAA Journal,2017,55(1): 214-27.
%
% Copyright Adel 2022.10
%
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

if nargin < 5
    cheapcon_function = [];
end

DRAW_FIGURE_FLAG = 0; % whether draw data
INFORMATION_FLAG = 1; % whether print data
CONVERGENCE_JUDGMENT_FLAG = 0; % whether judgment convergence
WRIRE_FILE_FLAG = 0; % whether write to file

% hyper parameter
sample_number_initial = min((variable_number+1)*(variable_number+2)/2,5*variable_number);
sample_number_iteration = variable_number;
sample_number_data = 100*sample_number_initial;
eta = 1/variable_number; % space decrease coefficient

penalty_SVM = 100;
m = 2; % clustering parameter

% max fval when normalize fval,con,coneq
nomlz_fval = 10;

% NFE and iteration setting
if isempty(NFE_max)
    NFE_max = 10+10*variable_number;
end

if isempty(iteration_max)
    iteration_max = 20+20*variable_number;
end

% min boundary of sample area
bou_min = 1e-3;

% surrogate add point protect range
protect_range = 1e-5;

done = 0;NFE = 0;iteration = 0;

% step 2
% use latin hypercube method to get initial sample x_list
X_updata = lhsdesign(sample_number_initial,variable_number).*(up_bou-low_bou)+low_bou;

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

X_updata = X_updata(2:end,:);

result_x_best = zeros(iteration_max,variable_number);
result_fval_best = zeros(iteration_max,1);

iteration = iteration+1;

x_data_list = lhsdesign(sample_number_data,variable_number).*...
    (up_bou-low_bou)+low_bou;
while ~done
    % step 3
    % updata data library by x_list
    [X_updata,Fval_updata,Con_updata,Coneq_updata,Vio_updata,Ks_Updata,repeat_index,NFE_updata] = data_library.dataUpdata(X_updata(2:end,:),protect_range);
    NFE = NFE+NFE_updata;

    % nomalization all data by max fval and to create surrogate model
    [X,Fval,Con,Coneq,Vio,Ks]=data_library.dataLoad;

    X_model = X;
    fval_max = max(abs(Fval),[],1);
    Fval_model=Fval/fval_max*nomlz_fval;
    if ~isempty(Con)
        con_maX = max(abs(Con),[],1);
        Con_model = Con./con_maX*nomlz_fval;
    else
        Con_model = [];
    end
    if ~isempty(Coneq)
        coneq_maX = max(abs(Coneq),[],1);
        Coneq_model = Coneq./coneq_maX*nomlz_fval;
    else
        Coneq_model = [];
    end
    if ~isempty(Ks)
        ks_maX = max(abs(Ks),[],1);
        Ks_model = Ks./ks_maX*nomlz_fval;
    else
        Ks_model = [];
    end

    % step 4
    % generate ERBF_QP model use normalization fval
    %     [ERBF_model_fval,ERBF_model_con,ERBF_model_coneq,output_ERBF] = getEnsemleRadialBasisModel...
    %         (X_model,Fval_model,Con_model,Coneq_model);
    %     object_function_surrogate = output_ERBF.object_function_surrogate;
    %     nonlcon_function_surrogate = output_ERBF.nonlcon_function_surrogate;

    [RBF_model_fval,RBF_model_con,RBF_model_coneq,output_RBF] = getRadialBasisModel...
        (X_model,Fval_model,Con_model,Coneq_model);
    object_function_surrogate = output_RBF.object_function_surrogate;
    nonlcon_function_surrogate = output_RBF.nonlcon_function_surrogate;

    % step 5
    % MSP guideline to obtain x_adapt
    [x_infill,~,exitflag,~] = findMinMSP...
        (object_function_surrogate,variable_number,low_bou,up_bou,nonlcon_function_surrogate,...
        cheapcon_function);

    if exitflag == -2
        % optimal feasiblilty if do not exist feasible point
        object_nonlcon_function_surrogate = @(x) objectNonlconFunctionSurrogate(x,nonlcon_function_surrogate);
        [x_infill,~,exitflag,~] = findMinMSP...
            (object_nonlcon_function_surrogate,variable_number,low_bou,up_bou,[],...
            cheapcon_function);
    end

    % check x_infill if exist in data library
    % if not,updata data libraray
    [x_infill,fval_infill,con_infill,coneq_infill,vio_infill,ks_infill,repeat_index,NFE_updata] = data_library.dataUpdata(x_infill,protect_range);
    NFE = NFE+NFE_updata;

    % when x_potential is exist in data library,x_potential_add will be
    % empty,this times we will use origin point data
    if isempty(x_infill)
        x_infill = X(repeat_index,:);
        fval_infill = Fval(repeat_index,:);
        if ~isempty(Con)
            con_infill = Con(repeat_index,:);
        else
            con_infill = [];
        end
        if ~isempty(Coneq)
            coneq_infill = Coneq(repeat_index,:);
        else
            coneq_infill = [];
        end
    end

    if DRAW_FIGURE_FLAG && variable_number < 3
        interpVisualize(RBF_model_fval,low_bou,up_bou);
        line(x_infill(1),x_infill(2),fval_infill/fval_max*nomlz_fval,'Marker','o','color','r','LineStyle','none')
    end

    % step 6
    % find best result to record
    [x_best,fval_best,con_best,coneq_best] = findMinRaw...
        (X,Fval,Con,Coneq,...
        cheapcon_function,nonlcon_torlance);
    vio_best = calViolation(con_best,coneq_best,nonlcon_torlance);

    if INFORMATION_FLAG
        fprintf('fval:    %f    violation:    %f    NFE:    %-3d\n',fval_best,vio_best,NFE);
%         fprintf('iteration:          %-3d    NFE:    %-3d\n',iteration,NFE);
%         fprintf('current x:          %s\n',num2str(x_infill));
%         fprintf('current value:      %f\n',fval_infill);
%         fprintf('current violation:  %s  %s\n',num2str(con_infill),num2str(coneq_infill));
%         fprintf('\n');
    end

    result_x_best(iteration,:) = x_best;
    result_fval_best(iteration,:) = fval_best;
    iteration = iteration+1;

    % forced interrupt
    if iteration > iteration_max || NFE >= NFE_max
        done = 1;
    end

    % convergence judgment
    if CONVERGENCE_JUDGMENT_FLAG
        if (iteration > 2 && ...
                abs((fval_infill-fval_potential_old)/fval_potential_old) < torlance)
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
        % step 7
        % using SVM to identify area which is interesting
        [X,Fval,Con,Coneq,Vio,Ks]=data_library.dataLoad;

        X_model = X;
        fval_max = max(abs(Fval),[],1);
        Fval_model=Fval/fval_max*nomlz_fval;
        if ~isempty(Con)
            con_maX = max(abs(Con),[],1);
            Con_model = Con./con_maX*nomlz_fval;
        else
            Con_model = [];
        end
        if ~isempty(Coneq)
            coneq_maX = max(abs(Coneq),[],1);
            Coneq_model = Coneq./coneq_maX*nomlz_fval;
        else
            Coneq_model = [];
        end
        if ~isempty(Ks)
            ks_maX = max(abs(Ks),[],1);
            Ks_model = Ks./ks_maX*nomlz_fval;
        else
            Ks_model = [];
        end

        if expensive_nonlcon_flag

            % generate filter
            filter_index_list = getParetoFront([Fval_model,Ks_model]); % filter point list
            feasible_index_list = Vio == 0; % feasible point list

            fval_label = -ones(size(X,1),1);
            fval_label(filter_index_list) = 1;

            % feasible point set label 1
            fval_label(feasible_index_list) = 1;

            % use filter and train SVM
            [SVM_predict_function,SVM_model] = classifySupportVectorMachine...
                (X,fval_label,penalty_SVM);

            if DRAW_FIGURE_FLAG && variable_number < 3
                classifyVisualization...
                    (SVM_model,low_bou,up_bou);
            end

            % get data to obtain clustering center
            x_sup_list = x_data_list(SVM_predict_function(x_data_list) == 1,:);

            if isempty(x_sup_list)
                % no sup point found use filter point
                if isempty(feasible_index_list)
                    if ~isempty(Con)
                        Con_filter = Con_model(filter_index_list,:);
                    else
                        Con_filter = [];
                    end
                    if ~isempty(Coneq)
                        Coneq_filter = Coneq(filter_index_list,:);
                    else
                        Coneq_filter = [];
                    end
                    max_totalcon_list = max([Con_filter,Coneq_filter],[],2);
                    [~,filter_min_index] = min(max_totalcon_list);
                    x_center = X(filter_index_list(filter_min_index),:);
                else
                    [~,min_fval_index] = min(Fval(feasible_index_list));
                    x_center = X(feasible_index_list(min_fval_index),:);
                end
            end
        else
            % interset sampling
            fval_threshold = prctile(Fval_model(~feasi_boolean_list),50-40*sqrt(NFE/NFE_max));

            % step 7-1
            % classify exist data
            less_list = Fval <= fval_threshold;
            fval_label = -ones(size(X,1),1);
            fval_label(less_list) = 1;

            % step 7-2
            % get a large number of x point,use SVM to predict x point
            [SVM_predict_function,SVM_model] = classifySupportVectorMachine...
                (X,fval_label,penalty_SVM);
            if DRAW_FIGURE_FLAG && variable_number < 3
                classifyVisualization...
                    (SVM_model,low_bou,up_bou);
            end
            % get data to obtain clustering center
            x_sup_list = x_data_list(SVM_predict_function(x_data_list) == 1,:);

            if isempty(x_sup_list)
                % no sup point found use filter point
                x_sup_list = X(less_list,:);
            end
        end

        % step 7-3
        % calculate clustering center
        if ~isempty(x_sup_list)
            FC_model = classifyFuzzyClustering...
                (x_sup_list,1,low_bou,up_bou,m);
            x_center = FC_model.center_list;
        end

        % updata ISR
        x_potential_nomlz = (x_infill-low_bou)./(up_bou-low_bou);
        x_center_nomlz = (x_center-low_bou)./(up_bou-low_bou);
        bou_range_nomlz = eta*norm(x_potential_nomlz-x_center_nomlz,2);
        if bou_range_nomlz < bou_min
            bou_range_nomlz = bou_min;
        end
        bou_range = bou_range_nomlz.*(up_bou-low_bou);
        low_bou_ISR = x_infill-bou_range;
        low_bou_ISR = max(low_bou_ISR,low_bou);
        up_bou_ISR = x_infill+bou_range;
        up_bou_ISR = min(up_bou_ISR,up_bou);

        if DRAW_FIGURE_FLAG && variable_number < 3
            bou_line = [low_bou_ISR;[low_bou_ISR(1),up_bou_ISR(2)];up_bou_ISR;[up_bou_ISR(1),low_bou_ISR(2)];low_bou_ISR];
            line(bou_line(:,1),bou_line(:,2));
            line(x_infill(1),x_infill(2),'Marker','x')
        end

        % sampling in ISR
        %         [x_list_exist,~,~,~] = dataLibraryRead...
        %             (data_library_name,low_bou_ISR,up_bou_ISR);
        %     [~,X_updata,~] = getLatinHypercube...
        %         (sample_number_iteration+size(x_list_exist,1),variable_number,x_list_exist,...
        %         low_bou_ISR,up_bou_ISR,cheapcon_function);
        X_updata = lhsdesign(sample_number_iteration,variable_number)...
            .*(up_bou_ISR-low_bou_ISR)+low_bou_ISR;
    end

    x_potential_old = x_infill;
    fval_potential_old = fval_infill;
    fval_best_old = fval_best;
end
result_x_best = result_x_best(1:iteration-1,:);
result_fval_best = result_fval_best(1:iteration-1);

output.result_x_best = result_x_best;
output.result_fval_best = result_fval_best;
output.data_library = data_library;

    function fval = objectNonlconFunctionSurrogate(x,nonlcon_function_surrogate)
        [con__,coneq__] = nonlcon_function_surrogate(x);
        fval = 0;
        if ~isempty(con__)
            fval = fval+sum(max(con__,0).^2);
        end
        if ~isempty(coneq__)
            fval = fval+sum(max(con__,0).^2);
        end
    end

    function [X_updata,fval_updata_list,con_updata_list,coneq_updata_list,NFE_updata,repeat_index] = dataLibraryWriteProtect...
            (data_library_name,model_function,x_add_list,...
            x_list,low_bou,up_bou,protect_range)
        % function updata data with same_point_avoid protect
        % return fval
        % all list is x_number x variable_number matrix
        % notice if x_add is exist in library,point will be delete
        %
        variable_number__ = size(x_list,2);
        NFE_updata = 0;
        X_updata = [];fval_updata_list = [];con_updata_list = [];coneq_updata_list = [];repeat_index = [];
        for x_index__ = 1:size(x_add_list,1)
            x_updata__ = x_add_list(x_index__,:);

            % check x_potential if exist in data library
            % if not,updata data libraray
            distance__ = sum((abs(x_updata__-x_list)./(up_bou-low_bou)),2);
            [~,min_index__] = min(distance__);
            if distance__(min_index__) < variable_number__*protect_range
                % distance to exist point of point to add is small than protect_range
                repeat_index = [repeat_index;min_index__];
            else
                [x_updata__,fval_updata__,con_updata__,coneq_updata__] = dataLibraryWrite...
                    (data_library_name,model_function,x_updata__);NFE_updata = NFE_updata+1;
                X_updata = [X_updata;x_updata__];
                fval_updata_list = [fval_updata_list;fval_updata__];
                con_updata_list = [con_updata_list;con_updata__];
                coneq_updata_list = [coneq_updata_list;coneq_updata__];
            end
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

function [x_best,fval_best,exitflag,output] = findMinMSP...
    (object_function_surrogate,variable_number,low_bou,up_bou,nonlcon_function_surrogate,...
    cheapcon_function)
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
% % object function convert to penalty function if input nonlcon function
% if ~isempty(nonlcon_function)
%     object_function = @(x) penaltyFunction(object_function,x,nonlcon_function);
%     constraint_function = cheapcon_function;
% end

% obtian total constraint function
if ~isempty(nonlcon_function_surrogate) || ~isempty(cheapcon_function)
    constraint_function = @(x) totalconFunction...
        (x,nonlcon_function_surrogate,cheapcon_function);
else
    constraint_function = [];
end

% generate initial population for ga
population_matrix = zeros(max(10,2*variable_number),variable_number);
for population_index = 1:size(population_matrix,1)
    x = rand(1,variable_number).*(up_bou-low_bou)+low_bou;
    if ~isempty(cheapcon_function)
        [con,coneq] = cheapcon_function(x);
        while sum([~(con < 0);abs(coneq) < 0])
            x = rand(1,variable_number).*(up_bou-low_bou)+low_bou;
            [con,coneq] = cheapcon_function(x);
        end
    end
    population_matrix(population_index,:) = x;
end

% optiaml
ga_option = optimoptions('ga','FunctionTolerance',1e-2,'ConstraintTolerance',1e-2,...
    'PopulationSize',max(10,2*variable_number),...
    'MaxGenerations',100,'InitialPopulationMatrix',population_matrix,...
    'display','none');
[x_best,fval_best,exitflag,output] = ga...
    (object_function_surrogate,variable_number,[],[],[],[],low_bou',up_bou',constraint_function,ga_option);
fmincon_option = optimoptions('fmincon','FunctionTolerance',1e-6,'ConstraintTolerance',1e-6,...
    'algorithm','sqp',....
    'display','none');
[x_best,fval_best,exitflag,output] = fmincon...
    (object_function_surrogate,x_best,[],[],[],[],low_bou,up_bou,constraint_function,fmincon_option);

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

function [x_best,fval_best,con_best,coneq_best] = findMinRaw...
    (x_list,fval_list,con_list,coneq_list,...
    cheapcon_function,nonlcon_torlance)
% find min fval in raw data
% x_list,rank is variable
% con_list,rank is con
% coneq_list,rank is coneq
% function will find min fval in con==0
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
    fval_list = fval_list(index);
    if ~isempty(con_list)
        con_list = con_list(index,:);
    end
    if ~isempty(coneq_list)
        coneq_list = coneq_list(index,:);
    end

    % min fval
    [fval_best,index_best] = min(fval_list);
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
    fval_best = fval_list(index_best);
    x_best = x_list(index_best,:);
    if ~isempty(con_list)
        con_best = con_list(index_best,:);
    end
    if ~isempty(coneq_list)
        coneq_best = coneq_list(index_best,:);
    end
end
end

function pareto_index_list = getParetoFront(data_list)
% distinguish pareto front of data list
% data_list is x_number x data_number matrix
% notice if all data of x1 is less than x2,x1 domain x2
%
x_number = size(data_list,1);
pareto_index_list = []; % sort all index of filter point list

% select no domain filter
for x_index = 1:x_number
    data = data_list(x_index,:);
    pareto_index = 1;
    add_filter_flag = 1;
    while pareto_index <= length(pareto_index_list)
        % compare x with exit pareto front point
        x_pareto_index = pareto_index_list(pareto_index,:);

        % contain constraint of x_filter
        data_pareto = data_list(x_pareto_index,:);

        % compare x with x_pareto
        judge = data >= data_pareto;
        if ~sum(~judge)
            add_filter_flag = 0;
            break;
        end

        % if better or equal than exit pareto point,reject pareto point
        judge = data <= data_pareto;
        if ~sum(~judge)
            pareto_index_list(pareto_index) = [];
            pareto_index = pareto_index-1;
        end

        pareto_index = pareto_index+1;
    end

    % add into pareto list if possible
    if add_filter_flag
        pareto_index_list = [pareto_index_list;x_index];
    end
end
end

%% FCM
function FC_model = classifyFuzzyClustering...
    (X,classify_number,low_bou,up_bou,m)
% get fuzzy cluster model
% X is x_number x variable_number matrix
% center_list is classify_number x variable_number matrix
%
if nargin < 4
    up_bou = [];
    if nargin < 3
        low_bou = [];
    end
end
iteration_max = 100;
torlance = 1e-3;

[x_number,variable_number] = size(X);

% nomalization data
if isempty(low_bou)
    low_bou = min(X,[],1);
else
    low_bou = low_bou(:)';
end
if isempty(up_bou)
    up_bou = max(X,[],1);
else
    up_bou = up_bou(:)';
end
X_nomlz = (X-low_bou)./(up_bou-low_bou);

% if x_number equal 1,clustering cannot done
if x_number==1
    FC_model.X = X;
    FC_model.X_normalize = X_nomlz;
    FC_model.center_list = X;
    FC_model.fval_loss_list = [];
    FC_model.x_class_list = ones(x_number,1);
    return;
end

U = zeros(x_number,classify_number);
center_list = rand(classify_number,variable_number)*0.5;
iteration = 0;
done = 0;
fval_loss_list = zeros(iteration_max,1);

% get X_center_dis_sq and classify x to each x center
% classify_number x x_number matrix
X_center_dis_sq = zeros(x_number,classify_number);
x_class_list = zeros(x_number,1);
for x_index = 1:x_number
    for classify_index = 1:classify_number
        temp = (X_nomlz(x_index,:)-center_list(classify_index,:));
        X_center_dis_sq(x_index,classify_index) = temp*temp';
    end
    [~,x_class_list(x_index)] = min(X_center_dis_sq(x_index,:));
end

while ~done
    % updata classify matrix U
    % classify matrix U is x weigth of each center
    for classify_index = 1:classify_number
        for x_index = 1:x_number
            U(x_index,classify_index) = ...
                1/sum((X_center_dis_sq(x_index,classify_index)./X_center_dis_sq(x_index,:)).^(1/(m-1)));
        end
    end

    % updata center_list
    center_list_old = center_list;
    for classify_index = 1:classify_number
        center_list(classify_index,:) = ...
            sum((U(:,classify_index)).^m.*X_nomlz,1)./...
            sum((U(:,classify_index)).^m,1);
    end

    % updata X_center_dis_sq
    for x_index = 1:x_number
        for classify_index = 1:classify_number
            temp = (X_nomlz(x_index,:)-center_list(classify_index,:));
            X_center_dis_sq(x_index,classify_index) = temp*temp';
        end
        [~,x_class_list(x_index)] = min(X_center_dis_sq(x_index,:));
    end

    %     plot(center_list(:,1),center_list(:,2));

    % forced interrupt
    if iteration > iteration_max
        done = 1;
    end

    % convergence judgment
    if sum(sum(center_list_old-center_list).^2) < torlance
        done = 1;
    end

    iteration = iteration+1;
    fval_loss_list(iteration) = sum(sum(U.^m.*X_center_dis_sq));
end
fval_loss_list(iteration+1:end) = [];
center_list = center_list.*(up_bou-low_bou)+low_bou;

FC_model.X = X;
FC_model.X_normalize = X_nomlz;
FC_model.center_list = center_list;
FC_model.fval_loss_list = fval_loss_list;
FC_model.x_class_list = x_class_list;

end

%% SVM
function [predict_function,SVM_model] = classifySupportVectorMachine...
    (X,Class,C,kernel_function)
% generate support vector machine model
% use fmincon to get alpha
% only support binary classification,-1 and 1
% X,Y is x_number x variable_number matrix
% C is penalty factor,default is empty
% kernel_function default is gauss kernal function
%
if nargin < 4
    kernel_function = [];
    if nargin < 3
        C = [];
    end
end

[x_number,variable_number] = size(X);

% normalization data
aver_X = mean(X);
stdD_X = std(X);
index__ = find(stdD_X == 0);
if  ~isempty(index__),stdD_X(index__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;

Y = Class;

% default kernal function
if isempty(kernel_function)
    % notice after standard normal distribution normalize
    % X usually distribution in -2 to 2,so divide by 16
    sigma = -100*log(1/sqrt(x_number))/variable_number^2/16;
    kernel_function = @(U,V) kernelGaussian(U,V,sigma);
end

% initialization kernal function process X_cov
K = kernel_function(X_nomlz,X_nomlz);

% min SVM object function to get alpha
object_function = @(alpha) -objectFunction(alpha,K,Y);
alpha = ones(x_number,1)*0.5;
low_bou_fmincon = 0*ones(x_number,1);
if isempty(C) || C==0
    up_bou_fmincon = [];
else
    up_bou_fmincon = C*ones(x_number,1);
end
Aeq = Y';
fmincon_options = optimoptions('fmincon','Display','none','Algorithm','sqp');
alpha = fmincon(object_function,alpha,...
    [],[],Aeq,0,low_bou_fmincon,up_bou_fmincon,[],fmincon_options);

% obtain other paramter
alpha_Y = alpha.*Y;

w = sum(alpha_Y.*X_nomlz);
index_list = find(alpha > 1e-6); % support vector
alpha_Y_cov = K*alpha_Y;
b = sum(Y(index_list)-alpha_Y_cov(index_list))/length(index_list);

% generate predict function
predict_function = @(x) classifySupportVectorMachinePredictor...
    (x,X_nomlz,alpha_Y,b,aver_X,stdD_X,kernel_function);

% output model
SVM_model.X = X;
SVM_model.Class = Class;
SVM_model.Y = Y;
SVM_model.X_nomlz = X_nomlz;
SVM_model.aver_X = aver_X;
SVM_model.stdD_X = stdD_X;
SVM_model.alpha = alpha;
SVM_model.w = w;
SVM_model.b = b;
SVM_model.kernel_function = kernel_function;
SVM_model.predict_function = predict_function;

    function fval = objectFunction(alpha,K,Y)
        % support vector machine maximum object function
        %
        alpha = alpha(:);
        alpha_Y__ = alpha.*Y;
        fval = sum(alpha)-alpha_Y__'*K*alpha_Y__/2;
    end
    function [Class_pred,Probability] = classifySupportVectorMachinePredictor...
            (X_pred,X_nomlz,alpha_Y,b,aver_X,stdD_X,kernel_function)
        % predict_fval is 1 or -1,predict_class is 1 or 0
        %
        % x input is colume vector
        %
        X_pred_nomlz = (X_pred-aver_X)./stdD_X;
        K_pred = kernel_function(X_pred_nomlz,X_nomlz);
        Probability = K_pred*alpha_Y+b;
        Probability = 1./(1+exp(-Probability));
        Class_pred = Probability > 0.5;
    end
    function K = kernelGaussian(U,V,sigma)
        % gaussian kernal function
        %
        K = zeros(size(U,1),size(V,1));
        vari_num = size(U,2);
        for vari_index = 1:vari_num
            K = K+(U(:,vari_index)-V(:,vari_index)').^2;
        end
        K = exp(-K*sigma);
    end
end

%% surrogate model
function [radialbasis_model_fval,radialbasis_model_con,radialbasis_model_coneq,output] = getRadialBasisModel...
    (x_list,fval_list,con_list,coneq_list)
% base on library_data to create radialbasis model and function
% if input model,function will updata model
% object_function is single fval output
% nonlcon_function is normal nonlcon_function which include con,coneq
% con is colume vector,coneq is colume vector
% var_function is same
%
[predict_function_fval,radialbasis_model_fval] = interpRadialBasisPreModel...
    (x_list,fval_list);

if ~isempty(con_list)
    predict_function_con = cell(size(con_list,2),1);
    radialbasis_model_con = struct('X',[],'Y',[],...
        'radialbasis_matrix',[],'beta',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],'basis_function',[],...
        'predict_function',[]);
    radialbasis_model_con = repmat(radialbasis_model_con,[size(con_list,2),1]);
    for con_index = 1:size(con_list,2)
        [predict_function_con{con_index},radialbasis_model_con(con_index)] = ...
            interpRadialBasisPreModel(x_list,con_list(:,con_index));
    end
else
    predict_function_con = [];
    radialbasis_model_con = [];
end

if ~isempty(coneq_list)
    predict_function_coneq = cell(size(con_list,2),1);
    radialbasis_model_coneq = struct('X',[],'Y',[],...
        'radialbasis_matrix',[],'beta',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],'basis_function',[],...
        'predict_function',[]);
    radialbasis_model_coneq = repmat(radialbasis_model_coneq,[size(coneq_list,2),1]);
    for coneq_index = 1:size(coneq_list,2)
        [predict_function_coneq{coneq_index},radialbasis_model_coneq(coneq_index)] = ...
            interpRadialBasisPreModel(x_list,coneq_list(:,coneq_index));
    end
else
    predict_function_coneq = [];
    radialbasis_model_coneq = [];
end

object_function_surrogate = @(predict_x) objectFunctionSurrogate(predict_x,predict_function_fval);
if isempty(predict_function_con) && isempty(predict_function_coneq)
    nonlcon_function_surrogate = [];
else
    nonlcon_function_surrogate = @(predict_x) nonlconFunctionSurrogate(predict_x,predict_function_con,predict_function_coneq);
end

output.object_function_surrogate = object_function_surrogate;
output.nonlcon_function_surrogate = nonlcon_function_surrogate;

    function fval = objectFunctionSurrogate...
            (X_predict,predict_function_fval)
        % connect all predict favl
        %
        fval = predict_function_fval(X_predict);
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
    c = (prod(max(X_nomlz)-min(Y_nomlz))/x_number)^(1/variable_number);
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

function [ERBF_model_fval,ERBF_model_con,ERBF_model_coneq,output] = getEnsemleRadialBasisModel...
    (x_list,fval_list,con_list,coneq_list)
% base on library_data to create kriging model and function
% object_function is single fval output
% nonlcon_function is normal nonlcon_function which include con,coneq
% con is colume vector,coneq is colume vector
%
[predict_function_fval,ERBF_model_fval] = interpEnsemleRadialBasisPreModel...
    (x_list,fval_list);

if ~isempty(con_list)
    predict_function_con = cell(size(con_list,2),1);
    ERBF_model_con = struct('X',[],'Y',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],...
        'basis_function_list',[],'c_list',[],'beta_list',[],...
        'rdibas_matrix_list',[],'inv_rdibas_matrix_list',[],'model_error_list',[],'w',[],...
        'predict_function',[]);
    ERBF_model_con = repmat(ERBF_model_con,[size(con_list,2),1]);
    for con_index = 1:size(con_list,2)
        [predict_function_con{con_index},ERBF_model_con(con_index)] = ...
            interpolationEnsemleRadialBasisPreModel(x_list,con_list(:,con_index));
    end
else
    predict_function_con = [];
    ERBF_model_con = [];
end

if ~isempty(coneq_list)
    predict_function_coneq = cell(size(con_list,2),1);
    ERBF_model_coneq = struct('X',[],'Y',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],...
        'basis_function_list',[],'c_list',[],'beta_list',[],...
        'rdibas_matrix_list',[],'inv_rdibas_matrix_list',[],'model_error_list',[],'w',[],...
        'predict_function',[]);
    ERBF_model_coneq = repmat(ERBF_model_coneq,[size(coneq_list,2),1]);
    for coneq_index = 1:size(coneq_list,2)
        [predict_function_coneq{coneq_index},ERBF_model_coneq(coneq_index)] = ...
            interpolationEnsemleRadialBasisPreModel(x_list,coneq_list(:,coneq_index));
    end
else
    predict_function_coneq = [];
    ERBF_model_coneq = [];
end

object_function_surrogate = @(predict_x) objectFunctionSurrogate(predict_x,predict_function_fval);
if isempty(predict_function_con) && isempty(predict_function_coneq)
    nonlcon_function_surrogate = [];
else
    nonlcon_function_surrogate = @(predict_x) nonlconFunctionSurrogate(predict_x,predict_function_con,predict_function_coneq);
end

output.object_function_surrogate = object_function_surrogate;
output.nonlcon_function_surrogate = nonlcon_function_surrogate;

    function fval = objectFunctionSurrogate...
            (X_predict,predict_function_fval)
        % connect all predict favl
        %
        fval = predict_function_fval(X_predict);
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

function [predict_function,ensemleradialbasis_model] = interpEnsemleRadialBasisPreModel...
    (X,Y)
% get ensemle radial basis function interpolation model function
% using quadratic programming to calculate weigth of each sub model
% using cubic interpolation optimal to decrese time use
% input initial data X,Y,which are real data
% X,Y are x_number x variable_number matrix
% aver_X,stdD_X is 1 x x_number matrix
% output is a radial basis model,include X,Y,base_function
% and predict_function
%
% reference: [1] SHI R,LIU L,LONG T,et al. An efficient ensemble of
% radial basis functions method based on quadratic programming [J].
% Engineering Optimization,2016,48(1202 - 25.
%
% Copyright 2023 Adel
%
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

c_initial = (prod(max(X_nomlz)-min(Y_nomlz))/x_number)^(1/variable_number);

% initialization distance of all X
X_dis = zeros(x_number,x_number);
for variable_index = 1:variable_number
    X_dis = X_dis+(X_nomlz(:,variable_index)-X_nomlz(:,variable_index)').^2;
end
X_dis = sqrt(X_dis);

% linear kernal function
basis_function_linear = @(r,c) r+c;
dRM_dc_function = @(x_number,X_dis,rdibas_matrix,c) ones(x_number,x_number);
object_function = @(c) objectFunctionRadiabasis....
    (X_dis,Y_nomlz,x_number,basis_function_linear,c,dRM_dc_function);

[linear_c_backward,fval_backward,~] = optimalCubicInterp...
    (object_function,-1e2,-1e2,1e2,1e-3);
[linear_c_forward,fval_forward,~] = optimalCubicInterp...
    (object_function,1e2,-1e2,1e2,1e-3);
if fval_forward < fval_backward
    c_linear = linear_c_forward;
else
    c_linear = linear_c_backward;
end

% gauss kernal function
basis_function_gauss = @(r,c) exp(-c*r.^2);
dRM_dc_function = @(x_number,X_dis,rdibas_matrix,c) -X_dis.^2.*rdibas_matrix;
object_function = @(c) objectFunctionRadiabasis....
    (X_dis,Y_nomlz,x_number,basis_function_gauss,c,dRM_dc_function);
[c_gauss,~,~,~] = optimalCubicInterp...
    (object_function,c_initial,1e-2,1e2,1e-3);

% spline kernal function
basis_function_spline = @(r,c) r.^2.*log(r.^2*c+1e-3);
dRM_dc_function = @(x_number,X_dis,rdibas_matrix,c) X_dis.^4./(X_dis.^2*c+1e-3);
object_function = @(c) objectFunctionRadiabasis....
    (X_dis,Y_nomlz,x_number,basis_function_spline,c,dRM_dc_function);
[c_spline,~,~,~] = optimalCubicInterp...
    (object_function,c_initial,1e-2,1e2,1e-3);

% triple kernal function
basis_function_triple = @(r,c) (r+c).^3;
dRM_dc_function = @(x_number,X_dis,rdibas_matrix,c) 3*(X_dis+c).^3;
object_function = @(c) objectFunctionRadiabasis....
    (X_dis,Y_nomlz,x_number,basis_function_triple,c,dRM_dc_function);
[c_triple,~,~,~] = optimalCubicInterp...
    (object_function,c_initial,1e-2,1e2,1e-3);

% multiquadric kernal function
basis_function_multiquadric = @(r,c) sqrt(r+c);
dRM_dc_function = @(x_number,X_dis,rdibas_matrix,c) 0.5./rdibas_matrix;
object_function = @(c) objectFunctionRadiabasis....
    (X_dis,Y_nomlz,x_number,basis_function_multiquadric,c,dRM_dc_function);
[c_binomial,~,~,~] = optimalCubicInterp...
    (object_function,c_initial,1e-2,1e2,1e-3);

% inverse multiquadric kernal function
basis_function_inverse_multiquadric = @(r,c) 1./sqrt(r+c);
dRM_dc_function = @(x_number,X_dis,rdibas_matrix,c) -0.5*rdibas_matrix.^3;
object_function = @(c) objectFunctionRadiabasis....
    (X_dis,Y_nomlz,x_number,basis_function_inverse_multiquadric,c,dRM_dc_function);

% c_initial = 1;
% drawFunction(object_function,1e-1,10);

[c_inverse_binomial,~,~,~] = optimalCubicInterp...
    (object_function,c_initial,1e-2,1e2,1e-3);

% generate total model
basis_function_list = {
    @(r) basis_function_linear(r,c_linear);
    @(r) basis_function_gauss(r,c_gauss);
    @(r) basis_function_spline(r,c_spline);
    @(r) basis_function_triple(r,c_triple);
    @(r) basis_function_multiquadric(r,c_binomial);
    @(r) basis_function_inverse_multiquadric(r,c_inverse_binomial);};
c_list = [c_linear;c_gauss;c_spline;c_triple;c_binomial;c_inverse_binomial];

model_number = size(basis_function_list,1);
beta_list = zeros(x_number,model_number);
rdibas_matrix_list = zeros(x_number,x_number,model_number);
inv_rdibas_matrix_list = zeros(x_number,x_number,model_number);

% calculate model matrix and error
model_error_list = zeros(model_number,x_number);
for model_index = 1:model_number
    basis_function = basis_function_list{model_index};
    [beta,rdibas_matrix,inv_rdibas_matrix] = interpRadialBasis...
        (X_dis,Y_nomlz,basis_function,x_number);
    beta_list(:,model_index) = beta;
    rdibas_matrix_list(:,:,model_index) = rdibas_matrix;
    inv_rdibas_matrix_list(:,:,model_index) = inv_rdibas_matrix;

    model_error_list(model_index,:) = (beta./...
        diag(inv_rdibas_matrix))';
end

% calculate weight of each model
C = model_error_list*model_error_list';
eta = trace(C)/x_number;
I_model = eye(model_number);
one_model = ones(model_number,1);
w = (C+eta*I_model)\one_model/...
    (one_model'*((C+eta*I_model)\one_model));
while min(w) < -0.05
    % minimal weight cannot less than zero too much
    eta = eta*10;
    w = (C+eta*I_model)\one_model/...
        (one_model'*((C+eta*I_model)\one_model));
end

% initialization predict function
predict_function = @(X_predict) interpEnsemleRadialBasisPredictor...
    (X_predict,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_number,variable_number,model_number,beta_list,basis_function_list,w);

ensemleradialbasis_model.X = X;
ensemleradialbasis_model.Y = Y;
ensemleradialbasis_model.aver_X = aver_X;
ensemleradialbasis_model.stdD_X = stdD_X;
ensemleradialbasis_model.aver_Y = aver_Y;
ensemleradialbasis_model.stdD_Y = stdD_Y;

ensemleradialbasis_model.basis_function_list = basis_function_list;
ensemleradialbasis_model.c_list = c_list;
ensemleradialbasis_model.beta_list = beta_list;
ensemleradialbasis_model.rdibas_matrix_list = rdibas_matrix_list;
ensemleradialbasis_model.inv_rdibas_matrix_list = inv_rdibas_matrix_list;
ensemleradialbasis_model.model_error_list = model_error_list;
ensemleradialbasis_model.w = w;

ensemleradialbasis_model.predict_function = predict_function;

% abbreviation:
% num: number,pred: predict,vari: variable
    function [fval,gradient] = objectFunctionRadiabasis....
            (X_dis,Y,x_number,basis_function,c,dRM_dc_function)
        % MSE_CV function,simple approximation to RMS
        % basis_function input is c and x_sq
        %
        basis_function = @(r) basis_function(r,c);
        [beta__,rdibas_matrix__,inv_rdibas_matrix__] = interpRadialBasis...
            (X_dis,Y,basis_function,x_number);
        U = beta__./diag(inv_rdibas_matrix__);
        fval = sum(U.^2);

        % calculate gradient
        if nargout > 1
            inv_rdibas_matrix_gradient = -inv_rdibas_matrix__*...
                dRM_dc_function...
                (x_number,X_dis,rdibas_matrix__,c)*inv_rdibas_matrix__;
            U_gradient = zeros(x_number,1);
            I = eye(x_number);
            for x_index = 1:x_number
                U_gradient(x_index) = (I(x_index,:)*inv_rdibas_matrix_gradient*Y)/...
                    inv_rdibas_matrix__(x_index,x_index)-...
                    beta__(x_index)*(I(x_index,:)*inv_rdibas_matrix_gradient*I(:,x_index))/...
                    inv_rdibas_matrix__(x_index,x_index)^2;
            end

            gradient = 2*sum(U.*U_gradient);
        end
    end

    function [beta,rdibas_matrix,inv_rdibas_matrix] = interpRadialBasis...
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
        inv_rdibas_matrix = rdibas_matrix\eye(x_number);
        beta = inv_rdibas_matrix*Y;
    end

    function [x_best,favl_best,NFE,output] = optimalCubicInterp...
            (object_function,x_initial,low_bou,up_bou,torlance,iteration_max)
        % cubic interp optimization,should provide fval and gradient
        % only work for one best(convex)
        %
        if nargin < 6
            iteration_max = [];
            if nargin < 5
                torlance = [];
                if nargin < 4
                    up_bou = [];
                    if nargin < 3
                        low_bou = [];
                        if nargin < 2
                            error('lack x initial');
                        end
                    end
                end
            end
        end

        INFORMATION_FLAG = 0; % whether show information

        draw_range = 0.001;
        draw_interval = draw_range*0.02;
        DRAW_FLAG = 0;

        if isempty(iteration_max)
            iteration_max = 10*length(x_initial);
        end

        if isempty(torlance)
            torlance = 1e-6;
        end

        x = x_initial;
        done = 0;
        iteration = 0;
        NFE = 0;
        result_x_list = [];
        result_fval_list = [];

        % decide which turn to search
        [fval,gradient] = object_function(x);NFE = NFE+1;
        result_x_list = [result_x_list;x];
        result_fval_list = [result_fval_list;fval];
        if gradient < -torlance
            direction = 1;
        elseif gradient > torlance
            direction = -1;
        else
            done = 1;
            x_best = x;
            favl_best = fval;
        end

        x_old = x;
        fval_old = fval;
        gradient_old = gradient;
        iteration = iteration+1;

        % move forward to first point
        if ~done
            x = x_old+direction*0.01;
            if x > up_bou
                x = up_bou;
            elseif x < low_bou
                x = low_bou;
            end
            [fval,gradient] = object_function(x);NFE = NFE+1;
            result_x_list = [result_x_list;x];
            result_fval_list = [result_fval_list;fval];
            quit_flag = judgeQuit...
                (x,x_old,fval,fval_old,gradient,torlance,iteration,iteration_max);
            if quit_flag
                done = 1;
                x_best = x;
                favl_best = fval;
            end
            iteration = iteration+1;
        end

        % main loop for cubic interp
        while ~done

            x_base = x_old;
            x_relative = x/x_old;
            interp_matrix = [1,1,1,1;
                3,2,1,0;
                x_relative^3,x_relative^2,x_relative,1;
                3*x_relative^2,2*x_relative,1,0];

            if rcond(interp_matrix) < eps
                disp('error');
            end

            interp_value = [fval_old;gradient_old*x_base;fval;gradient*x_base];
            [x_inter_rel,coefficient_cubic] = minCubicInterpolate(interp_matrix,interp_value);
            x_inter = x_inter_rel*x_base;

            if DRAW_FLAG
                x_draw = 1:direction*draw_interval:direction*draw_range;
                x_draw = x_draw/x_base;
                line(x_draw*x_base,coefficient_cubic(1)*x_draw.^3+coefficient_cubic(2)*x_draw.^2+...
                    coefficient_cubic(3)*x_draw+coefficient_cubic(4));
            end

            % limit search space,process constraints
            if x_inter > up_bou
                x_inter = up_bou;
            elseif x_inter < low_bou
                x_inter = low_bou;
            end

            [fval_inter,gradient_inter] = object_function(x_inter);NFE = NFE+1;

            % only work for one best(convex)
            % three situation discuss
            if gradient < 0
                x_old = x;
                fval_old = fval;
                gradient_old = gradient;
            else
                if gradient_inter < 0
                    x_old = x;
                    fval_old = fval;
                    gradient_old = gradient;
                end
            end

            x = x_inter;
            fval = fval_inter;
            gradient = gradient_inter;

            quit_flag = judgeQuit...
                (x,x_old,fval,fval_old,gradient,torlance,iteration,iteration_max);
            if quit_flag
                done = 1;
                x_best = x;
                favl_best = fval;
            end

            result_x_list = [result_x_list;x];
            result_fval_list = [result_fval_list;fval];
            iteration = iteration+1;
        end
        output.result_x_list = result_x_list;
        output.result_fval_list = result_fval_list;

        function [lamada,coefficient_cubic] = minCubicInterpolate(interpolate_matrix,interpolate_value)
            % calculate min cubic curve
            %
            coefficient_cubic = interpolate_matrix\interpolate_value;

            temp_sqrt = 4*coefficient_cubic(2)^2-12*coefficient_cubic(1)*coefficient_cubic(3);
            if temp_sqrt>=0
                temp_lamada = -coefficient_cubic(2)/3/coefficient_cubic(1)+...
                    sqrt(temp_sqrt)/6/coefficient_cubic(1);
                if (temp_lamada*6*coefficient_cubic(1)+2*coefficient_cubic(2))>0
                    lamada = temp_lamada;
                else
                    lamada = -coefficient_cubic(2)/3/coefficient_cubic(1)-...
                        sqrt(temp_sqrt)...
                        /6/coefficient_cubic(1);
                end
            else
                lamada = -coefficient_cubic(2)/3/coefficient_cubic(1);
            end
        end
        function quit_flag = judgeQuit...
                (x,x_old,fval,fval_old,gradient,torlance,iteration,iteration_max)
            quit_flag = 0;
            if abs(fval-fval_old)/fval_old < torlance
                quit_flag = 1;
            end
            if abs(gradient) < torlance
                quit_flag = 1;
            end
            if abs(x-x_old) < 1e-5
                quit_flag = 1;
            end
            if iteration >= iteration_max
                quit_flag = 1;
            end
        end
    end

    function Y_pred = interpEnsemleRadialBasisPredictor...
            (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,model_number,beta_list,basis_function_list,w)
        % ensemle radial basis function interpolation predict function
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

        % calculate each sub model predict fval and get predict_y
        y_pred_nomlz_list = zeros(x_pred_num,model_number);
        for model_index__ = 1:model_number
            basis_function__ = basis_function_list{model_index__};
            beta__ = beta_list(:,model_index__);
            y_pred_nomlz_list(:,model_index__) = basis_function__(X_dis_pred)*beta__;
        end
        Y_pred_nomlz = y_pred_nomlz_list*w;

        % normalize data
        Y_pred = Y_pred_nomlz*stdD_Y+aver_Y;
    end

end
