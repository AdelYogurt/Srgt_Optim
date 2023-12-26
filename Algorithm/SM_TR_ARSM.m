clc;
clear;
close all hidden;

benchmark = BenchmarkFunction();

benchmark_type = 'single';
% benchmark_name = 'GP';
% benchmark_name = 'Wei';
% benchmark_name = 'PK';
% benchmark_name = 'Forrester';
% benchmark_name = 'PVD4';
% benchmark_name = 'HK';
benchmark_name = 'G06';

[model_function,variable_number,low_bou,up_bou,...
    object_function,A,B,Aeq,Beq,nonlcon_function] = benchmark.getBenchmarkFunction(benchmark_type,benchmark_name);

% x_initial = rand(1,variable_number).*(up_bou-low_bou)+low_bou;
% [x_best,fval_best,~,output] = fmincon(object_function,x_initial,A,B,Aeq,Beq,low_bou,up_bou,[],optimoptions('fmincon','Algorithm','sqp','MaxFunctionEvaluations',10000,'Display','iter-detailed'))

cheapcon_function = [];

%% single run

[x_best,fval_best,NFE,output] = optimalSurrogateTRARSM...
    (model_function,variable_number,low_bou,up_bou,...
    cheapcon_function,50,100)
result_x_best = output.result_x_best;
result_fval_best = output.result_fval_best;

figure(1);
plot(result_fval_best);

%% repeat run

% repeat_number = 10;
% result_fval = zeros(repeat_number,1);
% result_NFE = zeros(repeat_number,1);
% max_NFE = 50;
% for repeat_index = 1:repeat_number
%     [x_best,fval_best,NFE,output] = optimalSurrogateTRARSM...
%         (model_function,variable_number,low_bou,up_bou,...
%         cheapcon_function,max_NFE,100,1e-6,1e-3);
% 
%     result_fval(repeat_index) = fval_best;
%     result_NFE(repeat_index) = NFE;
% end
% 
% fprintf('Fval     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_fval),mean(result_fval),max(result_fval),std(result_fval));
% fprintf('NFE     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_NFE),mean(result_NFE),max(result_NFE),std(result_NFE));
% save([benchmark_name,'_TR_ARSM','.mat']);

%% main
function [x_best,fval_best,NFE,output] = optimalSurrogateTRARSM...
    (model_function,variable_number,low_bou,up_bou,...
    cheapcon_function,....
    NFE_max,iteration_max,torlance,nonlcon_torlance,...
    x_initial_list)
% Trust-Region-Based Adaptive Response Surface optimization algorithm
%
% model_function should output format is [fval,con,coneq]
%
% reference: [1] LONG T, LI X, SHI R, et al., Gradient-Free
% Trust-Region-Based Adaptive Response Surface Method for Expensive
% Aircraft Optimization[J]. AIAA Journal, 2018, 56(2): 862-73.
%
% Copyright 2023 3 Adel
%
if nargin < 10
    x_initial_list = [];
    if nargin < 9 || isempty(nonlcon_torlance)
        nonlcon_torlance = 1e-3;
        if nargin < 8 || isempty(torlance)
            torlance = 1e-6;
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

% if noob user input cheapcon function handle while no constraint
if ~isempty(cheapcon_function)
    [con,coneq] = cheapcon_function(rand(variable_number,1).*(up_bou-low_bou)+low_bou);
    if isempty(con) && isempty(coneq)
        cheapcon_function = [];
    end
end

DRAW_FIGURE_FLAG = 1; % whether draw data
INFORMATION_FLAG = 1; % whether print data
CONVERGENCE_JUDGMENT_FLAG = 1; % whether judgment convergence
WRIRE_FILE_FLAG = 0; % whether write to file

% Latin hypercube sample count
sample_number = (variable_number+1)*(variable_number+2)/2;

% hyper parameter
enlarge_range = 2; % adapt region enlarge parameter
range_max = 0.5;
range_min = 0.01;

% augmented lagrange parameter
lambda_initial = 10;
miu = 1;
miu_max = 1000;
gama = 2;

% max fval when normalize fval,con,coneq
nomlz_fval = 10;

% surrogate add point protect range
protect_range = 1e-5;

% NFE and iteration setting
if isempty(NFE_max)
    NFE_max = 10+10*variable_number;
end

if isempty(iteration_max)
    iteration_max = 20+20*variable_number;
end

done = 0;NFE = 0;iteration = 0;

% trust region updata
low_bou_iter = low_bou;
up_bou_iter = up_bou;

% Step 1
% generate latin hypercube sequence
if isempty(x_initial_list)
    X_updata = getLatinHypercube...
        (sample_number,variable_number,low_bou,up_bou,[],cheapcon_function);
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
X_updata=X_updata(2:end,:);

result_x_best = zeros(iteration_max,variable_number);
result_fval_best = zeros(iteration_max,1);

iteration = iteration+1;

% loop
while ~done
    % Step 2
    % updata data library by X
    [~,~,~,~,~,~,~,NFE_updata]=data_library.dataUpdata(X_updata,0);
    NFE = NFE+NFE_updata;

    % Step 3
    % load data
    % RPS_list is used to interpolation
    [X,Fval,Con,Coneq,Vio,Ks]=data_library.dataLoad(low_bou_iter,up_bou_iter);
    
    % nomalization data with max value
    X_model = X;
    fval_max = mean(abs(Fval),1);
    Fval_model = Fval./fval_max*nomlz_fval;
    if ~isempty(Con)
        con_max = mean(abs(Con),1);
        Con_model = Con./con_max*nomlz_fval;
    else
        Con_model = [];
    end
    if ~isempty(Coneq)
        coneq_max = mean(abs(Coneq),1);
        Coneq_model = Coneq./coneq_max*nomlz_fval;
    else
        Coneq_model = [];
    end
    
    % get surrogate model and function
    % avoid too close point
    select_index_list = 1:size(X_model,1);
    index = 1;
    while index < length(select_index_list)
        select_index = select_index_list(index);
        distance = sum(((X_model(select_index,:)-...
            [X_model(1:select_index-1,:);X_model(select_index+1:end,:)])./...
            (up_bou_iter-low_bou_iter)).^2,2);
        if min(distance) < protect_range^2
            select_index_list(index) = [];
        else
            index = index+1;
        end
    end
    X_model = X_model(select_index_list,:);
    Fval_model = Fval_model(select_index_list,:);
    if ~isempty(Con_model)
        Con_model = Con_model(select_index_list,:);
    end
    if ~isempty(Coneq_model)
        Coneq_model = Coneq_model(select_index_list,:);
    end
    
    % if point too less,add more point
    if size(X_model,1) < (variable_number+1)*(variable_number+2)/2
        % generate latin hypercube sequence
        X_updata = getLatinHypercube...
            (min(NFE_max-NFE,sample_number-size(x_list_exist,1)),variable_number,...
            low_bou_iter,up_bou_iter,X,cheapcon_function);

        % update x_updata_list into data library
        [X_updata,Fval_updata,Con_updata,Coneq_updata,Vio_updata,Ks_updata,repeat_index,NFE_updata]=data_library.dataUpdata(X_updata,0);
        NFE = NFE+NFE_updata;
        
        % normalization data and updata into list
        X_model = [X_model;X_updata];
        Fval_model = [Fval_model;Fval_updata./fval_max*nomlz_fval];
        if ~isempty(Con_model)
            Con_model = [Con_model;Con_updata./con_max*nomlz_fval];
        end
        if ~isempty(Coneq_model)
            Coneq_model = [Coneq_model;Coneq_updata./coneq_max*nomlz_fval];
        end
    end

    % generate surrogate model
    [RSM_fval,RSM_con,RSM_coneq,output_RSM] = getRespSurfModel...
        (X_model,Fval_model,Con_model,Coneq_model);
    object_function_surrogate = output_RSM.object_function_surrogate;
    nonlcon_function_surrogate = output_RSM.nonlcon_function_surrogate;
    
    % generate merit function
    if expensive_nonlcon_flag
        if iteration == 1
            % initialize augmented Lagrange method parameter 
            con_number = size(Con,2);
            coneq_number = size(Coneq,2);
            lambda_con = lambda_initial*ones(1,con_number);
            lambda_coneq = lambda_initial*ones(1,coneq_number);
        end
        % generate merit function fval and add fval
        merit_function = @(x) meritFunction...
            (x,object_function_surrogate,nonlcon_function_surrogate,...
            lambda_con,lambda_coneq,miu);
    end
    
    % Step 4
    % get x_infill
    fmincon_option = optimoptions('fmincon','display','none','algorithm','sqp');
    B = RSM_fval.beta(2:1+variable_number);
    temp = RSM_fval.beta(2+variable_number:end);
    C = diag(temp(1:variable_number)); temp = temp(variable_number+1:end);
    for variable_index = 1:variable_number-1
        C(variable_index,1+variable_index:end) = temp(1:variable_index)'/2;
        C(1+variable_index:end,variable_index) = temp(1:variable_index)/2;
        temp = temp(variable_index:end);
    end
    x_initial = (-C\B)';

    if expensive_nonlcon_flag
        [x_infill,~,~,~] = fmincon...
            (merit_function,x_initial,[],[],[],[],low_bou_iter,up_bou_iter,cheapcon_function,fmincon_option);
        fval_infill_predict = object_function_surrogate(x_infill);
    else
        [x_infill,fval_infill_predict,~,~] = fmincon...
            (object_function_surrogate,x_initial,[],[],[],[],low_bou_iter,up_bou_iter,cheapcon_function,fmincon_option);
    end
    
    % check x_infill if exist in data library
    % if not,updata data libraray
    [x_infill,fval_infill,con_infill,coneq_infill,vio_infill,ks_infill,repeat_index,NFE_updata]=data_library.dataUpdata(x_infill,protect_range);
    NFE = NFE+NFE_updata;
    
    [X,Fval,Con,Coneq,Vio,Ks]=data_library.dataLoad();

    % process error
    if isempty(x_infill)
        x_infill = X(repeat_index,:);
        fval_infill = Fval(repeat_index,:);
        if ~isempty(con_list)
            con_infill = Con(repeat_index,:);
        end
        if ~isempty(coneq_list)
            coneq_infill = Coneq(repeat_index,:);
        end
        if ~isempty(vio_list)
            vio_infill = Vio(repeat_index,:);
        end
    end

    % updata penalty factor and lagrangian
    if expensive_nonlcon_flag
        lambda_con = lambda_con+2*miu*max(con_infill,-lambda_con/2/miu);
        lambda_coneq = lambda_coneq+2*miu*coneq_infill;
        if miu < miu_max
            miu = gama*miu;
        else
            miu = miu_max;
        end
    end

    if DRAW_FIGURE_FLAG
        interpVisualize(RSM_fval,low_bou_iter,up_bou_iter);
        if length(x_infill) < 2
            line(x_infill(1),fval_infill/fval_max*nomlz_fval,'Marker','o','Color','r');
        else
            line(x_infill(1),x_infill(2),fval_infill/fval_max*nomlz_fval,'Marker','o','Color','r');
        end
    end
    
    % Step 5
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
                abs((fval_infill-fval_infill_old)/fval_infill_old) < torlance)
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
    
    % Step 6
    % TR guideline first iteration
    if iteration == 2
        bou_range_nomlz = 0.5;
    else
        bou_range_nomlz_old = bou_range_nomlz;
        
        r = (fval_infill_old-fval_infill)/(fval_infill_old-fval_infill_predict);
        x_dis = norm((x_infill_old-x_infill)./(up_bou-low_bou),2);
        if x_dis <= range_min
            x_dis = 0.1;
        end
        
        % scale only can be in (range_max-range_min)
        scale = enlarge_range*x_dis;
        % add rand influence avoid stable
        if scale > (range_max-range_min)
            scale = (range_max-range_min);
        end

        % mapping r into range_min - min(enlarge_range*x_dis,(range_max-range_min))
        % bou_range_nomlz = scale/2 while r = 0
        bou_range_nomlz = scale/(1+exp(-(r-0.5)))+range_min;
        
        if abs(bou_range_nomlz-bou_range_nomlz_old) < torlance
           bou_range_nomlz = bou_range_nomlz*(1+rand()*range_min); 
        end
    end
    bou_range = bou_range_nomlz.*(up_bou-low_bou);
    
    % updata trust range
    low_bou_iter = x_best-bou_range;
    low_bou_iter = max(low_bou_iter,low_bou);
    up_bou_iter = x_best+bou_range;
    up_bou_iter = min(up_bou_iter,up_bou);
    
    % Step 7
    % check whether exist data
    x_list_exist=data_library.dataLoad(low_bou_iter,up_bou_iter);
    
    % generate latin hypercube sequence
    X_updata = getLatinHypercube...
        (min(NFE_max-NFE,sample_number-size(x_list_exist,1)),variable_number,...
        low_bou_iter,up_bou_iter,x_list_exist,cheapcon_function);
    
    x_infill_old = x_infill;
    fval_infill_old = fval_infill;
    fval_best_old = fval_best;
end

result_x_best = result_x_best(1:iteration-1,:);
result_fval_best = result_fval_best(1:iteration-1);

output.result_x_best = result_x_best;
output.result_fval_best = result_fval_best;

    function fval = meritFunction...
            (x,object_function,nonlcon_function,lambda_con,lambda_coneq,miu)
        % penalty function
        % augmented lagrange multiplier method was used
        %
        fval = object_function(x);
        if ~isempty(nonlcon_function)
            [con__,coneq__] = nonlcon_function(x);
            if ~isempty(con__)
                psi = max(con__,-lambda_con/2/miu);
                fval = fval+sum(lambda_con.*psi+miu*psi.*psi);
            end
            if ~isempty(coneq__)
                fval = fval+sum(lambda_coneq.*coneq__+miu*coneq__.*coneq__);
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

function [x_best,fval_best,con_best,coneq_best] = findMinRaw...
    (x_list,fval_list,con_list,coneq_list,...
    cheapcon_function,nonlcon_torlance)
% find min fval in raw data
% x_list,rank is variable
% con_list,rank is con
% coneq_list,rank is coneq
% function will find min fval in con == 0
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

%% surrogate
function [respsurf_model_fval,respsurf_model_con,respsurf_model_coneq,output] = getRespSurfModel...
    (x_list,fval_list,con_list,coneq_list)
% base on library_data to create respsurf model and function
% if input model,function will updata model
% object_function is single fval output
% nonlcon_function is normal nonlcon_function which include con,coneq
% con is colume vector,coneq is colume vector
% var_function is same
%

[predict_function_fval,respsurf_model_fval] = interpRespSurfPreModel...
    (x_list,fval_list);

if ~isempty(con_list)
    predict_function_con = cell(size(con_list,2),1);
    respsurf_model_con = struct('X',[],'Y',[],...
        'beta',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],...
        'predict_function',[]);
    respsurf_model_con = repmat(respsurf_model_con,[size(con_list,2),1]);
    for con_index = 1:size(con_list,2)
        [predict_function_con{con_index},respsurf_model_con(con_index)] = interpRespSurfPreModel...
            (x_list,con_list(:,con_index));
    end
else
    predict_function_con = [];
    respsurf_model_con = [];
end

if ~isempty(coneq_list)
    predict_function_coneq = cell(size(coneq_list,2),1);
    respsurf_model_coneq = struct('X',[],'Y',[],...
        'beta',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],...
        'predict_function',[]);
    respsurf_model_coneq = repmat(respsurf_model_coneq,[size(coneq_list,2),1]);
    for coneq_index = 1:size(coneq_list,2)
        [predict_function_coneq{coneq_index},respsurf_model_coneq(coneq_index)] = interpRespSurfPreModel...
            (x_list,coneq_list(:,coneq_index));
    end
else
    predict_function_coneq = [];
    respsurf_model_coneq = [];
end

object_function_surrogate = @(X_predict) objectFunctionSurrogate(X_predict,predict_function_fval);
if isempty(respsurf_model_con) && isempty(respsurf_model_coneq)
    nonlcon_function_surrogate = [];
else
    nonlcon_function_surrogate = @(X_predict) nonlconFunctionSurrogate(X_predict,predict_function_con,predict_function_coneq);
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

function [predict_function,respsurf_model] = interpRespSurfPreModel(X,Y)
% polynomial response surface interpolation pre model function
%
% input data will be normalize by average and standard deviation of data
%
% input:
% X,Y(initial data,which are real data,x_number x variable_number matrix)
%
% output:
% predict_function,respond surface model(include X,Y,base_function,...)
%
% Copyright 2022 Adel
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

beta = interpRespSurf(X_nomlz,Y_nomlz,x_number,variable_number);

% initialization predict function
predict_function = @(X_predict) interpolationRespSurfPredictor...
    (X_predict,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_number,variable_number,beta);

respsurf_model.X = X;
respsurf_model.Y = Y;
respsurf_model.beta = beta;

respsurf_model.aver_X = aver_X;
respsurf_model.stdD_X = stdD_X;
respsurf_model.aver_Y = aver_Y;
respsurf_model.stdD_Y = stdD_Y;

respsurf_model.predict_function = predict_function;

% abbreviation:
% num: number,pred: predict,vari: variable
    function beta = interpRespSurf(X,Y,x_num,vari_num)
        % interpolation polynomial responed surface core function
        % calculation beta
        %
        X_cross = zeros(x_num,(vari_num-1)*vari_num/2);

        cross_index = 1;
        for i_index = 1:vari_num
            for j_index = i_index+1:vari_num
                X_cross(:,cross_index) = X(:,i_index).*X(:,j_index);
                cross_index = cross_index+1;
            end
        end
        X_inter = [ones(x_num,1),X,X.^2,X_cross];
        
        X_inter_X_inter = X_inter'*X_inter;
        beta = X_inter_X_inter\X_inter'*Y;
    end

    function [Y_pred] = interpolationRespSurfPredictor...
            (X_pred,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,beta)
        % polynomial response surface interpolation predict function
        % input predict_x and respsurf_model model
        % predict_x is row vector
        % output the predict value
        %
        [x_pred_num,~] = size(X_pred);

        % normalize data
        X_pred = (X_pred-aver_X)./stdD_X;
        
        % predict value
        X_cross = zeros(x_pred_num,(vari_num-1)*vari_num/2);
        cross_index = 1;
        for i_index = 1:vari_num
            for j_index = i_index+1:vari_num
                X_cross(:,cross_index) = X_pred(:,i_index).*X_pred(:,j_index);
                cross_index = cross_index+1;
            end
        end
        X_pred_inter = [ones(x_pred_num,1),X_pred,X_pred.^2,X_cross];
        
        % predict variance
        Y_pred = X_pred_inter*beta;
        
        % normalize data
        Y_pred = Y_pred*stdD_Y+aver_Y;
    end

end

%% LHD
function [X_sample,dist_min_nomlz,X_total] = getLatinHypercube...
    (sample_number,variable_number,...
    low_bou,up_bou,X_exist,cheapcon_function)
% generate latin hypercube design
%
% ESLHS method is used(sample and iteration)
%
% election combination mode of point and find best combination
%
% input:
% sample number(new point to sample),variable_number
% low_bou,up_bou,x_exist_list
%
% output:
% X_sample,dist_min_nomlz(min distance of normalize data)
% X_total,include all data in area
%
% reference: [1] LONG T, LI X, SHI R, et al., Gradient-Free
% Trust-Region-Based Adaptive Response Surface Method for Expensive
% Aircraft Optimization[J]. AIAA Journal, 2018, 56(2): 862-73.
%
% Copyright 2022 Adel
%
if nargin < 6
    cheapcon_function = [];
    if nargin < 5
        X_exist = [];
        if nargin < 4
            up_bou = ones(1,variable_number);
            if nargin < 3
                low_bou = zeros(1,variable_number);
                if nargin < 2
                    error('getLatinHypercube: lack variable_number');
                end
            end
        end
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
    X_total = [];
    X_sample = [];
    dist_min_nomlz = [];
    return;
end

iteration_max = min(100*sample_number,100);

exist_number = size(X_exist,1);
total_number = sample_number+exist_number;
if sample_number <= 0
    X_total = X_exist;
    X_sample = [];
    dist_min_nomlz = getMinDistance(X_exist_nomlz);
    return;
end

% get quasi-feasible point
x_initial_number = 100*sample_number;
x_quasi_number = 10*sample_number;
if ~isempty(cheapcon_function)
    X_supply_quasi_nomlz = [];

    % check if have enough X_supply_nomlz
    iteration = 0;
    while size(X_supply_quasi_nomlz,1) < x_quasi_number && iteration < 100
        X_supply_initial_nomlz = lhsdesign(x_initial_number,variable_number);

        qusai_index = [];
        for x_index = 1:size(X_supply_initial_nomlz,1)
            if cheapcon_function(X_supply_initial_nomlz(x_index,:).*(up_bou-low_bou)+low_bou) <= 0
                qusai_index = [qusai_index,x_index];
            end
        end
        X_supply_quasi_nomlz = [X_supply_quasi_nomlz;X_supply_initial_nomlz(qusai_index,:)];

        iteration = iteration+1;
    end

    if iteration == 100 && isempty(X_supply_quasi_nomlz)
        error('getLatinHypercube: feasiable quasi point cannot be found');
    end
else
    X_supply_quasi_nomlz = lhsdesign(x_quasi_number,variable_number);
end

% iterate and get final x_supply_list
iteration = 0;
x_supply_quasi_number = size(X_supply_quasi_nomlz,1);
dist_min_nomlz = 0;
X_sample_nomlz = [];

% dist_min_nomlz_result = zeros(1,iteration);
while iteration <= iteration_max
    % random select x_new_number X to X_trial_nomlz
    x_select_index = randperm(x_supply_quasi_number,sample_number);
    
    % get distance min itertion X_
    distance_min_iteration = getMinDistanceIter...
        (X_supply_quasi_nomlz(x_select_index,:),X_exist_nomlz);
    
    % if distance_min_iteration is large than last time
    if distance_min_iteration > dist_min_nomlz
        dist_min_nomlz = distance_min_iteration;
        X_sample_nomlz = X_supply_quasi_nomlz(x_select_index,:);
    end
    
    iteration = iteration+1;
%     dist_min_nomlz_result(iteration) = dist_min_nomlz;
end
dist_min_nomlz = getMinDistance([X_sample_nomlz;X_exist_nomlz]);
X_sample = X_sample_nomlz.*(up_bou-low_bou)+low_bou;
X_total = [X_exist;X_sample];

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
        % this version do not consider distance between x exist
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
