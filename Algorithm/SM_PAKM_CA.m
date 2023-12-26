clc;
clear;
close all hidden;

benchmark=BenchmarkFunction();

benchmark_type='single';
% benchmark_name='GP';
% benchmark_name='Wei';
% benchmark_name='PK';
% benchmark_name='EP20';
% benchmark_name='Forrester';
% benchmark_name='PVD4';
benchmark_name='G01';
% benchmark_name='G06';

[model_function,variable_number,low_bou,up_bou,...
    object_function,A,B,Aeq,Beq,nonlcon_function]=benchmark.getBenchmarkFunction(benchmark_type,benchmark_name);

% x_initial=rand(1,variable_number).*(up_bou-low_bou)+low_bou;
% [x_best,fval_best,~,output]=fmincon(object_function,x_initial,A,B,Aeq,Beq,low_bou,up_bou,[],optimoptions('fmincon','Algorithm','sqp','MaxFunctionEvaluations',10000,'Display','iter-detailed'))

cheapcon_function=[];

%% single run

[x_best,fval_best,NFE,output]=optimalSurrogatePAKMCA...
    (model_function,variable_number,low_bou,up_bou,...
    cheapcon_function,200,500)
result_x_best=output.result_x_best;
result_fval_best=output.result_fval_best;

figure(1);
plot(result_fval_best);

%% repeat run

% repeat_number=10;
% result_fval=zeros(repeat_number,1);
% result_NFE=zeros(repeat_number,1);
% max_NFE=200;
% for repeat_index=1:repeat_number
%     [x_best,fval_best,NFE,output]=optimalSurrogatePAKMCA...
%         (model_function,variable_number,low_bou,up_bou,...
%         cheapcon_function,max_NFE,300,1e-6,1e-3);
% 
%     result_fval(repeat_index)=fval_best;
%     result_NFE(repeat_index)=NFE;
% end
% 
% fprintf('Fval     : lowest=%4.4f,mean=%4.4f,worst=%4.4f,std=%4.4f \n',min(result_fval),mean(result_fval),max(result_fval),std(result_fval));
% fprintf('NFE     : lowest=%4.4f,mean=%4.4f,worst=%4.4f,std=%4.4f \n',min(result_NFE),mean(result_NFE),max(result_NFE),std(result_NFE));
% save([benchmark_name,'_PAKM_CA','.mat']);

%% main
function [x_best,fval_best,NFE,output]=optimalSurrogatePAKMCA ...
    (model_function,variable_number,low_bou,up_bou,...
    cheapcon_function,...
    NFE_max,iteration_max,torlance,nonlcon_torlance,...
    x_initial_list)
% Parallel Adaptive Kriging Method with Constraint Aggregation
% only support constraints problem
%
% x_list is x_number x variable_number matrix
% both nonlcon_function and cheapcon_function format is [con,coneq]
% model_function should output fval,format is [fval,con,coneq]
% con or coneq can be colume vector if there was more than one constrain
%
% referance: LONG T,WEI Z,SHI R,et al. Parallel Adaptive Kriging Method
% with Constraint Aggregation for Expensive Black-Box Optimization Problems
% [J]. AIAA Journal,2021,59(9): 3465-79.
%
% Copyright Adel 2023.2
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
    cheapcon_function=[];
end

DRAW_FIGURE_FLAG=0; % whether draw data
INFORMATION_FLAG=1; % whether print data
CONVERGENCE_JUDGMENT_FLAG=0; % whether judgment convergence
WRIRE_FILE_FLAG=0; % whether write to file

if isempty(iteration_max)
    iteration_max=100;
end

% hyper parameter
if variable_number < 10
    sample_number_initial=min((variable_number+1)*(variable_number+2)/2,5*variable_number);
else
    sample_number_initial=variable_number+1;
end
if variable_number <= 5
    sample_number_iteration=2;
else
    sample_number_iteration=3;
end
rou=4;
rou_min=1;
rou_max=64;
rou_decrease=0.5;
rou_increase=2;

% max fval when normalize fval,con,coneq
nomlz_fval=10;

% surrogate add point protect range
protect_range=1e-5;

% NFE and iteration setting
if isempty(NFE_max)
    NFE_max=10+10*variable_number;
end

if isempty(iteration_max)
    iteration_max=20+20*variable_number;
end

done=0;NFE=0;iteration=0;

% step 2
% generate initial sample X
if isempty(x_initial_list)
    X_updata=lhsdesign(sample_number_initial,variable_number).*(up_bou-low_bou)+low_bou;
else
    X_updata=x_initial_list;
end

% detech expensive constraints and initializa data library
data_library=DataLibrary(model_function,variable_number,low_bou,up_bou,...
    nonlcon_torlance,[],WRIRE_FILE_FLAG);

% detech expensive constraints and initialize data library
% updata data library by X
[~,~,~,~,~,~,~,NFE_updata]=data_library.dataUpdata(X_updata(1,:),0);
NFE=NFE+NFE_updata;
X_updata=X_updata(2:end,:);

result_x_best=zeros(iteration_max,variable_number);
result_fval_best=zeros(iteration_max,1);

iteration=iteration+1;

KRG_model_fval=[];
KRG_model_con=[];
KRG_model_coneq=[];
KRG_model_KS=[];

while ~done
    % step 3
    % updata data library by x_list
    [~,~,~,~,~,~,~,NFE_updata]=data_library.dataUpdata(X_updata,0);
    NFE=NFE+NFE_updata;

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
    [KRG_model_fval,KRG_model_con,KRG_model_coneq,output_kriging]=getKrigingModel...
        (X_model,Fval_model,Con_model,Coneq_model,...
        KRG_model_fval,KRG_model_con,KRG_model_coneq);
    object_function_surrogate=output_kriging.object_function_surrogate;
    nonlcon_function_surrogate=output_kriging.nonlcon_function_surrogate;

    % step 5
    % MSP guideline to obtain x_adapt
    [x_infill,~,exitflag,~]=findMinMSP...
        (object_function_surrogate,variable_number,low_bou,up_bou,nonlcon_function_surrogate,...
        cheapcon_function);

    if exitflag == -2
        % optimal feasiblilty if do not exist feasible point
        object_nonlcon_function_surrogate=@(x) objectNonlconFunctionSurrogate(x,nonlcon_function_surrogate);
        [x_infill,~,exitflag,~]=findMinMSP...
            (object_nonlcon_function_surrogate,variable_number,low_bou,up_bou,[],...
            cheapcon_function);
    end

    % updata infill point
    [x_infill,fval_infill,con_infill,coneq_infill,vio_infill,ks_infill,repeat_index,NFE_updata] = ...
        data_library.dataUpdata(x_infill,protect_range);
    NFE = NFE+NFE_updata;

    % process error
    if isempty(x_infill)
        % continue;
        x_infill = X(repeat_index,:);
        fval_infill = Fval(repeat_index,:);
        if ~isempty(Con)
            con_infill = Con(repeat_index,:);
        end
        if ~isempty(Coneq)
            coneq_infill = Coneq(repeat_index,:);
        end
        if ~isempty(Vio)
            vio_infill = Vio(repeat_index,:);
        end
    end

    if DRAW_FIGURE_FLAG && variable_number < 3
        interpVisualize(KRG_model_fval,low_bou,up_bou);
        line(x_infill(1),x_infill(2),fval_infill/fval_max*nomlz_fval,'Marker','o','color','r')
    end

    % step 6
    % find best result to record
    [x_best,fval_best,con_best,coneq_best]=findMinRaw...
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

    result_x_best(iteration,:)=x_best;
    result_fval_best(iteration,:)=fval_best;
    iteration=iteration+1;

    % forced interrupt
    if iteration > iteration_max || NFE >= NFE_max
        done=1;
    end

    % convergence judgment
    if CONVERGENCE_JUDGMENT_FLAG
        if (iteration > 2 && ...
                abs((fval_infill-fval_potential_old)/fval_potential_old) < torlance)
            done=1;
            if ~isempty(con_best)
                if sum(con_best > nonlcon_torlance)
                    done=0;
                end
            end
            if ~isempty(coneq_best)
                if sum(abs(coneq_best) > nonlcon_torlance)
                    done=0;
                end
            end
        end
    end

    % PCFEI Function-Based Infill Sampling Mechanism
    if ~done
        % step 1
        % construct kriging model of KS function
        % updata rou
        if (max(con_infill) < nonlcon_torlance)
            rou=rou*rou_increase;
        else
            rou=rou*rou_decrease;
        end
        rou=max(rou,rou_min);
        rou=min(rou,rou_max);

        % modify
        [X,Fval,Con,Coneq,Vio,Ks]=data_library.dataLoad;
        X_model = X;
        fval_max = max(abs(Fval),[],1);
        Fval_model=Fval/fval_max*nomlz_fval;
        if ~isempty(Ks)
            ks_maX = max(abs(Ks),[],1);
            Ks_model = Ks./ks_maX*nomlz_fval;
        else
            Ks_model = [];
        end

%         Ks_model=log(sum(exp(con_nomlz_list*rou),2))/rou;
        [KRG_model_KS,~,~,output]=getKrigingModel...
            (X_model,Ks_model,[],[],...
            KRG_model_KS);
        ks_function_surrogate=output.object_function_surrogate;

        % step 2
        % contruct EI,PF function
        object_function_EI=@(X) EIFunction(object_function_surrogate,X,min(Fval_model));
        object_function_PF=@(X) PFFunction(ks_function_surrogate,X);
        object_function_IF=@(X) IFFunction(x_infill,X,exp(KRG_model_fval.hyp),variable_number);

        % step 3
        % multi objective optimization to get pareto front
        object_function_PCFEI=@(x) [-object_function_EI(x),-object_function_PF(x)];
        gamultiobj_option=optimoptions('gamultiobj','Display','none');
        [x_pareto_list,fval_pareto_list,exitflag,output_gamultiobj]=gamultiobj...'
            (object_function_PCFEI,variable_number,[],[],[],[],low_bou,up_bou,[],gamultiobj_option);

        % step 4
        % base on PCFEI value to get first sample_number_iteration point
        if size(x_pareto_list,1) < sample_number_iteration
            X_updata = x_pareto_list;
        else
            EI_list=-fval_pareto_list(:,1);
            EI_list=EI_list/max(EI_list);
            PF_list=-fval_pareto_list(:,2);
            PF_list=PF_list/max(PF_list);
            IF_list=object_function_IF(x_pareto_list);
            IF_list=IF_list/max(IF_list);
            PCFEI_list=EI_list.*PF_list.*IF_list;
            [~,index_list]=sort(PCFEI_list);
            
            X_updata=x_pareto_list(index_list((end+1-sample_number_iteration):end),:);
        end
    end

    x_potential_old=x_infill;
    fval_potential_old=fval_infill;
    fval_best_old=fval_best;
end
result_x_best=result_x_best(1:iteration-1,:);
result_fval_best=result_fval_best(1:iteration-1);

output.result_x_best=result_x_best;
output.result_fval_best=result_fval_best;

    function fval=objectNonlconFunctionSurrogate(x,nonlcon_function_surrogate)
        [con__,coneq__]=nonlcon_function_surrogate(x);
        fval=0;
        if ~isempty(con__)
            fval=fval+sum(max(con__,0).^2);
        end
        if ~isempty(coneq__)
            fval=fval+sum(max(con__,0).^2);
        end
    end
end

%% auxiliary function
function [x_best,fval_best,exitflag,output]=findMinMSP...
    (object_function_surrogate,variable_number,low_bou,up_bou,nonlcon_function_surrogate,...
    cheapcon_function)
% find min fval use MSP guideline
% MSP: object_funtion is object_function (generate by surrogate model)
% nonlcon_function generate by surrogate model
% use ga as optimal method
%
if nargin < 6
    cheapcon_function=[];
    if nargin < 5
        nonlcon_function_surrogate=[];
        if nargin < 4
            up_bou=[];
            if nargin < 3
                low_bou=[];
            end
        end
    end
end
% % object function convert to penalty function if input nonlcon function
% if ~isempty(nonlcon_function)
%     object_function=@(x) penaltyFunction(object_function,x,nonlcon_function);
%     constraint_function=cheapcon_function;
% end

% obtian total constraint function
if ~isempty(nonlcon_function_surrogate) || ~isempty(cheapcon_function)
    constraint_function=@(x) totalconFunction...
        (x,nonlcon_function_surrogate,cheapcon_function);
else
    constraint_function=[];
end

% generate initial population for ga
population_matrix=zeros(min(10,2*variable_number),variable_number);
for population_index=1:size(population_matrix,1)
    x=rand(1,variable_number).*(up_bou-low_bou)+low_bou;
    if ~isempty(cheapcon_function)
        [con,coneq]=cheapcon_function(x);
        while sum([~(con < 0);abs(coneq) < 0])
            x=rand(1,variable_number).*(up_bou-low_bou)+low_bou;
            [con,coneq]=cheapcon_function(x);
        end
    end
    population_matrix(population_index,:)=x;
end

% optiaml
ga_option=optimoptions('ga','FunctionTolerance',1e-2,'ConstraintTolerance',1e-2,...
    'PopulationSize',min(10,2*variable_number),...
    'MaxGenerations',50,'InitialPopulationMatrix',population_matrix,...
    'display','none');
[x_best,fval_best,exitflag,output]=ga...
    (object_function_surrogate,variable_number,[],[],[],[],low_bou,up_bou,constraint_function,ga_option);
fmincon_option=optimoptions('fmincon','FunctionTolerance',1e-6,'ConstraintTolerance',1e-6,...
    'algorithm','sqp',....
    'display','none');
[x_best,fval_best,exitflag,output]=fmincon...
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

function [x_best,fval_best,con_best,coneq_best]=findMinRaw...
    (x_list,fval_list,con_list,coneq_list,...
    cheapcon_function,nonlcon_torlance)
% find min fval in raw data
% x_list,rank is variable
% con_list,rank is con
% coneq_list,rank is coneq
% function will find min fval in con==0
% if there was not feasible x,find min consum
%
con_best=[];
coneq_best=[];
max_nonlcon_list=zeros(size(x_list,1),1);
max_cheapcon_list=zeros(size(x_list,1),1);
% process expendsive con
if ~isempty(con_list)
    max_nonlcon_list=max(con_list,[],2);
end
if ~isempty(coneq_list)
    max_nonlcon_list=max(abs(coneq_list),[],2);
end

% add cheap con
for x_index=1:size(x_list,1)
    if ~isempty(cheapcon_function)
        [con,coneq]=cheapcon_function(x_list(x_index,:));
        max_cheapcon_list(x_index)=max_cheapcon_list(x_index)+...
            sum(max(con,0))+sum(coneq.*coneq);
    end
end

con_judge_list=(max_nonlcon_list > nonlcon_torlance)+...
    (max_cheapcon_list > 0);
index=find(con_judge_list == 0);
if ~isempty(index)
    % feasible x
    x_list=x_list(index,:);
    fval_list=fval_list(index);
    if ~isempty(con_list)
        con_list=con_list(index,:);
    end
    if ~isempty(coneq_list)
        coneq_list=coneq_list(index,:);
    end

    % min fval
    [fval_best,index_best]=min(fval_list);
    x_best=x_list(index_best,:);
    if ~isempty(con_list)
        con_best=con_list(index_best,:);
    end
    if ~isempty(coneq_list)
        coneq_best=coneq_list(index_best,:);
    end
else
    % min consum
    [~,index_best]=min(max_nonlcon_list);
    fval_best=fval_list(index_best);
    x_best=x_list(index_best,:);
    if ~isempty(con_list)
        con_best=con_list(index_best,:);
    end
    if ~isempty(coneq_list)
        coneq_best=coneq_list(index_best,:);
    end
end
end

function fval=EIFunction(object_function_surrogate,X,fval_min)
% EI function
[Fval_pred,Fval_var]=object_function_surrogate(X);
normal_fval=(fval_min-Fval_pred)./sqrt(Fval_var);
EI_l=(fval_min-Fval_pred).*normcdf(normal_fval);
EI_g=Fval_var.*normpdf(normal_fval);
fval=EI_l+EI_g;
end

function fval=PFFunction(object_function_surrogate,X)
% PF function
[Con_pred,Con_var]=object_function_surrogate(X);
fval=normcdf(-Con_pred./sqrt(Con_var));
end

function fval=IFFunction(x_best,X,theta,variable_number)
fval=zeros(size(X,1),1);
for variable_index=1:variable_number
    fval=fval+(X(:,variable_index)-x_best(:,variable_index)').^2*theta(variable_index);
end
fval=1-exp(-fval);
end

%% surrogate model
function [kriging_model_fval,kriging_model_con,kriging_model_coneq,output]=getKrigingModel...
    (x_list,fval_list,con_list,coneq_list,...
    kriging_model_fval,kriging_model_con,kriging_model_coneq)
% base on library_data to create kriging model and function
% if input model,function will updata model
% object_function is multi fval output
% nonlcon_function is normal nonlcon_function which include con,coneq
% con is colume vector,coneq is colume vector
% var_function is same
%
if size(x_list,1) ~= size(fval_list,1)
    error('getKrigingModel: x_list size no equal fval_list size')
end

if isempty(kriging_model_fval)
    [predict_function_fval,kriging_model_fval]=interpKrigingPreModel...
        (x_list,fval_list);
else
    [predict_function_fval,kriging_model_fval]=interpKrigingPreModel...
        (x_list,fval_list,kriging_model_fval.hyp);
end

if ~isempty(con_list)
    predict_function_con=cell(size(con_list,2),1);
    if size(x_list,1) ~= size(con_list,1)
        error('getKrigingModel: x_list size no equal con_list size')
    end
    if isempty(kriging_model_con)
        kriging_model_con=struct('X',[],'Y',[],...
            'fval_regression',[],'covariance',[],'inv_covariance',[],...
            'hyp',[],'beta',[],'gama',[],'sigma_sq',[],...
            'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],...
            'predict_function',[]);
        kriging_model_con=repmat(kriging_model_con,1,[size(con_list,2)]);
        for con_index=1:size(con_list,2)
            [predict_function_con{con_index},kriging_model_con(con_index)]=interpKrigingPreModel...
                (x_list,con_list(:,con_index));
        end
    else
        for con_index=1:size(con_list,2)
            [predict_function_con{con_index},kriging_model_con(con_index)]=interpKrigingPreModel...
                (x_list,con_list(:,con_index),kriging_model_con(con_index).hyp);
        end
    end
else
    predict_function_con=[];
    kriging_model_con=[];
end

if ~isempty(coneq_list)
    predict_function_coneq=cell(size(coneq_list,2),1);
    if size(x_list,1) ~= size(coneq_list,1)
        error('getKrigingModel: x_list size no equal coneq_list size')
    end
    if isempty(kriging_model_coneq)
        kriging_model_coneq=struct('X',[],'Y',[],...
            'fval_regression',[],'covariance',[],'inv_covariance',[],...
            'hyp',[],'beta',[],'gama',[],'sigma_sq',[],...
            'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],...
            'predict_function',[]);
        kriging_model_coneq=repmat(kriging_model_coneq,1,[size(coneq_list,2)]);
        for coneq_index=1:size(coneq_list,2)
            [predict_function_coneq{coneq_index},kriging_model_coneq(coneq_index)]=interpKrigingPreModel...
                (x_list,coneq_list(:,coneq_index));
        end
    else
        for coneq_index=1:size(coneq_list,2)
            [predict_function_coneq{coneq_index},kriging_model_coneq(coneq_index)]=interpKrigingPreModel...
                (x_list,coneq_list(:,coneq_index),kriging_model_coneq(coneq_index).hyp);
        end
    end
else
    predict_function_coneq=[];
    kriging_model_coneq=[];
end

object_function_surrogate=@(X_predict) objectFunctionSurrogate(X_predict,predict_function_fval);
if isempty(predict_function_con) && isempty(predict_function_coneq)
    nonlcon_function_surrogate=[];
else
    nonlcon_function_surrogate=@(X_predict) nonlconFunctionSurrogate(X_predict,predict_function_con,predict_function_coneq);
end

output.object_function_surrogate=object_function_surrogate;
output.nonlcon_function_surrogate=nonlcon_function_surrogate;

    function [fval,fval_var]=objectFunctionSurrogate...
            (X_predict,predict_function_fval)
        % connect all predict favl
        %
        [fval,fval_var]=predict_function_fval(X_predict);
    end

    function [con,con_var,coneq,coneq_var]=nonlconFunctionSurrogate...
            (X_predict,predict_function_con,predict_function_coneq)
        % connect all predict con and coneq
        %
        if isempty(predict_function_con)
            con=[];
            con_var=[];
        else
            con=zeros(size(X_predict,1),length(predict_function_con));
            con_var=zeros(size(X_predict,1),length(predict_function_con));
            for con_index__=1:length(predict_function_con)
                [con(:,con_index__),con_var(:,con_index__)]=....
                    predict_function_con{con_index__}(X_predict);
            end
        end
        if isempty(predict_function_coneq)
            coneq=[];
            coneq_var=[];
        else
            coneq=zeros(size(X_predict,1),length(predict_function_coneq));
            coneq_var=zeros(size(X_predict,1),length(predict_function_coneq));
            for coneq_index__=1:length(predict_function_coneq)
                [coneq(:,coneq_index__),coneq_var(:,coneq_index__)]=...
                    predict_function_coneq{coneq_index__}(X_predict);
            end
        end
    end
end

function [predict_function,kriging_model]=interpKrigingPreModel...
    (X,Y,hyp)
% nomalization method is grassian
% add multi x_predict input support
% prepare model,optimal theta and calculation parameter
% X,Y are x_number x variable_number matrix
% aver_X,stdD_X is 1 x x_number matrix
% theta beta gama sigma_sq is normalizede,so predict y is normalize
% theta=exp(hyp)
%
% input initial data X,Y,which are real data
%
% output is a kriging model,include predict_function...
% X,Y,base_function_list
%
% Copyright 2023.2 Adel
%
[x_number,variable_number]=size(X);
if nargin < 3
    hyp=zeros(1,variable_number);
end

% normalize data
aver_X=mean(X);
stdD_X=std(X);
aver_Y=mean(Y);
stdD_Y=std(Y);
index__=find(stdD_X == 0);
if  ~isempty(index__),stdD_X(index__)=1; end
index__=find(stdD_Y == 0);
if  ~isempty(index__),stdD_Y(index__)=1; end
X_nomlz=(X-aver_X)./stdD_X;
Y_nomlz=(Y-aver_Y)./stdD_Y;

% initial X_dis_sq
X_dis_sq=zeros(x_number,x_number,variable_number);
for variable_index=1:variable_number
    X_dis_sq(:,:,variable_index)=...
        (X_nomlz(:,variable_index)-X_nomlz(:,variable_index)').^2;
end

% regression function define
% notice reg_function process no normalization data
% reg_function=@(X) regZero(X);
reg_function=@(X) regLinear(X);

% calculate reg
fval_reg_nomlz=(reg_function(X)-aver_Y)./stdD_Y;

% optimal to get hyperparameter
fmincon_option=optimoptions('fmincon','Display','none',...
    'OptimalityTolerance',1e-2,...
    'FiniteDifferenceStepSize',1e-5,...,
    'MaxIterations',10,'SpecifyObjectiveGradient',false);
low_bou_hyp=-3*ones(1,variable_number);
up_bou_hyp=3*ones(1,variable_number);
object_function_hyp=@(hyp) objectNLLKriging...
    (X_dis_sq,Y_nomlz,x_number,variable_number,hyp,fval_reg_nomlz);

% [fval,gradient]=object_function_hyp(hyp)
% [~,gradient_differ]=differ(object_function_hyp,hyp)

% drawFunction(object_function_hyp,low_bou_hyp,up_bou_hyp);

hyp=fmincon...
    (object_function_hyp,hyp,[],[],[],[],low_bou_hyp,up_bou_hyp,[],fmincon_option);

% get parameter
[covariance,inv_covariance,beta,sigma_sq]=interpKriging...
    (X_dis_sq,Y_nomlz,x_number,variable_number,exp(hyp),fval_reg_nomlz);
gama=inv_covariance*(Y_nomlz-fval_reg_nomlz*beta);
FTRF=fval_reg_nomlz'*inv_covariance*fval_reg_nomlz;

% initialization predict function
predict_function=@(X_predict) interpKrigingPredictor...
    (X_predict,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_number,variable_number,exp(hyp),beta,gama,sigma_sq,...
    inv_covariance,fval_reg_nomlz,FTRF,reg_function);

kriging_model.X=X;
kriging_model.Y=Y;
kriging_model.fval_regression=fval_reg_nomlz;
kriging_model.covariance=covariance;
kriging_model.inv_covariance=inv_covariance;

kriging_model.hyp=hyp;
kriging_model.beta=beta;
kriging_model.gama=gama;
kriging_model.sigma_sq=sigma_sq;
kriging_model.aver_X=aver_X;
kriging_model.stdD_X=stdD_X;
kriging_model.aver_Y=aver_Y;
kriging_model.stdD_Y=stdD_Y;

kriging_model.predict_function=predict_function;

% abbreviation:
% num: number,pred: predict,vari: variable,hyp: hyper parameter
% NLL: negative log likelihood
    function [fval,gradient]=objectNLLKriging...
            (X_dis_sq,Y,x_num,vari_num,hyp,F_reg)
        % function to minimize sigma_sq
        %
        theta=exp(hyp);
        [cov,inv_cov,~,sigma2,inv_FTRF,Y_Fmiu]=interpKriging...
            (X_dis_sq,Y,x_num,vari_num,theta,F_reg);

        % calculation negative log likelihood
        L=chol(cov)';
        fval=x_num/2*log(sigma2)+sum(log(diag(L)));

        % calculate gradient
        if nargout > 1
            % gradient
            gradient=zeros(vari_num,1);
            for vari_index=1:vari_num
                dcov_dtheta=-(X_dis_sq(:,:,vari_index).*cov)*theta(vari_index)/vari_num;

                dinv_cov_dtheta=...
                    -inv_cov*dcov_dtheta*inv_cov;

                dinv_FTRF_dtheta=-inv_FTRF*...
                    (F_reg'*dinv_cov_dtheta*F_reg)*...
                    inv_FTRF;
                
                dmiu_dtheta=dinv_FTRF_dtheta*(F_reg'*inv_cov*Y)+...
                    inv_FTRF*(F_reg'*dinv_cov_dtheta*Y);
                
                dY_Fmiu_dtheta=-F_reg*dmiu_dtheta;

                dsigma2_dtheta=(dY_Fmiu_dtheta'*inv_cov*Y_Fmiu+...
                    Y_Fmiu'*dinv_cov_dtheta*Y_Fmiu+...
                    Y_Fmiu'*inv_cov*dY_Fmiu_dtheta)/x_num;
                
                dlnsigma2_dtheta=1/sigma2*dsigma2_dtheta;

                dlndetR=trace(inv_cov*dcov_dtheta);

                gradient(vari_index)=x_num/2*dlnsigma2_dtheta+0.5*dlndetR;
            end
        end
    end

    function [cov,inv_cov,beta,sigma_sq,inv_FTRF,Y_Fmiu]=interpKriging...
            (X_dis_sq,Y,x_num,vari_num,theta,F_reg)
        % kriging interpolation kernel function
        % Y(x)=beta+Z(x)
        %
        cov=zeros(x_num,x_num);
        for vari_index=1:vari_num
            cov=cov+X_dis_sq(:,:,vari_index)*theta(vari_index);
        end
        cov=exp(-cov/vari_num)+eye(x_num)*1e-3;

        % coefficient calculation
        inv_cov=cov\eye(x_num);
        inv_FTRF=(F_reg'*inv_cov*F_reg)\eye(size(F_reg,2));

        % basical bias
        beta=inv_FTRF*(F_reg'*inv_cov*Y);
        Y_Fmiu=Y-F_reg*beta;
        sigma_sq=(Y_Fmiu'*inv_cov*Y_Fmiu)/x_num;
        
    end

    function [Y_pred,Var_pred]=interpKrigingPredictor...
            (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,theta,beta,gama,sigma_sq,...
            inv_cov,fval_reg_nomlz,FTRF,reg_function)
        % kriging interpolation predict function
        % input predict_x and kriging model
        % predict_x is row vector
        % output the predict value
        %
        [x_pred_num,~]=size(X_pred);
        fval_reg_pred=reg_function(X_pred);

        % normalize data
        X_pred_nomlz=(X_pred-aver_X)./stdD_X;
        fval_reg_pred_nomlz=(fval_reg_pred-aver_Y)./stdD_Y;
        
        % predict covariance
        predict_cov=zeros(x_num,x_pred_num);
        for vari_index=1:vari_num
            predict_cov=predict_cov+...
                (X_nomlz(:,vari_index)-X_pred_nomlz(:,vari_index)').^2*theta(vari_index);
        end
        predict_cov=exp(-predict_cov/vari_num);

        % predict base fval
        
        Y_pred=fval_reg_pred_nomlz*beta+predict_cov'*gama;
        
        % predict variance
        u__=fval_reg_nomlz'*inv_cov*predict_cov-fval_reg_pred_nomlz';
        Var_pred=sigma_sq*...
            (1+u__'/FTRF*u__+...
            -predict_cov'*inv_cov*predict_cov);
        
        % normalize data
        Y_pred=Y_pred*stdD_Y+aver_Y;
        Var_pred=diag(Var_pred)*stdD_Y*stdD_Y;
    end

    function F_reg=regZero(X)
        % zero order base funcion
        %
        F_reg=ones(size(X,1),1); % zero
    end

    function F_reg=regLinear(X)
        % first order base funcion
        %
        F_reg=[ones(size(X,1),1),X]; % linear
    end
end
