clc;
clear;
close all hidden;

benchmark_type='single';

%% test case 
benchmark_name_list = {'G23'};
benchmark_error_list = {
    [1,5,100;
    2,0.5,1000;
    3,0.5,1000;
    4,0.5,1000;
    1,0.5,1000;
    2,0.5,1000;
    3,0.5,1000;
    4,0.5,1000;
    1,0.5,1000];
    [1,5,100;
    2,0.5,1000;
    3,0.5,1000;
    4,0.5,1000;
    1,0.5,1000;]
    [1,2,200;
    2,1,1000;
    3,1,1000;
    4,1,1000;
    1,1,1000;
    2,1,1000;];
    [1,-2,1000;
    2,0.05,100;
    3,0.05,1000;
    4,0.05,1000;
    1,0.05,1000;
    2,0.05,1000;
    3,0.05,1000;];};

cheapcon_function = [];

%% single run

% [x_best,obj_best,NFE,output] = optimalMFRBFGPCRS...
%     (Model_function,Cost,variable_number,low_bou,up_bou,...
%     cheapcon_function,300,500)
% result_x_best = output.result_x_best;
% result_obj_best = output.result_obj_best;
%
% figure(1);
% plot(result_obj_best);

%% repeat run
% mkdir('torl3')
% for benchmark_index=1:length(benchmark_name_list)
%     benchmark_name=benchmark_name_list{benchmark_index};
%     benchmark_error=benchmark_error_list{benchmark_index};
% 
%     [MF_model,variable_number,low_bou,up_bou,...
%         object_function,A,B,Aeq,Beq,nonlcon_function] = Benchmark().getBenchmarkMF(benchmark_type,benchmark_name,benchmark_error);
% 
%     Ratio = [1,4];
%     Cost = [1,0.25];
%     repeat_number = 25;
%     result_obj = zeros(repeat_number,1);
%     result_NFE = zeros(repeat_number,1);
%     max_NFE = 200;
%     for repeat_idx = 1:repeat_number
%         [x_best,obj_best,NFE,output] = optimalMFRBFGPCRS...
%             (MF_model,Cost,Ratio,variable_number,low_bou,up_bou,...
%             cheapcon_function,max_NFE,300,1e-6,1e-3);
% 
%         result_obj(repeat_idx) = obj_best;
%         result_NFE(repeat_idx) = NFE;
%         data_lib_HF=output.data_lib_HF;
% 
%         plot(data_lib_HF.Obj(data_lib_HF.result_best_idx),'o-')
%     end
% 
%     fprintf('Obj     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_obj),mean(result_obj),max(result_obj),std(result_obj));
%     fprintf('NFE     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_NFE),mean(result_NFE),max(result_NFE),std(result_NFE));
%     save(['torl3/',benchmark_name,'_MF_RBF_GPC_RS_',num2str(max_NFE),'.mat']);
% end

% mkdir('torl0')
for benchmark_index=1:length(benchmark_name_list)
    benchmark_name=benchmark_name_list{benchmark_index};
    benchmark_error=benchmark_error_list{benchmark_index};

    [MF_model,variable_number,low_bou,up_bou,...
        object_function,A,B,Aeq,Beq,nonlcon_function] = Benchmark().getBenchmarkMF(benchmark_type,benchmark_name,benchmark_error);

    Ratio = [1,4];
    Cost = [1,0.01];
    repeat_number = 25;
    result_obj = zeros(repeat_number,1);
    result_vio = zeros(repeat_number,1);
    result_NFE = zeros(repeat_number,1);
    max_NFE = 200;
    for repeat_idx = 1:repeat_number
        [x_best,obj_best,NFE,output,vio_best] = optimalMFCEGO...
            (MF_model,Cost,Ratio,variable_number,low_bou,up_bou,...
            cheapcon_function,max_NFE,300,1e-6,0);

        result_obj(repeat_idx) = obj_best;
        result_vio(repeat_idx) = vio_best;
        result_NFE(repeat_idx) = NFE;
        data_lib_HF=output.data_lib_HF;

%         plot(data_lib_HF.Obj(data_lib_HF.result_best_idx),'o-')
    end
    
    fprintf('feasible number: %d\n',sum(result_vio==0))
    result_obj_feas=result_obj(result_vio==0);
    fprintf('Obj     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_obj_feas),mean(result_obj_feas),max(result_obj_feas),std(result_obj_feas));
    fprintf('NFE     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_NFE),mean(result_NFE),max(result_NFE),std(result_NFE));
    save(['torl0/',benchmark_name,'_MFCEGO_',num2str(max_NFE),'.mat']);
end

%% main
function [x_best,obj_best,NFE,output,vio_best] = optimalMFCEGO...
    (MF_model,Cost,Ratio,vari_num,low_bou,up_bou,...
    cheapcon_fcn,....
    NFE_max,iter_max,torl,nonlcon_torl)
% MF-RBF-AL-KTR optimization algorithm
% Model_function is cell of multi fidelity function handle
% Cost is array of each fidelity model function
% from start to end is high to low fidelity
%
% abbreviation:
% num: number, vari: variable_number, ...
% MF: multi-fidelity, HF: high-fidelity, LF: low-fidelity,
%
% Copyright 2023 4 Adel
%
if nargin < 11 || isempty(nonlcon_torl)
    nonlcon_torl = 1e-3;
    if nargin < 10 || isempty(torl)
        torl = 1e-6;
        if nargin < 9
            iter_max = [];
            if nargin < 8
                NFE_max = [];
            end
        end
    end
end

if nargin < 7
    cheapcon_fcn = [];
end

DRAW_FIGURE_FLAG = 1; % whether draw data
INFORMATION_FLAG = 1; % whether print data
CONVERGENCE_JUDGMENT_FLAG = 0; % whether judgment convergence
WRIRE_FILE_FLAG = 0; % whether write to file

fidelity_number = length(MF_model);
HF_model = MF_model{1};
LF_model = MF_model{2};
cost_HF = Cost(1);
cost_LF = Cost(2);
ratio_HF = Ratio(1);
ratio_LF = Ratio(2);

% hyper parameter
sample_num_initial = ceil(vari_num*1.5);
multi_start=10;

nomlz_value = 10; % max obj when normalize obj,con,coneq
protect_range = 1e-5; % surrogate add point protect range

% NFE and iteration setting
if isempty(NFE_max)
    NFE_max = 10+10*vari_num;
end

if isempty(iter_max)
    iter_max = 20+20*vari_num;
end

done = 0;NFE = 0;iter = 0; NFE_list = zeros(fidelity_number,1);

% step 1
% generate initial data lib
sample_num_initial_HF=ceil(sample_num_initial*ratio_HF);
sample_num_initial_LF=ceil(sample_num_initial*ratio_LF);
X_updata_LF = lhsdesign(sample_num_initial_LF,vari_num,'iterations',50,'criterion','maximin').*(up_bou-low_bou)+low_bou;
X_updata_HF = getNestedHypercube(X_updata_LF,sample_num_initial_HF,vari_num,low_bou,up_bou);

data_lib_HF = struct('model_function',HF_model,'variable_number',vari_num,'low_bou',low_bou,'up_bou',up_bou,...
    'nonlcon_torlance',nonlcon_torl,'filename_data','result_total.txt','write_file_flag',WRIRE_FILE_FLAG,...
    'X',[],'Obj',[],'Con',[],'Coneq',[],'Vio',[],'Ks',[],'value_format','%.8e ','result_best_idx',[]);
data_lib_LF = struct('model_function',LF_model,'variable_number',vari_num,'low_bou',low_bou,'up_bou',up_bou,...
    'nonlcon_torlance',nonlcon_torl,'filename_data','result_total.txt','write_file_flag',WRIRE_FILE_FLAG,...
    'X',[],'Obj',[],'Con',[],'Coneq',[],'Vio',[],'Ks',[],'value_format','%.8e ','result_best_idx',[]);

% detech expensive constraints and initialize data lib
[data_lib_HF,~,~,con_new,coneq_new,~,~,~,NFE_updata]=dataUpdata(data_lib_HF,X_updata_HF(1,:),0);
NFE = NFE+NFE_updata*cost_HF;
NFE_list(1) = NFE_list(1)+NFE_updata;
if ~isempty(data_lib_HF.Vio)
    expensive_nonlcon_flag = 1;
else
    expensive_nonlcon_flag = 0;
end
con_num=length(con_new);
coneq_num=length(coneq_new);

% updata data lib by x_list
[data_lib_HF,~,~,~,~,~,~,~,NFE_updata]=dataUpdata(data_lib_HF,X_updata_HF(2:end,:),0);
NFE = NFE+NFE_updata*cost_HF;
NFE_list(1) = NFE_list(1)+NFE_updata;

% updata data lib by x_list
[data_lib_LF,~,~,~,~,~,~,~,NFE_updata]=dataUpdata(data_lib_LF,X_updata_LF,0);
NFE = NFE+NFE_updata*cost_LF;
NFE_list(2) = NFE_list(2)+NFE_updata;

% find fesiable data in current data lib
if expensive_nonlcon_flag
    Bool_feas_HF = data_lib_HF.Vio == 0;
    Bool_feas_LF = data_lib_LF.Vio == 0;
end

fmincon_options = optimoptions('fmincon','Display','none','Algorithm','sqp','ConstraintTolerance',0,'FiniteDifferenceStepSize',1e-5);

result_x_best = zeros(iter_max,vari_num);
result_obj_best = zeros(iter_max,1);

model_HK.LF_model.hyp=0;
model_HK.hyp=0;

model_obj=model_HK;
model_con_list=repmat({model_HK},[con_num,1]);
model_coneq_list=repmat({model_HK},[coneq_num,1]);

iter = iter+1;
while ~done
    % step 2
    % nomalization all data by max obj and to create surrogate model
    [X_HF,Obj_HF,Con_HF,Coneq_HF,Vio_HF,Ks_HF]=dataLoad(data_lib_HF);
    [X_LF,Obj_LF,Con_LF,Coneq_LF,Vio_LF,Ks_LF]=dataLoad(data_lib_LF);

    X_MF={X_HF,X_LF};Obj_MF={Obj_HF,Obj_LF};
    if ~isempty(Con_HF)
        Con_MF={Con_HF;Con_LF};
    else
        Con_MF=[];
    end
    if ~isempty(Coneq_HF)
        Coneq_MF={Coneq_HF;Coneq_LF};
    else
        Coneq_MF=[];
    end
    if ~isempty(Vio_HF)
        Vio_MF={Vio_HF;Vio_LF};
    else
        Vio_MF=[];
    end
    if ~isempty(Ks_HF)
        Ks_MF={Ks_HF;Ks_LF};
    else
        Ks_MF=[];
    end

    [X_model_MF,Obj_model_MF,Con_model_MF,Coneq_model_MF,Vio_model_MF,Ks_model_MF,...
        obj_max,con_max_list,coneq_max_list,vio_max,ks_max]=getModelData...
        (fidelity_number,X_MF,Obj_MF,Con_MF,Coneq_MF,Vio_MF,Ks_MF,nomlz_value);

    % get local infill point, construct surrogate model
    [obj_surr_fcn,con_surr_fcn,output_model] = getSurrogateFunction...
        (X_model_MF,Obj_model_MF,Con_model_MF,Coneq_model_MF,...
        model_obj,model_con_list,model_coneq_list);
    model_obj=output_model.model_obj;
    model_con_list=output_model.model_con_list;
    model_coneq_list=output_model.model_coneq_list;

    % step 3
    % find CEI or POF
    if any(Bool_feas_HF)
        infill_fcn=@(x) Infill_CEI(x,obj_surr_fcn,con_surr_fcn,min(Obj_model_MF{1}));
    else
        infill_fcn=@(x) Infill_PoF(x,con_surr_fcn);
    end
    X_potential=lhsdesign(multi_start,vari_num);
    Val_potential=zeros(multi_start,1);
    for x_idx=1:multi_start
        [X_potential(x_idx,:),Val_potential(x_idx),~,~]=fmincon...
            (infill_fcn,X_potential(x_idx,:),[],[],[],[],low_bou,up_bou,cheapcon_fcn,fmincon_options);
    end
    [val_min,idx]=min(Val_potential);
    x_infill=X_potential(idx,:);

    % step 4
    % updata infill point
    [data_lib_HF,x_infill,~,~,~,vio_infill_HF,~,repeat_idx,NFE_updata] = ...
        dataUpdata(data_lib_HF,x_infill,protect_range);
    NFE = NFE+NFE_updata*cost_HF;
    NFE_list(1) = NFE_list(1)+NFE_updata;

    if isempty(x_infill)
        % process error
    else
        if ~isempty(vio_infill_HF) && vio_infill_HF > 0
            Bool_feas_HF=[Bool_feas_HF;false(1)];
        else
            Bool_feas_HF=[Bool_feas_HF;true(1)];
        end
    end

    [data_lib_LF,x_infill,~,~,~,vio_infill_LF,~,repeat_idx,NFE_updata] = ...
        dataUpdata(data_lib_LF,x_infill,protect_range);
    NFE = NFE+NFE_updata*cost_LF;
    NFE_list(1) = NFE_list(1)+NFE_updata;

    if isempty(x_infill)
        % process error
    else
        if ~isempty(vio_infill_LF) && vio_infill_LF > 0
            Bool_feas_LF=[Bool_feas_LF;false(1)];
        else
            Bool_feas_LF=[Bool_feas_LF;true(1)];
        end
    end

    if DRAW_FIGURE_FLAG && vari_num < 3
        interpVisualize(model_obj,low_bou,up_bou);
        line(x_infill(1),x_infill(2),obj_infill./obj_max*nomlz_value,'Marker','o','color','r','LineStyle','none');
    end

    % find best result to record
    [X_HF,Obj_HF,~,~,Vio_HF,~]=dataLoad(data_lib_HF);
    idx = find(Vio_HF == 0);
    if isempty(idx)
        [vio_best,min_idx] = min(Vio_HF);
        obj_best = Obj_HF(min_idx);
        x_best = X_HF(min_idx,:);
    else
        [obj_best,min_idx] = min(Obj_HF(idx));
        vio_best = 0;
        x_best = X_HF(idx(min_idx),:);
    end

    if INFORMATION_FLAG
        fprintf('obj:    %f    violation:    %f    NFE:    %-3d\n',obj_best,vio_best,NFE);
        %         fprintf('iteration:          %-3d    NFE:    %-3d\n',iteration,NFE);
        %         fprintf('x:          %s\n',num2str(x_infill));
        %         fprintf('value:      %f\n',obj_infill);
        %         fprintf('violation:  %s  %s\n',num2str(con_infill),num2str(coneq_infill));
        %         fprintf('\n');
    end

    result_x_best(iter,:) = x_best;
    result_obj_best(iter,:) = obj_best;
    iter = iter+1;

    % forced interrupt
    if iter > iter_max || NFE >= NFE_max
        done = 1;
    end

    % convergence judgment
    if CONVERGENCE_JUDGMENT_FLAG
        if ( ((iter > 2) && (abs((obj_infill-obj_infill_old)/obj_infill_old) < torl)) && ...
                ((~isempty(vio_infill) && vio_infill == 0) || isempty(vio_infill)) )
            done = 1;
        end
    end

    obj_best_old = obj_best;
end

% find best result to record
x_best = data_lib_HF.X(data_lib_HF.result_best_idx(end),:);
obj_best = data_lib_HF.Obj(data_lib_HF.result_best_idx(end),:);
vio_best = data_lib_HF.Vio(data_lib_HF.result_best_idx(end),:);

result_x_best = result_x_best(1:iter-1,:);
result_obj_best = result_obj_best(1:iter-1);

output.result_x_best = result_x_best;
output.result_obj_best = result_obj_best;

output.NFE_list = NFE_list;

output.data_lib_HF = data_lib_HF;
output.data_lib_LF = data_lib_LF;

end

%% auxiliary function
function [X_model_MF,Obj_model_MF,Con_model_MF,Coneq_model_MF,Vio_model_MF,Ks_model_MF,...
    obj_max,con_max_list,coneq_max_list,vio_max,ks_max]=getModelData...
    (fidelity_number,X_MF,Obj_MF,Con_MF,Coneq_MF,Vio_MF,Ks_MF,nomlz_value)
% normalize data to construct surrogate model
%
X_model_MF=X_MF;

[Obj_model_MF,obj_max]=calNormalizeData...
    (fidelity_number,Obj_MF,nomlz_value);

if ~isempty(Con_MF)
    [Con_model_MF,con_max_list]=calNormalizeData...
        (fidelity_number,Con_MF,nomlz_value);
else
    Con_model_MF=[];
    con_max_list=[];
end

if ~isempty(Coneq_MF)
    [Coneq_model_MF,coneq_max_list]=calNormalizeData...
        (fidelity_number,Coneq_MF,nomlz_value);
else
    Coneq_model_MF=[];
    coneq_max_list=[];
end

if ~isempty(Vio_MF)
    [Vio_model_MF,vio_max]=calNormalizeData...
        (fidelity_number,Vio_MF,nomlz_value);
else
    Vio_model_MF=[];
    vio_max=[];
end

if ~isempty(Ks_MF)
    [Ks_model_MF,ks_max]=calNormalizeData...
        (fidelity_number,Ks_MF,nomlz_value);
else
    Ks_model_MF=[];
    ks_max=[];
end

    function [Data_model_MF,data_max]=calNormalizeData...
            (fidelity_number,Data_MF,nomlz_value)
        Data_model_MF=cell(1,fidelity_number);
        Data_total=[];
        for fidelity_idx=1:fidelity_number
            Data_total=[Data_total;Data_MF{fidelity_idx}];
        end
        data_max = max(abs(Data_total),[],1);
        for fidelity_idx=1:fidelity_number
            Data_model_MF{fidelity_idx}=Data_MF{fidelity_idx}./data_max*nomlz_value;
        end
    end
end

function vio_list = calVio(con_list,coneq_list,nonlcon_torlance)
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

%% infill function
function obj = Infill_CEI(x, obj_surr_fcn,con_surr_fcn,fmin)
% the kriging prediction and varince
[u,s] = obj_surr_fcn(x);
% the EI value
EI = (fmin-u).*normcdf((fmin-u)./s)+s.*normpdf((fmin-u)./s);

% the kriging prediction and varince
[u_g, s_g] = con_surr_fcn(x);
% the PoF value
PoF = prod(normpdf((0-u_g)./s_g),2);
CEI = EI.*PoF;
obj = CEI;
end

function obj = Infill_PoF(x, con_surr_fcn)
% the kriging prediction and varince
[u, s] = con_surr_fcn(x);
% the PoF value
PoF = prod(normcdf((0-u)./s),2);
obj = PoF;
end

%% surrogate model
function [obj_fcn_surr,nonlcon_fcn_surr,output] = getSurrogateFunction...
    (X_MF,Obj_MF,Con_MF,Coneq_MF,model_obj,model_con_list,model_coneq_list)
% base on input data to generate surrogate predict function
% nonlcon_function_surrogate if format of nonlcon function in fmincon
% judge MF and SF quality and select best one
%

[pred_func_obj,model_obj] = interpHieraKrigingPreModel(X_MF{1},Obj_MF{1},model_obj.hyp,X_MF{2},Obj_MF{2},model_obj.LF_model.hyp);

if ~isempty(Con_MF)
    con_number = size(Con_MF{1},2);
    pred_funct_con = cell(1,con_number);
    for con_idx = 1:con_number
        [pred_funct_con{con_idx},model_con_list{con_idx}] = interpHieraKrigingPreModel...
            (X_MF{1},Con_MF{1}(:,con_idx),model_con_list{con_idx}.hyp,X_MF{2},Con_MF{2}(:,con_idx),model_con_list{con_idx}.LF_model.hyp);
    end
else
    pred_funct_con = [];
end

if ~isempty(Coneq_MF)
    coneq_number = size(Coneq_MF{1},2);
    pred_funct_coneq = cell(1,coneq_number);
    for coneq_idx = 1:size(Coneq_MF,2)
        [pred_funct_coneq{coneq_idx},model_coneq_list{coneq_idx}] = interpHieraKrigingPreModel...
            (X_MF{1},Coneq_MF{1}(:,coneq_idx),model_coneq_list{coneq_idx}.hyp,X_MF{2},Coneq_MF{2}(:,coneq_idx),model_coneq_list{coneq_idx}.LF_model.hyp);
    end
else
    pred_funct_coneq = [];
end

obj_fcn_surr = @(X_predict) objectFunctionSurrogate(X_predict,pred_func_obj);
if isempty(pred_funct_con) && isempty(pred_funct_coneq)
    nonlcon_fcn_surr = [];
else
    nonlcon_fcn_surr = @(X_predict) nonlconFunctionSurrogate(X_predict,pred_funct_con,pred_funct_coneq);
end

output.model_obj=model_obj;
output.model_con_list=model_con_list;
output.model_coneq_list=model_coneq_list;

    function [obj, obj_var] = objectFunctionSurrogate...
            (X_predict,predict_function_obj)
        % connect all predict favl
        %
        [obj, obj_var] = predict_function_obj(X_predict);
    end

    function [con, con_var, coneq, coneq_var] = nonlconFunctionSurrogate...
            (X_predict,predict_function_con,predict_function_coneq)
        % connect all predict con and coneq
        %
        if isempty(predict_function_con)
            con = [];
            con_var = [];
        else
            con = zeros(size(X_predict,1),length(predict_function_con));
            con_var = zeros(size(X_predict,1),length(predict_function_con));
            for con_idx__ = 1:length(predict_function_con)
                [con(:,con_idx__),con_var(:,con_idx__)] = ....
                    predict_function_con{con_idx__}(X_predict);
            end
        end
        if isempty(predict_function_coneq)
            coneq = [];
            coneq_var = [];
        else
            coneq = zeros(size(X_predict,1),length(predict_function_coneq));
            coneq_var = zeros(size(X_predict,1),length(predict_function_con));
            for coneq_idx__ = 1:length(predict_function_coneq)
                [coneq(:, coneq_index__), coneq_var(:, coneq_index__)] = ...
                    predict_function_coneq{coneq_idx__}(X_predict);
            end
        end
    end

end

function [predict_function, HK_model] = interpHieraKrigingPreModel...
    (XHF, YHF, varargin)
% construct Hierarchical Kriging model
% XHF, YHF are x_HF_number x variable_number matrix
% XLF, YLF are x_LF_number x variable_number matrix
% aver_X, stdD_X is 1 x x_HF_number matrix
% theta beta gama sigma_sq is normalizede, so predict y is normalize
% hyp: hyp_HF, hyp_LF
% notice theta = exp(hyp)
%
% input:
% XHF, YHF, hyp_HF(can be []), XLF, YLF, hyp_LF(can be [])
% XHF, YHF, hyp_HF(can be []), LF_model
%
% output:
% predict_function, HK_model
%
% reference: [1] HAN Z-H, GöRTZ S. Hierarchical Kriging Model for
% Variable-Fidelity Surrogate Modeling [J]. AIAA Journal, 2012, 50(9):
% 1885-96.
%
% Copyright 2023.2 Adel
%
[x_HF_number, variable_number] = size(XHF);
switch nargin
    case 4
        hyp_HF = varargin{1};
        LF_model = varargin{2};

        % check whether LF model exist predict_function
        if ~isfield(LF_model, 'predict_function')
            error('interpHieraKrigingPreModel: low fidelity lack predict function');
        end
    case 6
        hyp_HF = varargin{1};
        XLF = varargin{2};
        YLF = varargin{3};
        hyp_LF = varargin{4};

        [x_LF_number, variable_number] = size(XLF);

        if isempty(hyp_LF)
            hyp_LF = 0;
        end

        % first step
        % construct low fidelity model

        % normalize data
        aver_X = mean(XLF);
        stdD_X = std(XLF);
        aver_Y = mean(YLF);
        stdD_Y = std(YLF);
        index__ = find(stdD_X == 0);
        if  ~isempty(index__), stdD_X(index__) = 1; end
        index__ = find(stdD_Y == 0);
        if  ~isempty(index__), stdD_Y(index__) = 1; end

        XLF_nomlz = (XLF-aver_X)./stdD_X;
        YLF_nomlz = (YLF-aver_Y)./stdD_Y;

        % initial X_dis_sq
        XLF_dis_sq = zeros(x_LF_number, x_LF_number, variable_number);
        for variable_index = 1:variable_number
            XLF_dis_sq(:, :, variable_index) = ...
                (XLF_nomlz(:, variable_index)-XLF_nomlz(:, variable_index)').^2;
        end

        % regression function define
        % notice reg_function process no normalization data
        % reg_function = @(X) regZero(X);
        reg_function = @(X) regLinear(X);

        % calculate reg
        obj_reg_LF = reg_function(XLF);
        obj_reg_nomlz_LF = (obj_reg_LF-aver_Y)./stdD_Y;

        % optimal to get hyperparameter
        low_bou_hyp = -3;
        up_bou_hyp = 3;
        fmincon_option = optimoptions('fmincon', 'Display', 'none', ...
            'OptimalityTolerance', 1e-2, ...
            'FiniteDifferenceStepSize', 1e-5, ..., 
            'MaxIterations', 10, 'SpecifyObjectiveGradient', true);
        prob_NLL_function = @(hyp) probNLLKriging...
            (XLF_dis_sq, YLF_nomlz, x_LF_number, variable_number, hyp, obj_reg_nomlz_LF);

        % [obj, gradient] = object_function_hyp(hyp_LF)
        % [~, gradient_differ] = differ(object_function_hyp, hyp_LF)

        hyp_LF = fmincon...
            (prob_NLL_function, hyp_LF, [], [], [], [], low_bou_hyp, up_bou_hyp, [], fmincon_option);

        % get parameter
        [cov_LF, inv_cov_LF, ~, beta_LF, sigma_sq_LF] = interpKriging...
            (XLF_dis_sq, YLF_nomlz, x_LF_number, variable_number, exp(hyp_LF), obj_reg_nomlz_LF);
        gama_LF = inv_cov_LF*(YLF_nomlz-obj_reg_nomlz_LF*beta_LF);
        FTRF_LF = obj_reg_nomlz_LF'*inv_cov_LF*obj_reg_nomlz_LF;

        % initialization predict function
        predict_function_LF = @(X_predict) interpKrigingPredictor...
            (X_predict, XLF_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
            x_LF_number, variable_number, exp(hyp_LF), beta_LF, gama_LF, sigma_sq_LF, ...
            inv_cov_LF, obj_reg_nomlz_LF, FTRF_LF, reg_function);

        LF_model.X = XLF;
        LF_model.Y = YLF;
        LF_model.obj_regression = obj_reg_LF;
        LF_model.covariance = cov_LF;
        LF_model.inv_covariance = inv_cov_LF;

        LF_model.hyp = hyp_LF;
        LF_model.beta = beta_LF;
        LF_model.gama = gama_LF;
        LF_model.sigma_sq = sigma_sq_LF;
        LF_model.aver_X = aver_X;
        LF_model.stdD_X = stdD_X;
        LF_model.aver_Y = aver_Y;
        LF_model.stdD_Y = stdD_Y;

        LF_model.predict_function = predict_function_LF;
    otherwise
        error('interpHieraKrigingPreModel: error input');
end
HK_model.LF_model = LF_model;

% second step
% construct hierarchical model
if isempty(hyp_HF)
    hyp_HF = 0;
end

predict_function_LF = LF_model.predict_function;

% normalize data
aver_X = mean(XHF);
stdD_X = std(XHF);
aver_Y = mean(YHF);
stdD_Y = std(YHF);
index__ = find(stdD_X == 0);
if  ~isempty(index__), stdD_X(index__) = 1; end
index__ = find(stdD_Y == 0);
if  ~isempty(index__), stdD_Y(index__) = 1; end
XHF_nomlz = (XHF-aver_X)./stdD_X;
YHF_nomlz = (YHF-aver_Y)./stdD_Y;

% initial X_dis_sq
XHF_dis_sq = zeros(x_HF_number, x_HF_number, variable_number);
for variable_index = 1:variable_number
    XHF_dis_sq(:, :, variable_index) = ...
        (XHF_nomlz(:, variable_index)-XHF_nomlz(:, variable_index)').^2;
end

% evaluate low fidelty predict value in high fidelity point as base obj
reg_function = @(X) predict_function_LF(X);

% calculate reg
obj_reg_HF = reg_function(XHF);
obj_reg_nomlz_HF = (obj_reg_HF-aver_Y)./stdD_Y;

% optimal to get hyperparameter
prob_NLL_function = @(hyp) probNLLKriging...
    (XHF_dis_sq, YHF_nomlz, x_HF_number, variable_number, hyp, obj_reg_nomlz_HF);

% [obj, gradient] = prob_NLL_function(hyp_HF)
% [~, gradient_differ] = differ(prob_NLL_function, hyp_HF)

% drawFunction(object_function_hyp, low_bou_hyp, up_bou_hyp);

hyp_HF = fmincon...
    (prob_NLL_function, hyp_HF, [], [], [], [], low_bou_hyp, up_bou_hyp, [], fmincon_option);

% calculate covariance and other parameter
[cov_HF, inv_cov_HF, ~, beta_HF, sigma_sq_HF] = interpKriging...
    (XHF_dis_sq, YHF_nomlz, x_HF_number, variable_number, exp(hyp_HF), obj_reg_nomlz_HF);
gama_HF = inv_cov_HF*(YHF_nomlz-obj_reg_nomlz_HF*beta_HF);
FTRF_HF = obj_reg_nomlz_HF'*inv_cov_HF*obj_reg_nomlz_HF;

% initialization predict function
predict_function = @(X_predict) interpKrigingPredictor...
    (X_predict, XHF_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
    x_HF_number, variable_number, exp(hyp_HF), beta_HF, gama_HF, sigma_sq_HF, ...
    inv_cov_HF, obj_reg_nomlz_HF, FTRF_HF, reg_function);

HK_model.X = {XHF, XLF};
HK_model.Y = {YHF, YLF};
HK_model.obj_regression = obj_reg_HF;
HK_model.covariance = cov_HF;
HK_model.inv_covariance = inv_cov_HF;

HK_model.hyp = hyp_HF;
HK_model.beta = beta_HF;
HK_model.gama = gama_HF;
HK_model.sigma_sq = sigma_sq_HF;

HK_model.aver_X = aver_X;
HK_model.stdD_X = stdD_X;
HK_model.aver_Y = aver_Y;
HK_model.stdD_Y = stdD_Y;

HK_model.predict_function = predict_function;

% abbreviation:
% num: number, pred: predict, vari: variable, hyp: hyper parameter
% NLL: negative log likelihood
    function [obj,gradient] = probNLLKriging...
            (X_dis_sq,Y,x_num,vari_num,hyp,F_reg)
        % function to minimize sigma_sq
        %
        theta = exp(hyp);
        [cov,inv_cov,L,~,sigma2,inv_FTRF,Y_Fmiu] = interpKriging...
            (X_dis_sq,Y,x_num,vari_num,theta,F_reg);

        % calculation negative log likelihood
        obj = x_num/2*log(sigma2)+sum(log(diag(L)));

        % calculate gradient
        if nargout > 1
            % gradient
            dcov_dtheta = zeros(x_num,x_num);
            for vari_index = 1:vari_num
                dcov_dtheta = dcov_dtheta + X_dis_sq(:,:,vari_index);
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
    
    function [cov,inv_cov,L,beta,sigma_sq,inv_FTRF,Y_Fmiu] = interpKriging...
            (X_dis_sq,Y,x_num,vari_num,theta,F_reg)
        % kriging interpolation kernel function
        % Y(x) = beta+Z(x)
        %
        cov = zeros(x_num,x_num);
        for vari_index = 1:vari_num
            cov = cov+X_dis_sq(:,:,vari_index)*theta;
        end
        cov = exp(-cov/vari_num)+eye(x_num)*1e-6;

        % coefficient calculation
        L = chol(cov)';
        inv_L = L\eye(x_num);
        inv_cov = inv_L'*inv_L;
        inv_FTRF = (F_reg'*inv_cov*F_reg)\eye(size(F_reg,2));

        % analytical solve sigma_sq
        beta = inv_FTRF*(F_reg'*inv_cov*Y);
        Y_Fmiu = Y-F_reg*beta;
        sigma_sq = (Y_Fmiu'*inv_cov*Y_Fmiu)/x_num;

    end

    function [Y_pred, Var_pred] = interpKrigingPredictor...
            (X_pred, X_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
            x_num, vari_num, theta, beta, gama, sigma_sq, ...
            inv_cov, obj_reg, FTRF, reg_function)
        % kriging interpolation predict function
        % input predict_x and kriging model
        % predict_x is row vector
        % output the predict value
        %
        [x_pred_num, ~] = size(X_pred);
        obj_reg_pred = reg_function(X_pred);

        % normalize data
        X_pred_nomlz = (X_pred-aver_X)./stdD_X;
        obj_reg_pred_nomlz = (obj_reg_pred-aver_Y)./stdD_Y;
        
        % predict covariance
        predict_cov = zeros(x_num, x_pred_num);
        for vari_index = 1:vari_num
            predict_cov = predict_cov+...
                (X_nomlz(:, vari_index)-X_pred_nomlz(:, vari_index)').^2*theta;
        end
        predict_cov = exp(-predict_cov/vari_num);

        % predict base obj
        
        Y_pred = obj_reg_pred_nomlz*beta+predict_cov'*gama;
        
        % predict variance
        u__ = obj_reg'*inv_cov*predict_cov-obj_reg_pred_nomlz';
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

%% LHD
function [X_sample,dist_min_nomlz,X_total] = getNestedHypercube...
    (X_base,sample_number,variable_number,...
    low_bou,up_bou,X_exist)
% generate nested latin hypercube design
% SLE method is used(sample and iteration, select max min distance group)
% election combination mode of point and find best combination
%
% input:
% X_base(which will be sample), sample number(new point to sample), ...
% variable_number, low_bou, up_bou, X_exist(exist point)
%
% output:
% X_sample, dist_min_nomlz(min distance of normalize data)
% X_total(include all data in area)
%
if nargin < 6
    X_exist = [];
    if nargin < 5
        up_bou = ones(1,variable_number);
        if nargin < 4
            low_bou = zeros(1,variable_number);
            if nargin < 3
                error('getLatinHypercube: lack input');
            end
        end
    end
end

iteration_max = 100*sample_number;

% check x_exist_list if meet boundary
if ~isempty(X_exist)
    if size(X_exist,2) ~= variable_number
        error('getLatinHypercube: x_exist_list variable_number error');
    end
    %     idx = find(X_exist < low_bou);
    %     idx = [idx,find(X_exist > up_bou)];
    %     if ~isempty(idx)
    %         error('getLatinHypercube: x_exist_list range error');
    %     end
    X_exist_nomlz = (X_exist-low_bou)./(up_bou-low_bou);
else
    X_exist_nomlz = [];
end

exist_number = size(X_exist,1);
total_number = sample_number+exist_number;
if sample_number <= 0
    X_total = X_exist;
    X_sample = [];
    dist_min_nomlz = calMinDistance(X_exist_nomlz);
    return;
end

% get quasi-feasible point
X_base_nomlz = (X_base-low_bou)./(up_bou-low_bou);

% iterate and get final x_supply_list
iteration = 0;
x_supply_quasi_number = size(X_base_nomlz,1);
dist_min_nomlz = 0;
X_sample_nomlz = [];

% dist_min_nomlz_result = zeros(1,iteration);
while iteration <= iteration_max
    % random select x_new_number X to X_trial_nomlz
    x_select_idx = randperm(x_supply_quasi_number,sample_number);

    % get distance min itertion X_
    distance_min_iteration = calMinDistanceIter...
        (X_base_nomlz(x_select_idx,:),X_exist_nomlz);

    % if distance_min_iteration is large than last time
    if distance_min_iteration > dist_min_nomlz
        dist_min_nomlz = distance_min_iteration;
        X_sample_nomlz = X_base_nomlz(x_select_idx,:);
    end

    iteration = iteration+1;
    %     dist_min_nomlz_result(iteration) = dist_min_nomlz;
end
dist_min_nomlz = calMinDistance([X_sample_nomlz;X_exist_nomlz]);
X_sample = X_sample_nomlz.*(up_bou-low_bou)+low_bou;
X_total = [X_exist;X_sample];

    function distance_min__ = calMinDistance(x_list__)
        % get distance min from x_list
        %
        if isempty(x_list__)
            distance_min__ = [];
            return;
        end

        % sort x_supply_list_initial to decrese distance calculate times
        x_list__ = sortrows(x_list__,1);
        [sample_number__,variable_number__] = size(x_list__);
        distance_min__ = variable_number__;
        for x_idx__ = 1:sample_number__
            x_curr__ = x_list__(x_idx__,:);
            x_next_idx__ = x_idx__ + 1;
            % only search in min_distance(x_list had been sort)
            search_range__ = variable_number__;
            while x_next_idx__ <= sample_number__ &&...
                    (x_list__(x_next_idx__,1)-x_list__(x_idx__,1))^2 ...
                    < search_range__
                x_next__ = x_list__(x_next_idx__,:);
                distance_temp__ = sum((x_next__-x_curr__).^2);
                if distance_temp__ < distance_min__
                    distance_min__ = distance_temp__;
                end
                if distance_temp__ < search_range__
                    search_range__ = distance_temp__;
                end
                x_next_idx__ = x_next_idx__+1;
            end
        end
        distance_min__ = sqrt(distance_min__);
    end
    function distance_min__ = calMinDistanceIter...
            (x_list__,x_exist_list__)
        % get distance min from x_list
        % this version do not consider distance between x exist
        %

        % sort x_supply_list_initial to decrese distance calculate times
        x_list__ = sortrows(x_list__,1);
        [sample_number__,variable_number__] = size(x_list__);
        distance_min__ = variable_number__;
        for x_idx__ = 1:sample_number__
            x_curr__ = x_list__(x_idx__,:);
            x_next_idx__ = x_idx__ + 1;
            % only search in min_distance(x_list had been sort)
            search_range__ = variable_number__;
            while x_next_idx__ <= sample_number__ &&...
                    (x_list__(x_next_idx__,1)-x_list__(x_idx__,1))^2 ...
                    < search_range__
                x_next__ = x_list__(x_next_idx__,:);
                distance_temp__ = sum((x_next__-x_curr__).^2);
                if distance_temp__ < distance_min__
                    distance_min__ = distance_temp__;
                end
                if distance_temp__ < search_range__
                    search_range__ = distance_temp__;
                end
                x_next_idx__ = x_next_idx__+1;
            end
            for x_exist_idx = 1:size(x_exist_list__,1)
                x_next__ = x_exist_list__(x_exist_idx,:);
                distance_temp__ = sum((x_next__-x_curr__).^2);
                if distance_temp__ < distance_min__
                    distance_min__ = distance_temp__;
                end
            end
        end
        distance_min__ = sqrt(distance_min__);
    end
end

%% data lib function
function [data_lib,x_new,obj_new,con_new,coneq_new,vio_new,ks_new,repeat_idx,NFE] = dataUpdata...
    (data_lib,x_origin_new,protect_range)
% updata data lib
% updata format:
% variable_number,obj_number,con_number,coneq_number
% x,obj,con,coneq
%
if nargin < 3
    protect_range = 0;
end

[x_new_num,~] = size(x_origin_new);
x_new = [];
obj_new = [];
con_new = [];
coneq_new = [];
vio_new = [];
ks_new = [];
repeat_idx = [];
NFE = 0;

if data_lib.write_file_flag
    file_data = fopen('result_total.txt','a');
end

% updata format:
% variable_number,obj_number,con_number,coneq_number
% x,obj,con,coneq
for x_idx = 1:x_new_num
    x = x_origin_new(x_idx,:);

    if protect_range ~= 0
        % updata data with same_point_avoid protect
        % check x_potential if exist in data lib
        % if exist, jump updata
        distance = sum((abs(x-data_lib.X)./(data_lib.up_bou-data_lib.low_bou)),2);
        [distance_min,min_idx] = min(distance);
        if distance_min < data_lib.variable_number*protect_range
            % distance to exist point of point to add is small than protect_range
            repeat_idx = [repeat_idx;min_idx];
            continue;
        end
    end

    [obj,con,coneq] = data_lib.model_function(x); % eval value
    NFE = NFE+1;

    obj = obj(:)';
    con = con(:)';
    coneq = coneq(:)';
    % calculate vio
    if isempty(con) && isempty(coneq)
        vio = [];
        ks = [];
    else
        vio = calVio(con,coneq,data_lib.nonlcon_torlance);
        ks = max([con,coneq]);
    end


    x_new = [x_new;x];
    obj_new = [obj_new;obj];
    if ~isempty(con)
        con_new = [con_new;con];
    end
    if ~isempty(coneq)
        coneq_new = [coneq_new;coneq];
    end
    if ~isempty(vio)
        vio_new = [vio_new;vio];
    end
    if ~isempty(ks)
        ks_new = [ks_new;ks];
    end

    if data_lib.write_file_flag
        % write data to txt_result
        fprintf(file_data,'%d ',repmat('%.8e ',1,data_lib.variable_number));
        fprintf(file_data,'%d ',length(obj));
        fprintf(file_data,'%d ',length(con));
        fprintf(file_data,'%d ',length(coneq));

        fprintf(file_data,data_lib.x_format,x);
        fprintf(file_data,repmat(data_lib.value_format,1,length(obj)),obj);
        fprintf(file_data,repmat(data_lib.value_format,1,length(con)),con);
        fprintf(file_data,repmat(data_lib.value_format,1,length(coneq)),coneq);
        fprintf(file_data,'\n');
    end

    data_lib=dataJoin(data_lib,x,obj,con,coneq,vio,ks);

    % record best
    if isempty(data_lib.result_best_idx)
        data_lib.result_best_idx = 1;
    else
        if isempty(vio) || vio == 0
            if obj <= data_lib.Obj(data_lib.result_best_idx)
                data_lib.result_best_idx = [data_lib.result_best_idx;size(data_lib.X,1)];
            else
                data_lib.result_best_idx = [data_lib.result_best_idx;data_lib.result_best_idx(end)];
            end
        else
            if vio <= data_lib.Vio(data_lib.result_best_idx)
                data_lib.result_best_idx = [data_lib.result_best_idx;size(data_lib.X,1)];
            else
                data_lib.result_best_idx = [data_lib.result_best_idx;data_lib.result_best_idx(end)];
            end
        end
    end
end

if data_lib.write_file_flag
    fclose(file_data);
    clear('file_data');
end
end

function [x_list,obj_list,con_list,coneq_list,vio_list,ks_list] = dataLoad...
    (data_lib,low_bou,up_bou)
% updata data to exist data lib
%
if nargin < 3
    up_bou = realmax;
    if nargin < 2
        low_bou = -realmax;
    end
end

idx=[];
for x_idx=1:size(data_lib.X,1)
    x=data_lib.X(x_idx,:);
    if all(x > low_bou) && all(x < up_bou)
        idx=[idx;x_idx];
    end
end

x_list = data_lib.X(idx,:);
obj_list = data_lib.Obj(idx,:);
if ~isempty(data_lib.Con)
    con_list = data_lib.Con(idx,:);
else
    con_list = [];
end
if ~isempty(data_lib.Coneq)
    coneq_list = data_lib.Coneq(idx,:);
else
    coneq_list = [];
end
if ~isempty(data_lib.Vio)
    vio_list = data_lib.Vio(idx,:);
else
    vio_list = [];
end
if ~isempty(data_lib.Ks)
    ks_list = data_lib.Ks(idx);
else
    ks_list = [];
end
end

function data_lib=dataJoin(data_lib,x,obj,con,coneq,vio,ks)
% updata data to exist data lib
%
data_lib.X = [data_lib.X;x];
data_lib.Obj = [data_lib.Obj;obj];
if ~isempty(data_lib.Con) || ~isempty(con)
    data_lib.Con = [data_lib.Con;con];
end
if ~isempty(data_lib.Coneq) || ~isempty(coneq)
    data_lib.Coneq = [data_lib.Coneq;coneq];
end
if ~isempty(data_lib.Vio) || ~isempty(vio)
    data_lib.Vio = [data_lib.Vio;vio];
end
if ~isempty(data_lib.Ks) || ~isempty(ks)
    data_lib.Ks = [data_lib.Ks;ks];
end
end
