clc;
clear;
close all hidden;

benchmark_type='single';

%% test case 
benchmark_name_list = {'G07','G09','G19','G23'};
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

mkdir('torl0')
for benchmark_index=4:length(benchmark_name_list)
    benchmark_name=benchmark_name_list{benchmark_index};
    benchmark_error=benchmark_error_list{benchmark_index};

    [MF_model,variable_number,low_bou,up_bou,...
        object_function,A,B,Aeq,Beq,nonlcon_function] = Benchmark().getBenchmarkMF(benchmark_type,benchmark_name,benchmark_error);

    Ratio = [1,4];
    Cost = [1,0.01];
    repeat_number = 25;
    result_obj = zeros(repeat_number,1);
    result_NFE = zeros(repeat_number,1);
    max_NFE = 200;
    for repeat_idx = 1:repeat_number
        [x_best,obj_best,NFE,output] = optimalMFRBFGPCRS...
            (MF_model,Cost,Ratio,variable_number,low_bou,up_bou,...
            cheapcon_function,max_NFE,300,1e-6,0);

        result_obj(repeat_idx) = obj_best;
        result_NFE(repeat_idx) = NFE;
        data_lib_HF=output.data_lib_HF;

%         plot(data_lib_HF.Obj(data_lib_HF.result_best_idx),'o-')
    end

    fprintf('Obj     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_obj),mean(result_obj),max(result_obj),std(result_obj));
    fprintf('NFE     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_NFE),mean(result_NFE),max(result_NFE),std(result_NFE));
    save(['torl0/',benchmark_name,'_MF_RBF_GPC_RS_',num2str(max_NFE),'.mat']);
end

%% main
function [x_best,obj_best,NFE,output] = optimalMFRBFGPCRS...
    (MF_model,Cost,Ratio,vari_num,low_bou,up_bou,...
    cheapcon_func,....
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
    cheapcon_func = [];
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
sample_num_initial = 4+2*vari_num;
sample_num_restart = sample_num_initial;
sample_num_add = ceil(log(sample_num_initial)/2);

% GPC sample parameter
min_bou_interest=1e-3;
max_bou_interest=1e-1;
trial_num = min(100*vari_num,100);

nomlz_value = 10; % max obj when normalize obj,con,coneq
protect_range = 1e-5; % surrogate add point protect range
identiy_torlance = 1e-3; % if inf norm of point less than identiy_torlance, point will be consider as same local best

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
Bool_conv_HF = false(size(data_lib_HF.X,1),1);
Bool_conv_LF = false(size(data_lib_LF.X,1),1);

hyp_MF.mean = 0;
hyp_MF.cov = [0,0,0,0,0];

hyp_SF.mean = 0;
hyp_SF.cov = [0,0];

X_local_best=[];
Obj_local_best=[];
X_potential=[];
Obj_potential=[];
Vio_potential=[];
detect_local_flag=true(1);

fmincon_options = optimoptions('fmincon','Display','none','Algorithm','sqp','ConstraintTolerance',0,'FiniteDifferenceStepSize',1e-5);

result_x_best = zeros(iter_max,vari_num);
result_obj_best = zeros(iter_max,1);

iter = iter+1;

add_LF_flag=true(1);
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
    [obj_surr_func,con_surr_func,output_model] = getSurrogateFunction...
        (X_model_MF,Obj_model_MF,Con_model_MF,Coneq_model_MF);
    model_obj=output_model.model_obj;
    model_con_list=output_model.model_con_list;
    model_coneq_list=output_model.model_coneq_list;

    % check if all model is SF
    all_SF_flag=true(1);
    if strcmp(model_obj.type,'MF')
        all_SF_flag=false(1);
    else
        if ~isempty(model_con_list)
            for con_idx =1:length(model_con_list)
                if strcmp(model_con_list{con_idx}.type,'MF')
                    all_SF_flag=false(1);
                    break;
                end
            end
        end

        if ~isempty(model_coneq_list)
            for coneq_idx =1:length(model_coneq_list)
                if strcmp(model_coneq_list{coneq_idx}.type,'MF')
                    all_SF_flag=false(1);
                    break;
                end
            end
        end
    end

    if all_SF_flag
        add_LF_flag=false(1);
    end

    if ~isempty(con_surr_func) || ~isempty(cheapcon_func)
        constraint_function = @(x) totalconFunction...
            (x,con_surr_func,cheapcon_func,[]);
    else
        constraint_function = [];
    end

    % step 3
    if detect_local_flag
        % detech potential local best point
        for x_idx=1:size(X_HF,1)
            x_initial=X_HF(x_idx,:);
            [x_potential,obj_potential_pred,exit_flag,output] = fmincon(obj_surr_func,x_initial,[],[],[],[],...
                low_bou,up_bou,constraint_function,fmincon_options);

            if exit_flag == 1 || exit_flag == 2
                % check if x_potential have existed
                add_flag=true(1);
                for x_check_idx=1:size(X_potential,1)
                    if sum(abs(X_potential(x_check_idx,:)-x_potential),2)/vari_num < identiy_torlance
                        add_flag=false(1);
                        break;
                    end
                end
                for x_check_idx=1:size(X_local_best,1)
                    if sum(abs(X_local_best(x_check_idx,:)-x_potential),2)/vari_num < identiy_torlance
                        add_flag=false(1);
                        break;
                    end
                end

                % updata into X_potential
                if add_flag
                    X_potential=[X_potential;x_potential];
                    Obj_potential=[Obj_potential;obj_potential_pred/nomlz_value.*obj_max];
                    [con_potential,coneq_potential]=con_surr_func(x_potential);
                    if ~isempty(con_potential)
                        con_potential=con_potential/nomlz_value.*con_max_list;
                    end
                    if ~isempty(coneq_potential)
                        coneq_potential=coneq_potential/nomlz_value.*coneq_max_list;
                    end
                    Vio_potential=[Vio_potential;calViolation(con_potential,coneq_potential,nonlcon_torl)];
                end
            end
        end

        % if X_potential is empty, try to use KS surrogate as x potential
        if isempty(X_potential)
            [ks_function_surrogate,model_ks] = getBestModel(X_model_MF,Ks_model_MF);

            [~,x_idx]=min(Vio_HF);
            x_initial=X_HF(x_idx,:);
            [x_potential,obj_potential_pred,exit_flag,output] = fmincon(ks_function_surrogate,x_initial,[],[],[],[],...
                low_bou,up_bou,cheapcon_func,fmincon_options);
            obj_potential_pred=obj_surr_func(x_potential);

            X_potential=[X_potential;x_potential];
            Obj_potential=[Obj_potential;obj_potential_pred/nomlz_value*obj_max];
            [con_potential,coneq_potential]=con_surr_func(x_potential);
            if ~isempty(con_potential)
                con_potential=con_potential/nomlz_value.*con_max_list;
            end
            if ~isempty(coneq_potential)
                coneq_potential=coneq_potential/nomlz_value.*coneq_max_list;
            end
            Vio_potential=[Vio_potential;calViolation(con_potential,coneq_potential,nonlcon_torl)];
        end

        detect_local_flag=false(1);
    else
        % updata X potential
        for x_idx=1:size(X_potential,1)
            x_potential=X_potential(x_idx,:);

            %             if Vio_potential(x_idx,:) == 0
            [x_potential,obj_potential_pred,exit_flag,output] = fmincon(obj_surr_func,x_potential,[],[],[],[],...
                low_bou,up_bou,constraint_function,fmincon_options);
            %             else
            %                 [x_potential,~,exit_flag,output] = fmincon(ks_function_surrogate,x_potential,[],[],[],[],...
            %                     low_bou,up_bou,cheapcon_function,fmincon_options);
            %                 obj_potential_pred=object_function_surrogate(x_potential);
            %             end

            X_potential(x_idx,:)=x_potential;
            Obj_potential(x_idx,:)=obj_potential_pred/nomlz_value*obj_max;
            [con_potential,coneq_potential]=con_surr_func(x_potential);
            if ~isempty(con_potential)
                con_potential=con_potential/nomlz_value.*con_max_list;
            end
            if ~isempty(coneq_potential)
                coneq_potential=coneq_potential/nomlz_value.*coneq_max_list;
            end
            Vio_potential(x_idx,:)=calViolation(con_potential,coneq_potential,nonlcon_torl);
        end

        % merge X potential
        % Upward merge
        for x_idx=size(X_potential,1):-1:1
            x_potential=X_potential(x_idx,:);

            % check if x_potential have existed
            merge_flag=false(1);
            for x_check_idx=1:x_idx-1
                if sum(abs(X_potential(x_check_idx,:)-x_potential),2)/vari_num < identiy_torlance
                    merge_flag=true(1);
                    break;
                end
            end

            % updata into X_potential
            if merge_flag
                X_potential(x_idx,:)=[];
                Obj_potential(x_idx,:)=[];
                Vio_potential(x_idx,:)=[];
            end
        end
    end

    % sort X potential by Vio
    [Vio_potential,idx]=sort(Vio_potential);
    Obj_potential=Obj_potential(idx,:);
    X_potential=X_potential(idx,:);

    % sort X potential by Obj
    flag=find(Vio_potential == 0, 1, 'last' );
    if isempty(flag) % mean do not have fesaible point
        [Obj_potential,idx]=sort(Obj_potential);
        Vio_potential=Vio_potential(idx,:);
        X_potential=X_potential(idx,:);
    else
        [Obj_potential(1:flag,:),idx_feas]=sort(Obj_potential(1:flag,:));
        [Obj_potential(flag+1:end,:),idx_infeas]=sort(Obj_potential(flag+1:end,:));
        idx=[idx_feas;idx_infeas+flag];

        Vio_potential=Vio_potential(idx,:);
        X_potential=X_potential(idx,:);
    end
    
    % step 4
    % select best potential point as x_infill
    x_infill=X_potential(1,:);
    obj_infill_pred=Obj_potential(1,:);

    % updata infill point
    [data_lib_HF,x_infill,obj_infill,con_infill,coneq_infill,vio_infill,ks_infill,repeat_idx,NFE_updata] = ...
        dataUpdata(data_lib_HF,x_infill,protect_range);
    NFE = NFE+NFE_updata*cost_HF;
    NFE_list(1) = NFE_list(1)+NFE_updata;

    if isempty(x_infill)
        % process error
        x_infill = data_lib_HF.X(repeat_idx,:);
        obj_infill = data_lib_HF.Obj(repeat_idx,:);
        if ~isempty(Con_HF)
            con_infill = data_lib_HF.Con(repeat_idx,:);
        end
        if ~isempty(Coneq_HF)
            coneq_infill = data_lib_HF.Coneq(repeat_idx,:);
        end
        if ~isempty(Vio_HF)
            vio_infill = data_lib_HF.Vio(repeat_idx,:);
        end
    else
        if ~isempty(vio_infill) && vio_infill > 0
            Bool_feas_HF=[Bool_feas_HF;false(1)];
        else
            Bool_feas_HF=[Bool_feas_HF;true(1)];
        end
        Bool_conv_HF=[Bool_conv_HF;false(1)];
    end
    Obj_potential(1,:)=obj_infill;
    Vio_potential(1,:)=vio_infill;

    if DRAW_FIGURE_FLAG && vari_num < 3
        interpVisualize(model_obj,low_bou,up_bou);
        line(x_infill(1),x_infill(2),obj_infill./obj_max*nomlz_value,'Marker','o','color','r','LineStyle','none');
    end

    % find best result to record
    [X_HF,Obj_HF,~,~,Vio_HF,~]=dataLoad(data_lib_HF);
    X_unconv_HF = X_HF(~Bool_conv_HF,:);
    Obj_unconv_HF=Obj_HF(~Bool_conv_HF,:);
    if ~isempty(Vio_HF)
        Vio_unconv_HF=Vio_HF(~Bool_conv_HF,:);
    else
        Vio_unconv_HF=[];
    end
    idx = find(Vio_unconv_HF == 0);
    if isempty(idx)
        [vio_best,min_idx] = min(Vio_unconv_HF);
        obj_best = Obj_unconv_HF(min_idx);
        x_best = X_unconv_HF(min_idx,:);
    else
        [obj_best,min_idx] = min(Obj_unconv_HF(idx));
        vio_best = 0;
        x_best = X_unconv_HF(idx(min_idx),:);
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

    if ~done
        [X_HF,Obj_HF,~,~,Vio_HF,~]=dataLoad(data_lib_HF);
        [X_LF,Obj_LF,~,~,Vio_LF,~]=dataLoad(data_lib_LF);
        % check if converage
        if ( ((iter > 2) && (abs((obj_infill-obj_infill_old)/obj_infill_old) < torl)) && ...
                ((~isempty(vio_infill) && vio_infill == 0) || isempty(vio_infill)) )
            % resample LHD
            % step 6.1
            X_local_best=[X_local_best;x_infill];
            Obj_local_best=[Obj_local_best;obj_infill];
            X_potential=X_potential(2:end,:);
            Obj_potential=Obj_potential(2:end,:);
            Vio_potential=Vio_potential(2:end,:);

            if isempty(X_potential)
                detect_local_flag=true(1);
                add_LF_flag=true(1);
            end

            % step 6.2
            % detech converage
            for x_idx = 1:size(X_HF,1)
                if ~Bool_conv_HF(x_idx)
                    x_single_pred = fmincon(obj_surr_func,X_HF(x_idx,:),[],[],[],[],low_bou,up_bou,con_surr_func,fmincon_options);

                    converage_flag=false(1);

                    for x_check_idx=1:size(X_local_best,1)
                        if sum(abs(X_local_best(x_check_idx,:)-x_single_pred),2)/vari_num < identiy_torlance
                            converage_flag=true(1);
                            break;
                        end
                    end

                    if converage_flag
                        % if converage to local minimum, set to infeasible
                        Bool_conv_HF(x_idx) = true(1);
                    end
                end
            end

            for x_idx = 1:size(X_LF,1)
                if ~Bool_conv_LF(x_idx)
                    x_single_pred = fmincon(obj_surr_func,X_LF(x_idx,:),[],[],[],[],low_bou,up_bou,con_surr_func,fmincon_options);

                    converage_flag=false(1);

                    for x_check_idx=1:size(X_local_best,1)
                        if sum(abs(X_local_best(x_check_idx,:)-x_single_pred),2)/vari_num < identiy_torlance
                            converage_flag=true(1);
                            break;
                        end
                    end

                    if converage_flag
                        % if converage to local minimum, set to infeasible
                        Bool_conv_LF(x_idx) = true(1);
                    end
                end
            end

            % step 6.3
            % use GPC to limit do not converage to exist local best
            if ~all(Bool_conv_HF)
                Class_HF = -1*ones(size(X_HF,1),1);
                Class_HF(Bool_conv_HF) = 1; % cannot go into converage area

                Class_LF = -1*ones(size(X_LF,1),1);
                Class_LF(Bool_conv_LF) = 1; % cannot go into converage area
                
                if add_LF_flag
                    [pred_func_GPCMF,model_GPCMF] = classifyGaussProcessMultiFidelity(X_HF,Class_HF,X_LF,Class_LF,hyp_MF);
                    con_GPC_fcn = @(x) conGPCFunction(x,pred_func_GPCMF);
                else
                    [pred_func_GPC,model_GPC] = classifyGaussProcess(X_HF,Class_HF,hyp_SF);
                    con_GPC_fcn = @(x) conGPCFunction(x,pred_func_GPC);
                end
            else
                con_GPC_fcn=[];
            end

            % step 6.4
            % resample latin hypercubic and updata into data lib
            usable_NFE=NFE_max-NFE;
            sample_num=min(sample_num_restart,ceil(usable_NFE/sum(Ratio)));
            sample_num_HF=ceil(sample_num*ratio_HF);
            sample_num_LF=ceil(sample_num*ratio_LF);

            % resample LF origin
            try
                X_add_LF = getLatinHypercube(sample_num_LF,vari_num,...
                low_bou,up_bou,X_LF,con_GPC_fcn);
            catch
                X_add_LF=lhsdesign(sample_num_LF,vari_num).*(up_bou-low_bou)+low_bou;
            end
            % resample HF from x_updata_LF
            X_add_HF = getNestedHypercube(X_add_LF,sample_num_HF,vari_num,...
                low_bou,up_bou,X_HF);

            if DRAW_FIGURE_FLAG && vari_num < 3
                classifyVisualization(model_GPCMF,low_bou,up_bou);
                line(X_HF(base_boolean,1),X_HF(base_boolean,2),'Marker','o','color','k','LineStyle','none');
            end
        else
            % step 5.1
            % check if improve
            improve = 0;
            if isempty(repeat_idx)
                Bool_comp=(~Bool_conv_HF)&Bool_feas_HF;
                Bool_comp(end)=false(1);
                if expensive_nonlcon_flag
                    min_vio = min(Vio_HF(~Bool_conv_HF(1:end-1)));
                    min_obj = min(Obj_HF(Bool_comp));

                    % if all point is infeasible,violation of point infilled is
                    % less than min violation of all point means improve.if
                    % feasible point exist,obj of point infilled is less than min
                    % obj means improve
                    if vio_infill == 0 || vio_infill < min_vio
                        if ~isempty(min_obj)
                            if obj_infill < min_obj
                                % improve, continue local search
                                improve = 1;
                            end
                        else
                            % improve, continue local search
                            improve = 1;
                        end
                    end
                else
                    min_obj = min(Obj_HF(Bool_comp));

                    % obj of point infilled is less than min obj means improve
                    if obj_infill < min_obj
                        % imporve, continue local search
                        improve = 1;
                    end
                end
            end

            % step 5.2
            % if obj no improve, use GPC to identify interest area
            % than, imporve interest area surrogate quality
            if ~improve
                if add_LF_flag
                    % construct GPCMF
                    train_num=min(size(data_lib_HF.X,1),11*vari_num-1+25);
                    [pred_func_GPCMF,model_GPCMF,x_pareto_center,hyp_MF]=trainFilterMF(data_lib_HF,data_lib_LF,Ratio,x_infill,hyp_MF,train_num,expensive_nonlcon_flag,Bool_conv_HF,Bool_conv_LF);
                    con_GPC_fcn=@(x) conGPCFunction(x,pred_func_GPCMF);
                else
                    % construct GPC
                    train_num=min(size(data_lib_HF.X,1),11*vari_num-1+25);
                    [pred_func_GPC,model_GPC,x_pareto_center,hyp_SF]=trainFilter(data_lib_HF,x_infill,hyp_SF,train_num,expensive_nonlcon_flag,Bool_conv_HF);
                    con_GPC_fcn=@(x) conGPCFunction(x,pred_func_GPC);
                end

                % step 5.3
                % identify interest area
                center_point=fmincon(con_GPC_fcn,x_pareto_center,[],[],[],[],low_bou,up_bou,cheapcon_func,fmincon_options);

                bou_interest=abs(center_point-x_infill);
                bou_interest=max(min_bou_interest.*(up_bou-low_bou),bou_interest);
                bou_interest=min(max_bou_interest.*(up_bou-low_bou),bou_interest);
%                 low_bou_interest=x_infill-bou_interest;
%                 up_bou_interest=x_infill+bou_interest;
%                 low_bou_interest=max(low_bou_interest,low_bou);
%                 up_bou_interest=min(up_bou_interest,up_bou);

                % generate trial point
                usable_NFE=NFE_max-NFE;
                sample_num=min(sample_num_add,ceil(usable_NFE/sum(Ratio)));
                sample_num_HF=ceil(sample_num*ratio_HF);
                sample_num_LF=ceil(sample_num*ratio_LF);

                trial_point=repmat(x_infill,trial_num,1);
                for variable_idx = 1:vari_num
                    trial_point(:,variable_idx)=trial_point(:,variable_idx)+...
                        normrnd(0,bou_interest(variable_idx),[trial_num,1]);
                end
                trial_point=max(trial_point,low_bou);
                trial_point=min(trial_point,up_bou);

                Bool_negetive=pred_func_GPCMF(trial_point) == -1;
                if sum(Bool_negetive) < sample_num_LF
                    value=con_GPC_fcn(trial_point);
                    thres=quantile(value,0.25);
                    Bool_negetive=value<thres;
                end
                trial_point=trial_point(Bool_negetive,:);

                % step 5.4
                if add_LF_flag
                    % select LF point from trial_point

                    max_dist=0;
                    iter_select=1;
                    while iter_select < 100
                        select_idx=randperm(size(trial_point,1),sample_num_LF);
                        dist=calMinDistanceIter(trial_point(select_idx,:),X_LF);
                        if max_dist < dist
                            X_add_LF=trial_point(select_idx,:);
                            max_dist=dist;
                        end
                        iter_select=iter_select+1;
                    end

                    % select HF point from LF
                    max_dist=0;
                    iter_select=1;
                    while iter_select < 100
                        select_idx=randperm(size(X_add_LF,1),sample_num_HF);
                        dist=calMinDistanceIter(X_add_LF(select_idx,:),X_HF);
                        if max_dist < dist
                            X_add_HF=X_add_LF(select_idx,:);
                            max_dist=dist;
                        end
                        iter_select=iter_select+1;
                    end
                else
                    X_add_LF=[];

                    % select HF point from trial_point
                    max_dist=0;
                    iter_select=1;
                    while iter_select < 100
                        select_idx=randperm(size(trial_point,1),sample_num_HF);
                        dist=calMinDistanceIter(trial_point(select_idx,:),X_HF);
                        if max_dist < dist
                            X_add_HF=trial_point(select_idx,:);
                            max_dist=dist;
                        end
                        iter_select=iter_select+1;
                    end
                end

                if DRAW_FIGURE_FLAG && vari_num < 3
                    classifyVisualization(model_GPCMF,low_bou,up_bou);
                    line(trial_point(:,1),trial_point(:,2),'Marker','o','color','k','LineStyle','none');
                    line(X_add_HF(:,1),X_add_HF(:,2),'Marker','o','color','g','LineStyle','none');
                end
            else
                X_add_HF=[];
                X_add_LF=[];
            end
        end

        % step 7
        % updata data lib
        [data_lib_HF,X_add_HF,~,~,~,Vio_updata_HF,~,~,NFE_updata] = ...
            dataUpdata(data_lib_HF,X_add_HF,protect_range);
        NFE = NFE+NFE_updata*cost_HF;
        NFE_list(1) = NFE_list(1)+NFE_updata;
        Bool_feas_HF = [Bool_feas_HF;Vio_updata_HF==0];
        Bool_conv_HF = [Bool_conv_HF;false(size(X_add_HF,1),1)];

        [data_lib_LF,X_add_LF,~,~,~,Vio_updata_LF,~,~,NFE_updata] = ...
            dataUpdata(data_lib_LF,X_add_LF,protect_range);
        NFE = NFE+NFE_updata*cost_LF;
        NFE_list(2) = NFE_list(2)+NFE_updata;
        Bool_feas_LF = [Bool_feas_LF;Vio_updata_LF==0];
        Bool_conv_LF = [Bool_conv_LF;false(size(X_add_LF,1),1)];

        % forced interrupt
        if iter > iter_max || NFE >= NFE_max
            done = 1;
        end
    end

    obj_best_old = obj_best;

    obj_infill_old = obj_infill;
    con_infill_old = con_infill;
    coneq_infill_old = coneq_infill;
    vio_infill_old = vio_infill;
end

% find best result to record
x_best = data_lib_HF.X(data_lib_HF.result_best_idx(end),:);
obj_best = data_lib_HF.Obj(data_lib_HF.result_best_idx(end),:);

result_x_best = result_x_best(1:iter-1,:);
result_obj_best = result_obj_best(1:iter-1);

output.result_x_best = result_x_best;
output.result_obj_best = result_obj_best;

output.x_local_best = X_local_best;
output.obj_local_best = Obj_local_best;
output.NFE_list = NFE_list;

output.data_lib_HF = data_lib_HF;
output.data_lib_LF = data_lib_LF;

%     function obj = meritFunction(x,object_function_surrogate,x_list,variable_number,up_bou,low_bou,wF,wD)
%         % function to consider surrogate obj and variance
%         %
%         obj_pred = object_function_surrogate(x);
%
%         obj_dist = -atan(min(sum(((x-x_list)./(up_bou-low_bou)/variable_number).^2,2))*200);
%
%         obj = obj_pred*wF+obj_dist*wD;
%     end

    function obj = meritFunction...
            (x_trial_list,object_function_surrogate,x_list,variable_number,...
            w_F,w_D)
        % function to evaluate sample point
        %

        % value scale
        F = object_function_surrogate(x_trial_list);
        F_min = min(F); F_max = max(F);
        F = (F-F_min)./(F_max-F_min);

        % distance scale
        dis = zeros(size(x_trial_list,1),size(x_list,1));
        for vari_idx = 1:variable_number
            dis = dis+(x_trial_list(:,vari_idx)-x_list(:,vari_idx)').^2;
        end
        D = min(sqrt(dis),[],2);
        D_min = min(D); D_max = max(D);
        D = (D_max-D)./(D_max-D_min);

        obj = w_F*F+w_D*D;
    end

    function [con,coneq] = conGPCFunction(x,pred_func_GPC)
        % function to obtain probability predict function
        %
        [~,~,con] = pred_func_GPC(x);
        coneq = [];
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

function pareto_idx_list = getParetoFront(data_list)
% distinguish pareto front of data list
% data_list is x_number x data_number matrix
% notice if all data of x1 is less than x2,x1 domain x2
%
x_number = size(data_list,1);
pareto_idx_list = []; % sort all idx of filter point list

% select no domain filter
for x_idx = 1:x_number
    data = data_list(x_idx,:);
    pareto_idx = 1;
    add_filter_flag = 1;
    while pareto_idx <= length(pareto_idx_list)
        % compare x with exit pareto front point
        x_pareto_idx = pareto_idx_list(pareto_idx,:);

        % contain constraint of x_filter
        data_pareto = data_list(x_pareto_idx,:);

        % compare x with x_pareto
        judge = data >= data_pareto;
        if ~sum(~judge)
            add_filter_flag = 0;
            break;
        end

        % if better or equal than exit pareto point,reject pareto point
        judge = data <= data_pareto;
        if ~sum(~judge)
            pareto_idx_list(pareto_idx) = [];
            pareto_idx = pareto_idx-1;
        end

        pareto_idx = pareto_idx+1;
    end

    % add into pareto list if possible
    if add_filter_flag
        pareto_idx_list = [pareto_idx_list;x_idx];
    end
end
end

function [con,coneq] = totalconFunction...
    (x,nonlcon_function,cheapcon_function,con_GPC_function)
con = [];
coneq = [];
if ~isempty(nonlcon_function)
    [expencon,expenconeq] = nonlcon_function(x);
    con = [con,expencon];
    coneq = [coneq,expenconeq];
end
if ~isempty(cheapcon_function)
    [expencon,expenconeq] = cheapcon_function(x);
    con = [con,expencon];
    coneq = [coneq,expenconeq];
end
if ~isempty(con_GPC_function)
    [expencon,expenconeq] = con_GPC_function(x);
    con = [con,expencon];
    coneq = [coneq,expenconeq];
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

function [predict_function_GPCMF,model_GPCMF,x_pareto_center,hyp]=trainFilterMF(data_lib_HF,data_lib_LF,Ratio,x_infill,hyp,train_num,expensive_nonlcon_flag,Bool_conv_HF,Bool_conv_LF)
% train filter of gaussian process classifer
%
ratio_HF=Ratio(1);
ratio_LF=Ratio(2);

[X_HF,Obj_HF,~,~,Vio_HF,Ks_HF]=dataLoad(data_lib_HF);
train_num_HF=min(train_num*ratio_HF,size(X_HF,1));
% base on distance select usable point
x_distance=sum(abs(X_HF-x_infill),2);
[~,idx]=sort(x_distance);
Obj_HF=Obj_HF(idx(1:train_num_HF),:);
Ks_HF=Ks_HF(idx(1:train_num_HF),:);
X_HF=X_HF(idx(1:train_num_HF),:);
% Vio_HF=Vio_HF(idx(1:train_num_HF),:);
Bool_conv_HF=Bool_conv_HF(idx(1:train_num_HF),:);
% Bool_feas_HF=Vio_HF==0;

[X_LF,Obj_LF,~,~,Vio_LF,Ks_LF]=dataLoad(data_lib_LF);
train_num_LF=min(train_num*ratio_LF,size(X_LF,1));
% base on distance select usable point
x_distance=sum(abs(X_LF-x_infill),2);
[~,idx]=sort(x_distance);
Obj_LF=Obj_LF(idx(1:train_num_LF),:);
Ks_LF=Ks_LF(idx(1:train_num_LF),:);
X_LF=X_LF(idx(1:train_num_LF),:);
% Vio_LF=Vio_LF(idx(1:train_num_LF),:);
Bool_conv_LF=Bool_conv_LF(idx(1:train_num_LF),:);
% Bool_feas_LF=Vio_LF==0;

if expensive_nonlcon_flag
    % base on filter to decide which x should be choose
    pareto_idx_list = getParetoFront([Obj_HF,Ks_HF]);

    Class_HF = ones(size(X_HF,1),1);
    Class_HF(pareto_idx_list) = -1;
    Class_HF(Bool_conv_HF) = 1; % cannot go into convarage area

    x_pareto_center=sum(X_HF(pareto_idx_list,:),1)/length(pareto_idx_list);

    pareto_idx_list = getParetoFront([Obj_LF,Ks_LF]);

    Class_LF = ones(size(X_LF,1),1);
    Class_LF(pareto_idx_list) = -1;
    Class_LF(Bool_conv_LF) = 1; % cannot go into convarage area

    [predict_function_GPCMF,model_GPCMF] = classifyGaussProcessMultiFidelity...
        (X_HF,Class_HF,X_LF,Class_LF,hyp);

else
    obj_threshold = prctile(Obj_HF,50-40*sqrt(NFE/NFE_max));
    Class_HF = ones(size(X_HF,1),1);
    Class_HF(Obj_HF < obj_threshold) = -1;
    Class_HF(Bool_conv_HF) = 1; % cannot go into convarage area

    Class_LF = ones(size(X_LF,1),1);
    Class_LF(Obj_LF < obj_threshold) = -1;
    Class_LF(Bool_conv_LF) = 1; % cannot go into convarage area

    x_pareto_center=sum(X_HF(Obj < obj_threshold,:),1)/sum(Obj_HF < obj_threshold);

    [predict_function_GPCMF,model_GPCMF] = classifyGaussProcessMultiFidelity...
        (X_HF,Class_HF,X_LF,Class_LF,hyp);
end
hyp=model_GPCMF.hyp;
end

function [pred_func_GPC,model_GPC,x_pareto_center,hyp]=trainFilter(data_lib,x_infill,hyp,train_num,expensive_nonlcon_flag,Bool_conv)
% train filter of gaussian process classifer
%
[X,Obj,~,~,Vio,Ks]=dataLoad(data_lib);
% base on distance select usable point
x_dist=sum(abs(X-x_infill),2);
[~,idx]=sort(x_dist);
Obj=Obj(idx(1:train_num),:);
Ks=Ks(idx(1:train_num),:);
X=X(idx(1:train_num),:);
Vio=Vio(idx(1:train_num),:);
Bool_conv=Bool_conv(idx(1:train_num),:);
Bool_feas=Vio==0;

if expensive_nonlcon_flag
    % base on filter to decide which x should be choose
    %     pareto_idx_list = getParetoFront([Obj(~Bool_feas),Ks(~Bool_feas)]);
    pareto_idx_list = getParetoFront([Obj,Ks]);

    Class = ones(size(X,1),1);
    Class(pareto_idx_list) = -1;
    %     Class(Bool_feas) = -1; % can go into feasiable area
    Class(Bool_conv) = 1; % cannot go into convarage area

    x_pareto_center=sum(X(pareto_idx_list,:),1)/length(pareto_idx_list);

    [pred_func_GPC,model_GPC] = classifyGaussProcess(X,Class,hyp);
else
    obj_threshold = prctile(Obj(~Bool_feas),50-40*sqrt(NFE/NFE_max));

    Class = ones(size(X,1),1);
    Class(Obj < obj_threshold) = -1;
    Class(Bool_feas) = -1; % can go into feasiable area
    Class(Bool_conv) = 1; % cannot go into convarage area

    x_pareto_center=sum(X(Obj < obj_threshold,:),1)/sum(Obj < obj_threshold);

    [pred_func_GPC,model_GPC] = classifyGaussProcess(X,Class,hyp);
end
hyp=model_GPC.hyp;
end

%% machine learning
function [predict_function,CGPMF_model] = classifyGaussProcessMultiFidelity...
    (X_HF,Class_HF,X_LF,Class_LF,hyp)
% generate multi fidelity gaussian process classifier model
% version 6,this version is assembly of gpml-3.6 EP method
% X,X_HF,X_LF is x_number x variable_number matirx
% Class,ClassH,ClassLF is x_number x 1 matrix
% low_bou,up_bou is 1 x variable_number matrix
%
% input:
% X_HF,ClassHF,X_LF,ClassLF,hyp(mean,cov(lenD,etaD,lenL,etaL,rho))
%
% abbreviation:
% pred: predicted,nomlz: normalization,num: number
% var: variance
%
% reference: [1]FSCAB C,PP D,EK E,et al. Multi-fidelity classification
% using Gaussian processes: Accelerating the prediction of large-scale
% computational models [J]. Computer Methods in Applied Mechanics and
% Engineering,357(C): 112602-.
%
X = [X_HF;X_LF];
Class = [Class_HF;Class_LF];
[x_number,variable_number] = size(X);
x_HF_number = size(X_HF,1);
x_LF_number = size(X_LF,1);
if nargin < 5
    hyp.mean = 0;
    hyp.cov = zeros(1,5);
end

% normalization data
aver_X = mean(X);
stdD_X = std(X);
idx__ = find(stdD_X  ==  0);
if  ~isempty(idx__),stdD_X(idx__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;

object_function = @(x) objectNLLMFGPC(x,{@infEP},{@meanConst},{@calCovMF},{@likErf},X_nomlz,Class);
hyp_x = [hyp.mean,hyp.cov];

% [obj,gradient] = object_function(hyp_x)
% [obj_differ,gradient_differ] = differ(object_function,hyp_x)

low_bou_hyp = -4*ones(1,6);
up_bou_hyp = 4*ones(1,6);
hyp_x = fmincon(object_function,hyp_x,[],[],[],[],low_bou_hyp,up_bou_hyp,[],...
    optimoptions('fmincon','Display','none','SpecifyObjectiveGradient',true,...
    'MaxFunctionEvaluations',20,'OptimalityTolerance',1e-6));

% hyp.mean = hyp_x(1);
hyp.mean = 0;
hyp.cov = hyp_x(2:end);
hyp.lik = [];
post = infEP(hyp,{@meanConst},{@calCovMF},{@likErf},X_nomlz,Class);
predict_function = @(x_pred) classifyGaussPredictor...
    (x_pred,hyp,{@meanConst},{@calCovMF},{@likErf},post,X_nomlz,aver_X,stdD_X);

% output model
X = {X_HF,X_LF};
Class = {Class_HF,Class_LF};
CGPMF_model.X = X;
CGPMF_model.Class = Class;
CGPMF_model.aver_X = aver_X;
CGPMF_model.stdD_X = stdD_X;
CGPMF_model.predict_function = predict_function;
CGPMF_model.hyp = hyp;
CGPMF_model.post = post;

    function [obj,gradient] = objectNLLMFGPC(x,inf,mean,cov,lik,X,Y)
        hyp_iter.mean = x(1);
        hyp_iter.cov = x(2:end);
        hyp_iter.lik = [];

        if nargout < 2
            [~,nlZ] = feval(inf{:},hyp_iter,mean,cov,lik,X,Y);
            obj = nlZ;
        elseif nargout < 3
            [~,nlZ,dnlZ] = feval(inf{:},hyp_iter,mean,cov,lik,X,Y);
            obj = nlZ;
            gradient = [dnlZ.mean,dnlZ.cov];
        end
    end

    function [class,possibility,miu_pre,var_pre] = classifyGaussPredictor...
            (X_pred,hyp,mean,cov,lik,post,X,aver_X,stdD_X)
        % predict function
        %
        X_pred_nomlz = (X_pred-aver_X)./stdD_X;
        pred_num = size(X_pred_nomlz,1);
        ys = ones(pred_num,1);

        alpha = post.alpha; L = post.L; sW = post.sW;
        %verify whether L contains valid Cholesky decomposition or something different
        Lchol = isnumeric(L) && all(all(tril(L,-1) == 0)&diag(L)'>0&isreal(diag(L))');
        ns = size(X_pred_nomlz,1);                                       % number of data points
        nperbatch = 1000;                       % number of data points per mini batch
        nact = 0;                       % number of already processed test data points
        ymu = zeros(ns,1); ys2 = ymu; miu_pre = ymu; var_pre = ymu; possibility = ymu;   % allocate mem
        while nact<ns               % process minibatches of test cases to save memory
            id = (nact+1):min(nact+nperbatch,ns);               % data points to process
            kss = feval(cov{:},hyp.cov,X_pred_nomlz(id,:),'diag');              % self-variance
            Ks = feval(cov{:},hyp.cov,X,X_pred_nomlz(id,:));        % avoid computation
            ms = feval(mean{:},hyp.mean,X_pred_nomlz(id,:));
            N = size(alpha,2);  % number of alphas (usually 1; more in case of sampling)
            Fmu = repmat(ms,1,N) + Ks'*full(alpha);        % conditional mean fs|f
            miu_pre(id) = sum(Fmu,2)/N;                                   % predictive means
            if Lchol    % L contains chol decomp  = > use Cholesky parameters (alpha,sW,L)
                V  = L'\(repmat(sW,1,length(id)).*Ks);
                var_pre(id) = kss - sum(V.*V,1)';                       % predictive variances
            else                % L is not triangular  = > use alternative parametrisation
                if isnumeric(L),LKs = L*Ks; else LKs = L(Ks); end    % matrix or callback
                var_pre(id) = kss + sum(Ks.*LKs,1)';                    % predictive variances
            end
            var_pre(id) = max(var_pre(id),0);   % remove numerical noise i.e. negative variances
            Fs2 = repmat(var_pre(id),1,N);     % we have multiple values in case of sampling
            if nargin<9
                [Lp,Ymu,Ys2] = feval(lik{:},hyp.lik,[],Fmu(:),Fs2(:));
            else
                Ys = repmat(ys(id),1,N);
                [Lp,Ymu,Ys2] = feval(lik{:},hyp.lik,Ys(:),Fmu(:),Fs2(:));
            end
            possibility(id)  = sum(reshape(Lp,[],N),2)/N;    % log probability; sample averaging
            ymu(id) = sum(reshape(Ymu,[],N),2)/N;          % predictive mean ys|y and ..
            ys2(id) = sum(reshape(Ys2,[],N),2)/N;                          % .. variance
            nact = id(end);          % set counter to idx of last processed data point
        end

        possibility = exp(possibility);
        class = ones(pred_num,1);
        idx_list = possibility < 0.5;
        class(idx_list) = -1;
    end

    function [K,dK_dcov] = calCovMF(cov,X,Z)
        % obtain covariance of x
        % cov: lenD,etaD,lenL,etaL,rho
        % len equal to 1/len_origin.^2
        %
        % % k = eta*exp(-sum(x_dis*theta)/vari_num);
        %
        [x_num,vari_num] = size(X);

        lenD = exp(cov(1));
        etaD = exp(cov(2));
        lenL = exp(cov(3));
        etaL = exp(cov(4));
        rho = exp(cov(5));

        if nargin > 2 && nargout < 2 && ~isempty(Z)
            % predict
            if strcmp(Z,'diag')
                K = rho*rho*etaL+etaD;
            else
                [z_num,vari_num] = size(Z);
                % initializate square of X inner distance sq/ vari_num
                sq_dis_v = zeros(x_num,z_num,vari_num);
                for len_idx = 1:vari_num
                    sq_dis_v(:,:,len_idx) = (X(:,len_idx)-Z(:,len_idx)').^2/vari_num;
                end

                % exp of x__x with D
                exp_disD = zeros(x_HF_number,z_num);
                for len_idx = 1:vari_num
                    exp_disD = exp_disD+...
                        sq_dis_v(1:x_HF_number,:,len_idx)*lenD;
                end
                exp_disD = exp(-exp_disD);

                % exp of x__x with L
                exp_disL = zeros(x_num,z_num);
                for len_idx = 1:vari_num
                    exp_disL = exp_disL+...
                        sq_dis_v(1:x_num,:,len_idx)*lenL;
                end
                exp_disL = exp(-exp_disL);

                % covariance
                K = exp_disL;
                K(1:x_HF_number,:) = rho*rho*etaL*K(1:x_HF_number,:)+etaD*exp_disD;
                K(x_HF_number+1:end,:) = rho*etaL*K(x_HF_number+1:end,:);
            end
        else
            % initializate square of X inner distance sq/ vari_num
            sq_dis_v = zeros(x_num,x_num,vari_num);
            for len_idx = 1:vari_num
                sq_dis_v(:,:,len_idx) = (X(:,len_idx)-X(:,len_idx)').^2/vari_num;
            end

            % exp of x__x with H
            exp_disD = zeros(x_num);
            for len_idx = 1:vari_num
                exp_disD(1:x_HF_number,1:x_HF_number) = exp_disD(1:x_HF_number,1:x_HF_number)+...
                    sq_dis_v(1:x_HF_number,1:x_HF_number,len_idx)*lenD;
            end
            exp_disD(1:x_HF_number,1:x_HF_number) = exp(-exp_disD(1:x_HF_number,1:x_HF_number));
            KD = etaD*exp_disD;

            % exp of x__x with L
            exp_disL = zeros(x_num);
            for len_idx = 1:vari_num
                exp_disL = exp_disL+...
                    sq_dis_v(1:end,1:end,len_idx)*lenL;
            end
            exp_disL = exp(-exp_disL);
            eta_exp_disL = etaL*exp_disL;

            % times rho: HH to rho2,HL to rho,LL to 1
            KL = eta_exp_disL;
            KL(1:x_HF_number,1:x_HF_number) = ...
                (rho*rho)*eta_exp_disL(1:x_HF_number,1:x_HF_number);
            KL(1:x_HF_number,(x_HF_number+1):end) = ...
                rho*eta_exp_disL(1:x_HF_number,(x_HF_number+1):end);
            KL((x_HF_number+1):end,1:x_HF_number) = ...
                KL(1:x_HF_number,(x_HF_number+1):end)';

            K = KL+KD;

            if nargout >= 2
                dK_dcov = cell(1,5);

                % len D
                dK_dlenD = zeros(x_num);
                for len_idx = 1:vari_num
                    dK_dlenD(1:x_HF_number,1:x_HF_number) = dK_dlenD(1:x_HF_number,1:x_HF_number) + ...
                        sq_dis_v(1:x_HF_number,1:x_HF_number,len_idx);
                end
                dK_dlenD(1:x_HF_number,1:x_HF_number) = -dK_dlenD(1:x_HF_number,1:x_HF_number).*...
                    KD(1:x_HF_number,1:x_HF_number)*lenD;
                dK_dcov{1} = dK_dlenD;

                % eta D
                dK_dcov{2} = KD;

                % len L
                dK_dlenL = zeros(x_num,x_num);
                for len_idx = 1:vari_num
                    dK_dlenL = dK_dlenL + sq_dis_v(:,:,len_idx);
                end
                dK_dlenL = -dK_dlenL.*KL.*lenL;
                dK_dcov{3} = dK_dlenL;

                % eta L
                dK_dcov{4} = KL;

                % rho
                dK_drho = zeros(x_num);
                dK_drho(1:x_HF_number,1:x_HF_number) = ...
                    2*rho*rho*eta_exp_disL(1:x_HF_number,1:x_HF_number);
                dK_drho(1:x_HF_number,(x_HF_number+1):end) = ...
                    rho*eta_exp_disL(1:x_HF_number,(x_HF_number+1):end);
                dK_drho((x_HF_number+1):end,1:x_HF_number) = ...
                    dK_drho(1:x_HF_number,(x_HF_number+1):end)';
                dK_dcov{5} = dK_drho;
            end
        end

    end

    function [post,nlZ,dnlZ] = infEP(hyp,mean,cov,lik,x,y)
        % Expectation Propagation approximation to the posterior Gaussian Process.
        % The function takes a specified covariance function (see covFunctions.m) and
        % likelihood function (see likFunctions.m),and is designed to be used with
        % gp.m. See also infMethods.m. In the EP algorithm,the sites are
        % updated in random order,for better performance when cases are ordered
        % according to the targets.
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch 2013-09-13.
        %
        % See also INFMETHODS.M.
        %
        persistent last_ttau last_tnu              % keep tilde parameters between calls
        tol = 1e-4; max_sweep = 10; min_sweep = 2;     % tolerance to stop EP iterations

        inf = 'infEP';
        if isnumeric(cov),K = cov;                    % use provided covariance matrix
        else K = feval(cov{:},hyp.cov,x); end       % evaluate the covariance matrix

        n = size(x,1);

        if isnumeric(mean),m = mean;                         % use provided mean vector
        else m = feval(mean{:},hyp.mean,x); end             % evaluate the mean vector

        % A note on naming: variables are given short but descriptive names in
        % accordance with Rasmussen & Williams "GPs for Machine Learning" (2006): mu
        % and s2 are mean and variance,nu and tau are natural parameters. A leading t
        % means tilde,a subscript _ni means "not i" (for cavity parameters),or _n
        % for a vector of cavity parameters. N(f|mu,Sigma) is the posterior.

        % marginal likelihood for ttau = tnu = zeros(n,1); equals n*log(2) for likCum*
        nlZ0 = -sum(feval(lik{:},hyp.lik,y,m,diag(K),inf));
        if any(size(last_ttau) ~= [n 1])      % find starting point for tilde parameters
            ttau = zeros(n,1); tnu  = zeros(n,1);        % init to zero if no better guess
            Sigma = K;                     % initialize Sigma and mu,the parameters of ..
            mu = m; nlZ = nlZ0;                  % .. the Gaussian posterior approximation
        else
            ttau = last_ttau; tnu  = last_tnu;   % try the tilde values from previous call
            [Sigma,mu,L,alpha,nlZ] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,inf);
            if nlZ > nlZ0                                           % if zero is better ..
                ttau = zeros(n,1); tnu  = zeros(n,1);       % .. then init with zero instead
                Sigma = K;                   % initialize Sigma and mu,the parameters of ..
                mu = m; nlZ = nlZ0;                % .. the Gaussian posterior approximation
            end
        end

        nlZ_old = Inf; sweep = 0;               % converged,max. sweeps or min. sweeps?
        while (abs(nlZ-nlZ_old) > tol && sweep < max_sweep) || sweep<min_sweep
            nlZ_old = nlZ; sweep = sweep+1;
            for i = randperm(n)       % iterate EP updates (in random order) over examples
                tau_ni = 1/Sigma(i,i)-ttau(i);      %  first find the cavity distribution ..
                nu_ni = mu(i)/Sigma(i,i)-tnu(i);                % .. params tau_ni and nu_ni

                % compute the desired derivatives of the indivdual log partition function
                [lZ,dlZ,d2lZ] = feval(lik{:},hyp.lik,y(i),nu_ni/tau_ni,1/tau_ni,inf);
                ttau_old = ttau(i); tnu_old = tnu(i);  % find the new tilde params,keep old
                ttau(i) = -d2lZ /(1+d2lZ/tau_ni);
                ttau(i) = max(ttau(i),0); % enforce positivity i.e. lower bound ttau by zero
                tnu(i)  = ( dlZ - nu_ni/tau_ni*d2lZ )/(1+d2lZ/tau_ni);

                dtt = ttau(i)-ttau_old; dtn = tnu(i)-tnu_old;      % rank-1 update Sigma ..
                si = Sigma(:,i); ci = dtt/(1+dtt*si(i));
                Sigma = Sigma - ci*si*si';                         % takes 70% of total time
                mu = mu - (ci*(mu(i)+si(i)*dtn)-dtn)*si;               % .. and recompute mu
            end
            % recompute since repeated rank-one updates can destroy numerical precision
            [Sigma,mu,L,alpha,nlZ] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,inf);
        end

        %         if sweep  ==  max_sweep && abs(nlZ-nlZ_old) > tol
        %             error('maximum number of sweeps exceeded in function infEP')
        %         end

        last_ttau = ttau; last_tnu = tnu;                       % remember for next call
        post.alpha = alpha; post.sW = sqrt(ttau); post.L = L;  % return posterior params

        if nargout>2                                           % do we want derivatives?
            dnlZ = hyp;                                   % allocate space for derivatives
            tau_n = 1./diag(Sigma)-ttau;             % compute the log marginal likelihood
            nu_n  = mu./diag(Sigma)-tnu;                    % vectors of cavity parameters
            sW = sqrt(ttau);
            F = alpha*alpha'-repmat(sW,1,n).*(L\(L'\diag(sW)));   % covariance hypers
            [K,dK] = feval(cov{:},hyp.cov,x,[]);
            for i = 1:length(hyp.cov)
                dnlZ.cov(i) = -sum(sum(F.*dK{i}))/2;
            end
            for i = 1:numel(hyp.lik)                                   % likelihood hypers
                dlik = feval(lik{:},hyp.lik,y,nu_n./tau_n,1./tau_n,inf,i);
                dnlZ.lik(i) = -sum(dlik);
            end
            [junk,dlZ] = feval(lik{:},hyp.lik,y,nu_n./tau_n,1./tau_n,inf);% mean hyps
            for i = 1:numel(hyp.mean)
                dm = feval(mean{:},hyp.mean,x,i);
                dnlZ.mean(i) = -dlZ'*dm;
            end
        end
    end

    function [Sigma,mu,L,alpha,nlZ] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,inf)
        % function to compute the parameters of the Gaussian approximation,Sigma and
        % mu,and the negative log marginal likelihood,nlZ,from the current site
        % parameters,ttau and tnu. Also returns L (useful for predictions).
        %
        n = length(y);                                      % number of training cases
        sW = sqrt(ttau);                                        % compute Sigma and mu
        L = chol(eye(n)+sW*sW'.*K);                            % L'*L = B = eye(n)+sW*K*sW
        V = L'\(repmat(sW,1,n).*K);
        Sigma = K - V'*V;
        alpha = tnu-sW.*(L\(L'\(sW.*(K*tnu+m))));
        mu = K*alpha+m; v = diag(Sigma);

        tau_n = 1./diag(Sigma)-ttau;             % compute the log marginal likelihood
        nu_n  = mu./diag(Sigma)-tnu;                    % vectors of cavity parameters
        lZ = feval(lik{:},hyp.lik,y,nu_n./tau_n,1./tau_n,inf);
        p = tnu-m.*ttau; q = nu_n-m.*tau_n;                        % auxiliary vectors
        nlZ = sum(log(diag(L))) - sum(lZ) - p'*Sigma*p/2 + (v'*p.^2)/2 ...
            - q'*((ttau./tau_n.*q-2*p).*v)/2 - sum(log(1+ttau./tau_n))/2;

    end

    function A = meanConst(hyp,x,i)
        % Constant mean function. The mean function is parameterized as:
        %
        % m(x) = c
        %
        % The hyperparameter is:
        %
        % hyp = [ c ]
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2010-08-04.
        %
        % See also MEANFUNCTIONS.M.
        %
        if nargin<2,A = '1'; return; end             % report number of hyperparameters
        if numel(hyp)~=1,error('Exactly one hyperparameter needed.'),end
        c = hyp;
        if nargin == 2
            A = c*ones(size(x,1),1);                                       % evaluate mean
        else
            if i == 1
                A = ones(size(x,1),1);                                          % derivative
            else
                A = zeros(size(x,1),1);
            end
        end
    end

    function [varargout] = likErf(hyp,y,mu,s2,inf)
        % likErf - Error function or cumulative Gaussian likelihood function for binary
        % classification or probit regression. The expression for the likelihood is
        %   likErf(t) = (1+erf(t/sqrt(2)))/2 = normcdf(t).
        %
        % Several modes are provided,for computing likelihoods,derivatives and moments
        % respectively,see likFunctions.m for the details. In general,care is taken
        % to avoid numerical issues when the arguments are extreme.
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2014-03-19.
        %
        % See also LIKFUNCTIONS.M.
        %
        if nargin<3,varargout = {'0'}; return; end   % report number of hyperparameters
        if nargin>1,y = sign(y); y(y == 0) = 1; else y = 1; end % allow only +/- 1 values
        if numel(y) == 0,y = 1; end

        if nargin<5                              % prediction mode if inf is not present
            y = y.*ones(size(mu));                                       % make y a vector
            s2zero = 1; if nargin>3&&numel(s2)>0&&norm(s2)>eps,s2zero = 0; end  % s2 == 0 ?
            if s2zero                                         % log probability evaluation
                lp = logphi(y.*mu);
            else                                                              % prediction
                lp = likErf(hyp,y,mu,s2,'infEP');
            end
            p = exp(lp); ymu = {}; ys2 = {};
            if nargout>1
                ymu = 2*p-1;                                                % first y moment
                if nargout>2
                    ys2 = 4*p.*(1-p);                                        % second y moment
                end
            end
            varargout = {lp,ymu,ys2};
        else                                                            % inference mode
            switch inf
                case 'infLaplace'
                    if nargin<6                                             % no derivative mode
                        f = mu; yf = y.*f;                            % product latents and labels
                        varargout = cell(nargout,1); [varargout{:}] = logphi(yf);   % query logphi
                        if nargout>1
                            varargout{2} = y.*varargout{2};
                            if nargout>3,varargout{4} = y.*varargout{4}; end
                        end
                    else                                                       % derivative mode
                        varargout = {[],[],[]};                         % derivative w.r.t. hypers
                    end

                case 'infEP'
                    if nargin<6                                             % no derivative mode
                        z = mu./sqrt(1+s2); dlZ = {}; d2lZ = {};
                        if numel(y) > 0,z = z.*y; end
                        if nargout <= 1,lZ = logphi(z);                         % log part function
                        else          [lZ,n_p] = logphi(z); end
                        if nargout > 1
                            if numel(y) == 0,y = 1; end
                            dlZ = y.*n_p./sqrt(1+s2);                      % 1st derivative wrt mean
                            if nargout>2,d2lZ = -n_p.*(z+n_p)./(1+s2); end         % 2nd derivative
                        end
                        varargout = {lZ,dlZ,d2lZ};
                    else                                                       % derivative mode
                        varargout = {[]};                                     % deriv. wrt hyp.lik
                    end
            end
        end
    end

    function [lp,dlp,d2lp,d3lp] = logphi(z)
        % Safe computation of logphi(z) = log(normcdf(z)) and its derivatives
        %                    dlogphi(z) = normpdf(x)/normcdf(x).
        % The function is based on idx 5725 in Hart et al. and gsl_sf_log_erfc_e.
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2013-11-13.
        %
        z = real(z);                                 % support for real arguments only
        lp = zeros(size(z));                                         % allocate memory
        id1 = z.*z<0.0492;                                 % first case: close to zero
        lp0 = -z(id1)/sqrt(2*pi);
        c = [ 0.00048204; -0.00142906; 0.0013200243174; 0.0009461589032;
            -0.0045563339802; 0.00556964649138; 0.00125993961762116;
            -0.01621575378835404; 0.02629651521057465; -0.001829764677455021;
            2*(1-pi/3); (4-pi)/3; 1; 1];
        f = 0; for i = 1:14,f = lp0.*(c(i)+f); end,lp(id1) = -2*f-log(2);
        id2 = z<-11.3137;                                    % second case: very small
        r = [ 1.2753666447299659525; 5.019049726784267463450;
            6.1602098531096305441; 7.409740605964741794425;
            2.9788656263939928886 ];
        q = [ 2.260528520767326969592;  9.3960340162350541504;
            12.048951927855129036034; 17.081440747466004316;
            9.608965327192787870698;  3.3690752069827527677 ];
        num = 0.5641895835477550741; for i = 1:5,num = -z(id2).*num/sqrt(2) + r(i); end
        den = 1.0;                   for i = 1:6,den = -z(id2).*den/sqrt(2) + q(i); end
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

function [predict_function,CGP_model] = classifyGaussProcess...
    (X,Class,hyp)
% generate gaussian process classifier model
% version 6,this version is assembly of gpml-3.6 EP method
% X is x_number x variable_number matirx,Y is x_number x 1 matrix
% low_bou,up_bou is 1 x variable_number matrix
% only support binary classification,-1 and 1
%
% input:
% X,Class,hyp(mean,cov(len,eta))
%
% abbreviation:
% pred: predicted,nomlz: normalization,num: number
% var: variance
%
[x_number,variable_number] = size(X);
if nargin < 5
    hyp.mean = 0;
    hyp.cov = zeros(1,2);
end

% normalization data
aver_X = mean(X);
stdD_X = std(X);
idx__ = find(stdD_X == 0);
if  ~isempty(idx__),stdD_X(idx__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;

object_function = @(x) objectNLLGPC(x,{@infEP},{@meanConst},{@calCov},{@likErf},X_nomlz,Class);
hyp_x = [hyp.mean,hyp.cov];

% [obj,gradient] = object_function(hyp_x)
% [obj_differ,gradient_differ] = differ(object_function,hyp_x)

hyp_low_bou = -3*ones(1,3);
hyp_up_bou = 3*ones(1,3);
hyp_x = fmincon(object_function,hyp_x,[],[],[],[],hyp_low_bou,hyp_up_bou,[],...
    optimoptions('fmincon','Display','none','SpecifyObjectiveGradient',true,...
    'MaxFunctionEvaluations',20,'OptimalityTolerance',1e-3));

% hyp.mean = hyp_x(1);
hyp.mean = 0;
hyp.cov = hyp_x(2:end);
hyp.lik = [];
post = infEP(hyp,{@meanConst},{@calCov},{@likErf},X_nomlz,Class);
predict_function = @(x_pred) classifyGaussPredictor...
    (x_pred,hyp,{@meanConst},{@calCov},{@likErf},post,X_nomlz,aver_X,stdD_X);

% output model
CGP_model.X = X;
CGP_model.Class = Class;
CGP_model.X_nomlz = X_nomlz;
CGP_model.aver_X = aver_X;
CGP_model.stdD_X = stdD_X;
CGP_model.predict_function = predict_function;
CGP_model.hyp = hyp;

    function [obj,gradient] = objectNLLGPC(x,inf,mean,cov,lik,X,Y)
        hyp_iter.mean = x(1);
        hyp_iter.cov = x(2:end);
        hyp_iter.lik = [];

        if nargout < 2
            [~,nlZ] = feval(inf{:},hyp_iter,mean,cov,lik,X,Y);
            obj = nlZ;
        elseif nargout < 3
            [~,nlZ,dnlZ] = feval(inf{:},hyp_iter,mean,cov,lik,X,Y);
            obj = nlZ;
            gradient = [dnlZ.mean,dnlZ.cov];
        end
    end

    function [class,possibility,miu_pre,var_pre] = classifyGaussPredictor...
            (X_pred,hyp,mean,cov,lik,post,X,aver_X,stdD_X)
        % predict function
        %
        X_pred_nomlz = (X_pred-aver_X)./stdD_X;
        pred_num = size(X_pred_nomlz,1);
        ys = ones(pred_num,1);

        alpha = post.alpha; L = post.L; sW = post.sW;
        nz = true(size(alpha,1),1);               % non-sparse representation
        %verify whether L contains valid Cholesky decomposition or something different
        Lchol = isnumeric(L) && all(all(tril(L,-1)==0)&diag(L)'>0&isreal(diag(L))');
        ns = size(X_pred_nomlz,1);                                       % number of data points
        nperbatch = 1000;                       % number of data points per mini batch
        nact = 0;                       % number of already processed test data points
        ymu = zeros(ns,1); ys2 = ymu; miu_pre = ymu; var_pre = ymu; possibility = ymu;   % allocate mem
        while nact<ns               % process minibatches of test cases to save memory
            id = (nact+1):min(nact+nperbatch,ns);               % data points to process
            kss = feval(cov{:},hyp.cov,X_pred_nomlz(id,:),'diag');              % self-variance
            Ks = feval(cov{:},hyp.cov,X(nz,:),X_pred_nomlz(id,:));        % avoid computation
            ms = feval(mean{:},hyp.mean,X_pred_nomlz(id,:));
            N = size(alpha,2);  % number of alphas (usually 1; more in case of sampling)
            Fmu = repmat(ms,1,N) + Ks'*full(alpha(nz,:));        % conditional mean fs|f
            miu_pre(id) = sum(Fmu,2)/N;                                   % predictive means
            if Lchol    % L contains chol decomp => use Cholesky parameters (alpha,sW,L)
                V  = L'\(repmat(sW,1,length(id)).*Ks);
                var_pre(id) = kss - sum(V.*V,1)';                       % predictive variances
            else                % L is not triangular => use alternative parametrisation
                if isnumeric(L),LKs = L*Ks; else LKs = L(Ks); end    % matrix or callback
                var_pre(id) = kss + sum(Ks.*LKs,1)';                    % predictive variances
            end
            var_pre(id) = max(var_pre(id),0);   % remove numerical noise i.e. negative variances
            Fs2 = repmat(var_pre(id),1,N);     % we have multiple values in case of sampling
            if nargin<9
                [Lp,Ymu,Ys2] = feval(lik{:},hyp.lik,[],Fmu(:),Fs2(:));
            else
                Ys = repmat(ys(id),1,N);
                [Lp,Ymu,Ys2] = feval(lik{:},hyp.lik,Ys(:),Fmu(:),Fs2(:));
            end
            possibility(id)  = sum(reshape(Lp,[],N),2)/N;    % log probability; sample averaging
            ymu(id) = sum(reshape(Ymu,[],N),2)/N;          % predictive mean ys|y and ..
            ys2(id) = sum(reshape(Ys2,[],N),2)/N;                          % .. variance
            nact = id(end);          % set counter to idx of last processed data point
        end

        possibility = exp(possibility);
        class = ones(pred_num,1);
        idx_list = find(possibility < 0.5);
        class(idx_list) = -1;
    end

    function [K,dK_dcov] = calCov(cov,X,Z)
        % obtain covariance of x
        % cov: eta,len(equal to 1/len.^2)
        %
        % k = eta*exp(-sum(x_dis*len)/vari_num);
        %
        [x_num,vari_num] = size(X);

        len = exp(cov(1));
        eta = exp(cov(2));

        % predict
        if nargin > 2 && nargout < 2 && ~isempty(Z)
            if strcmp(Z,'diag')
                K = eta;
            else
                [z_number,vari_num] = size(Z);
                % initializate square of X inner distance/ vari_num
                K = zeros(x_num,z_number);
                for len_idx = 1:vari_num
                    K = K+(X(:,len_idx)-Z(:,len_idx)').^2*len/vari_num;
                end
                K = eta*exp(-K);
            end
        else
            % initializate square of X inner distance sq
            sq_dis_v = zeros(x_num,x_num,vari_num);
            for len_idx = 1:vari_num
                sq_dis_v(:,:,len_idx) = (X(:,len_idx)-X(:,len_idx)').^2/vari_num;
            end

            % exp of x__x with theta
            exp_dis = zeros(x_num);
            for len_idx = 1:vari_num
                exp_dis = exp_dis+sq_dis_v(:,:,len_idx)*len;
            end
            exp_dis = exp(-exp_dis);
            K = exp_dis*eta;

            if nargout >= 2
                dK_dcov = cell(1,2);
                dK_dlen = zeros(x_num,x_num);
                for len_idx = 1:vari_num
                    dK_dlen = dK_dlen + sq_dis_v(:,:,len_idx);
                end
                dK_dlen = -dK_dlen.*K*len;
                dK_dcov{1} = dK_dlen;

                dK_dcov{2} = K;
            end
        end

    end

    function [post nlZ dnlZ] = infEP(hyp,mean,cov,lik,x,y)
        % Expectation Propagation approximation to the posterior Gaussian Process.
        % The function takes a specified covariance function (see covFunctions.m) and
        % likelihood function (see likFunctions.m),and is designed to be used with
        % gp.m. See also infMethods.m. In the EP algorithm,the sites are
        % updated in random order,for better performance when cases are ordered
        % according to the targets.
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch 2013-09-13.
        %
        % See also INFMETHODS.M.
        %
        persistent last_ttau last_tnu              % keep tilde parameters between calls
        tol = 1e-4; max_sweep = 10; min_sweep = 2;     % tolerance to stop EP iterations

        inf = 'infEP';
        n = size(x,1);
        if isnumeric(cov),K = cov;                    % use provided covariance matrix
        else K = feval(cov{:},hyp.cov,x); end       % evaluate the covariance matrix
        if isnumeric(mean),m = mean;                         % use provided mean vector
        else m = feval(mean{:},hyp.mean,x); end             % evaluate the mean vector

        % A note on naming: variables are given short but descriptive names in
        % accordance with Rasmussen & Williams "GPs for Machine Learning" (2006): mu
        % and s2 are mean and variance,nu and tau are natural parameters. A leading t
        % means tilde,a subscript _ni means "not i" (for cavity parameters),or _n
        % for a vector of cavity parameters. N(f|mu,Sigma) is the posterior.

        % marginal likelihood for ttau = tnu = zeros(n,1); equals n*log(2) for likCum*
        nlZ0 = -sum(feval(lik{:},hyp.lik,y,m,diag(K),inf));
        if any(size(last_ttau) ~= [n 1])      % find starting point for tilde parameters
            ttau = zeros(n,1); tnu  = zeros(n,1);        % init to zero if no better guess
            Sigma = K;                     % initialize Sigma and mu,the parameters of ..
            mu = m; nlZ = nlZ0;                  % .. the Gaussian posterior approximation
        else
            ttau = last_ttau; tnu  = last_tnu;   % try the tilde values from previous call
            [Sigma,mu,L,alpha,nlZ] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,inf);
            if nlZ > nlZ0                                           % if zero is better ..
                ttau = zeros(n,1); tnu  = zeros(n,1);       % .. then init with zero instead
                Sigma = K;                   % initialize Sigma and mu,the parameters of ..
                mu = m; nlZ = nlZ0;                % .. the Gaussian posterior approximation
            end
        end

        nlZ_old = Inf; sweep = 0;               % converged,max. sweeps or min. sweeps?
        while (abs(nlZ-nlZ_old) > tol && sweep < max_sweep) || sweep<min_sweep
            nlZ_old = nlZ; sweep = sweep+1;
            for i = randperm(n)       % iterate EP updates (in random order) over examples
                tau_ni = 1/Sigma(i,i)-ttau(i);      %  first find the cavity distribution ..
                nu_ni = mu(i)/Sigma(i,i)-tnu(i);                % .. params tau_ni and nu_ni

                % compute the desired derivatives of the indivdual log partition function
                [lZ,dlZ,d2lZ] = feval(lik{:},hyp.lik,y(i),nu_ni/tau_ni,1/tau_ni,inf);
                ttau_old = ttau(i); tnu_old = tnu(i);  % find the new tilde params,keep old
                ttau(i) =                     -d2lZ  /(1+d2lZ/tau_ni);
                ttau(i) = max(ttau(i),0); % enforce positivity i.e. lower bound ttau by zero
                tnu(i)  = ( dlZ - nu_ni/tau_ni*d2lZ )/(1+d2lZ/tau_ni);

                dtt = ttau(i)-ttau_old; dtn = tnu(i)-tnu_old;      % rank-1 update Sigma ..
                si = Sigma(:,i); ci = dtt/(1+dtt*si(i));
                Sigma = Sigma - ci*si*si';                         % takes 70% of total time
                mu = mu - (ci*(mu(i)+si(i)*dtn)-dtn)*si;               % .. and recompute mu
            end
            % recompute since repeated rank-one updates can destroy numerical precision
            [Sigma,mu,L,alpha,nlZ] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,inf);
        end

        %         if sweep == max_sweep && abs(nlZ-nlZ_old) > tol
        %             error('maximum number of sweeps exceeded in function infEP')
        %         end

        last_ttau = ttau; last_tnu = tnu;                       % remember for next call
        post.alpha = alpha; post.sW = sqrt(ttau); post.L = L;  % return posterior params

        if nargout>2                                           % do we want derivatives?
            dnlZ = hyp;                                   % allocate space for derivatives
            tau_n = 1./diag(Sigma)-ttau;             % compute the log marginal likelihood
            nu_n  = mu./diag(Sigma)-tnu;                    % vectors of cavity parameters
            sW = sqrt(ttau);
            F = alpha*alpha'-repmat(sW,1,n).*(L\(L'\diag(sW)));   % covariance hypers
            [K,dK] = feval(cov{:},hyp.cov,x,[]);
            for i = 1:length(hyp.cov)
                dnlZ.cov(i) = -sum(sum(F.*dK{i}))/2;
            end
            for i = 1:numel(hyp.lik)                                   % likelihood hypers
                dlik = feval(lik{:},hyp.lik,y,nu_n./tau_n,1./tau_n,inf,i);
                dnlZ.lik(i) = -sum(dlik);
            end
            [junk,dlZ] = feval(lik{:},hyp.lik,y,nu_n./tau_n,1./tau_n,inf);% mean hyps
            for i = 1:numel(hyp.mean)
                dm = feval(mean{:},hyp.mean,x,i);
                dnlZ.mean(i) = -dlZ'*dm;
            end
        end
    end

    function [Sigma,mu,L,alpha,nlZ] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,inf)
        % function to compute the parameters of the Gaussian approximation,Sigma and
        % mu,and the negative log marginal likelihood,nlZ,from the current site
        % parameters,ttau and tnu. Also returns L (useful for predictions).
        %
        n = length(y);                                      % number of training cases
        sW = sqrt(ttau);                                        % compute Sigma and mu
        L = chol(eye(n)+sW*sW'.*K);                            % L'*L = B = eye(n)+sW*K*sW
        V = L'\(repmat(sW,1,n).*K);
        Sigma = K - V'*V;
        alpha = tnu-sW.*(L\(L'\(sW.*(K*tnu+m))));
        mu = K*alpha+m; v = diag(Sigma);

        tau_n = 1./diag(Sigma)-ttau;             % compute the log marginal likelihood
        nu_n  = mu./diag(Sigma)-tnu;                    % vectors of cavity parameters
        lZ = feval(lik{:},hyp.lik,y,nu_n./tau_n,1./tau_n,inf);
        p = tnu-m.*ttau; q = nu_n-m.*tau_n;                        % auxiliary vectors
        nlZ = sum(log(diag(L))) - sum(lZ) - p'*Sigma*p/2 + (v'*p.^2)/2 ...
            - q'*((ttau./tau_n.*q-2*p).*v)/2 - sum(log(1+ttau./tau_n))/2;
    end

    function A = meanConst(hyp,x,i)

        % Constant mean function. The mean function is parameterized as:
        %
        % m(x) = c
        %
        % The hyperparameter is:
        %
        % hyp = [ c ]
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2010-08-04.
        %
        % See also MEANFUNCTIONS.M.

        if nargin<2,A = '1'; return; end             % report number of hyperparameters
        if numel(hyp)~=1,error('Exactly one hyperparameter needed.'),end
        c = hyp;
        if nargin==2
            A = c*ones(size(x,1),1);                                       % evaluate mean
        else
            if i==1
                A = ones(size(x,1),1);                                          % derivative
            else
                A = zeros(size(x,1),1);
            end
        end
    end

    function [varargout] = likErf(hyp,y,mu,s2,inf,i)
        % likErf - Error function or cumulative Gaussian likelihood function for binary
        % classification or probit regression. The expression for the likelihood is
        %   likErf(t) = (1+erf(t/sqrt(2)))/2 = normcdf(t).
        %
        % Several modes are provided,for computing likelihoods,derivatives and moments
        % respectively,see likFunctions.m for the details. In general,care is taken
        % to avoid numerical issues when the arguments are extreme.
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2014-03-19.
        %
        % See also LIKFUNCTIONS.M.
        %
        if nargin<3,varargout = {'0'}; return; end   % report number of hyperparameters
        if nargin>1,y = sign(y); y(y==0) = 1; else y = 1; end % allow only +/- 1 values
        if numel(y)==0,y = 1; end

        if nargin<5                              % prediction mode if inf is not present
            y = y.*ones(size(mu));                                       % make y a vector
            s2zero = 1; if nargin>3&&numel(s2)>0&&norm(s2)>eps,s2zero = 0; end  % s2==0 ?
            if s2zero                                         % log probability evaluation
                lp = logphi(y.*mu);
            else                                                              % prediction
                lp = likErf(hyp,y,mu,s2,'infEP');
            end
            p = exp(lp); ymu = {}; ys2 = {};
            if nargout>1
                ymu = 2*p-1;                                                % first y moment
                if nargout>2
                    ys2 = 4*p.*(1-p);                                        % second y moment
                end
            end
            varargout = {lp,ymu,ys2};
        else                                                            % inference mode
            switch inf
                case 'infLaplace'
                    if nargin<6                                             % no derivative mode
                        f = mu; yf = y.*f;                            % product latents and labels
                        varargout = cell(nargout,1); [varargout{:}] = logphi(yf);   % query logphi
                        if nargout>1
                            varargout{2} = y.*varargout{2};
                            if nargout>3,varargout{4} = y.*varargout{4}; end
                        end
                    else                                                       % derivative mode
                        varargout = {[],[],[]};                         % derivative w.r.t. hypers
                    end

                case 'infEP'
                    if nargin<6                                             % no derivative mode
                        z = mu./sqrt(1+s2); dlZ = {}; d2lZ = {};
                        if numel(y)>0,z = z.*y; end
                        if nargout <= 1,lZ = logphi(z);                         % log part function
                        else          [lZ,n_p] = logphi(z); end
                        if nargout > 1
                            if numel(y)==0,y = 1; end
                            dlZ = y.*n_p./sqrt(1+s2);                      % 1st derivative wrt mean
                            if nargout>2,d2lZ = -n_p.*(z+n_p)./(1+s2); end         % 2nd derivative
                        end
                        varargout = {lZ,dlZ,d2lZ};
                    else                                                       % derivative mode
                        varargout = {[]};                                     % deriv. wrt hyp.lik
                    end
            end
        end
    end

    function [lp,dlp,d2lp,d3lp] = logphi(z)
        % Safe computation of logphi(z) = log(normcdf(z)) and its derivatives
        %                    dlogphi(z) = normpdf(x)/normcdf(x).
        % The function is based on idx 5725 in Hart et al. and gsl_sf_log_erfc_e.
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2013-11-13.
        %
        z = real(z);                                 % support for real arguments only
        lp = zeros(size(z));                                         % allocate memory
        id1 = z.*z<0.0492;                                 % first case: close to zero
        lp0 = -z(id1)/sqrt(2*pi);
        c = [ 0.00048204; -0.00142906; 0.0013200243174; 0.0009461589032;
            -0.0045563339802; 0.00556964649138; 0.00125993961762116;
            -0.01621575378835404; 0.02629651521057465; -0.001829764677455021;
            2*(1-pi/3); (4-pi)/3; 1; 1];
        f = 0; for i = 1:14,f = lp0.*(c(i)+f); end,lp(id1) = -2*f-log(2);
        id2 = z<-11.3137;                                    % second case: very small
        r = [ 1.2753666447299659525; 5.019049726784267463450;
            6.1602098531096305441; 7.409740605964741794425;
            2.9788656263939928886 ];
        q = [ 2.260528520767326969592;  9.3960340162350541504;
            12.048951927855129036034; 17.081440747466004316;
            9.608965327192787870698;  3.3690752069827527677 ];
        num = 0.5641895835477550741; for i = 1:5,num = -z(id2).*num/sqrt(2) + r(i); end
        den = 1.0;                   for i = 1:6,den = -z(id2).*den/sqrt(2) + q(i); end
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

function [center_list,FC_model] = clusteringFuzzy(X,classify_number,m)
% get fuzzy cluster model
% X is x_number x variable_number matrix
% center_list is classify_number x variable_number matrix
%
iteration_max = 100;
torlance = 1e-6;

[x_number,variable_number] = size(X);

% normalization data
aver_X = mean(X);
stdD_X = std(X);
idx__ = find(stdD_X == 0);
if  ~isempty(idx__),stdD_X(idx__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;

% if x_number equal 1,clustering cannot done
if x_number == 1
    FC_model.X = X;
    FC_model.X_normalize = X_nomlz;
    FC_model.center_list = X;
    FC_model.obj_loss_list = [];
    return;
end

U = zeros(classify_number,x_number);
center_list = rand(classify_number,variable_number)*0.5;
iteration = 0;
done = 0;
obj_loss_list = zeros(iteration_max,1);

% get X_center_dis_sq
X_center_dis_sq = zeros(classify_number,x_number);
for classify_idx = 1:classify_number
    for x_idx = 1:x_number
        X_center_dis_sq(classify_idx,x_idx) = ...
            getSq((X_nomlz(x_idx,:)-center_list(classify_idx,:)));
    end
end

while ~done
    % updata classify matrix U
    for classify_idx = 1:classify_number
        for x_idx = 1:x_number
            U(classify_idx,x_idx) = ...
                1/sum((X_center_dis_sq(classify_idx,x_idx)./X_center_dis_sq(:,x_idx)).^(1/(m-1)));
        end
    end

    % updata center_list
    center_list_old = center_list;
    for classify_idx = 1:classify_number
        center_list(classify_idx,:) = ...
            sum((U(classify_idx,:)').^m.*X_nomlz,1)./...
            sum((U(classify_idx,:)').^m,1);
    end

    % updata X_center_dis_sq
    X_center_dis_sq = zeros(classify_number,x_number);
    for classify_idx = 1:classify_number
        for x_idx = 1:x_number
            X_center_dis_sq(classify_idx,x_idx) = ...
                getSq((X_nomlz(x_idx,:)-center_list(classify_idx,:)));
        end
    end

    %     plot(center_list(:,1),center_list(:,2));

    % forced interrupt
    if iteration > iteration_max
        done = 1;
    end

    % convergence judgment
    if sum(sum(center_list_old-center_list).^2)<torlance
        done = 1;
    end

    iteration = iteration+1;
    obj_loss_list(iteration) = sum(sum(U.^m.*X_center_dis_sq));
end
obj_loss_list(iteration+1:end) = [];
center_list = center_list.*stdD_X+aver_X;

FC_model.X = X;
FC_model.X_normalize = X_nomlz;
FC_model.center_list = center_list;
FC_model.obj_loss_list = obj_loss_list;

    function sq = getSq(dx)
        % dx is 1 x variable_number matrix
        %
        sq = dx*dx';
    end
end

%% surrogate model
function [object_function_surrogate,nonlcon_function_surrogate,output] = getSurrogateFunction...
    (X_MF,Obj_MF,Con_MF,Coneq_MF)
% base on input data to generate surrogate predict function
% nonlcon_function_surrogate if format of nonlcon function in fmincon
% judge MF and SF quality and select best one
%

[pred_func_obj,model_obj] = getBestModel(X_MF,Obj_MF);

if ~isempty(Con_MF)
    con_number = size(Con_MF{1},2);
    pred_funct_con = cell(1,con_number);
    model_con_list = cell(1,con_number);
    for con_idx = 1:con_number
        [pred_funct_con{con_idx},model_con_list{con_idx}] = getBestModel...
            (X_MF,{Con_MF{1}(:,con_idx),Con_MF{2}(:,con_idx)});
    end
else
    pred_funct_con = [];
    model_con_list = [];
end

if ~isempty(Coneq_MF)
    coneq_number = size(Coneq_MF{1},2);
    pred_funct_coneq = cell(1,coneq_number);
    model_coneq_list = cell(1,coneq_number);
    for coneq_idx = 1:size(Coneq_MF,2)
        [pred_funct_coneq{coneq_idx},model_coneq_list{coneq_idx}] = getBestModel...
            (X_MF,{Coneq_MF{1}(:,coneq_idx),Coneq_MF{2}(:,coneq_idx)},add_LF_flag);
    end
else
    pred_funct_coneq = [];
    model_coneq_list = [];
end

object_function_surrogate = @(X_predict) objectFunctionSurrogate(X_predict,pred_func_obj);
if isempty(pred_funct_con) && isempty(pred_funct_coneq)
    nonlcon_function_surrogate = [];
else
    nonlcon_function_surrogate = @(X_predict) nonlconFunctionSurrogate(X_predict,pred_funct_con,pred_funct_coneq);
end

output.model_obj=model_obj;
output.model_con_list=model_con_list;
output.model_coneq_list=model_coneq_list;

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
            for con_idx__ = 1:length(predict_function_con)
                con(:,con_idx__) = ....
                    predict_function_con{con_idx__}(X_predict);
            end
        end
        if isempty(predict_function_coneq)
            coneq = [];
        else
            coneq = zeros(size(X_predict,1),length(predict_function_coneq));
            for coneq_idx__ = 1:length(predict_function_coneq)
                coneq(:,coneq_idx__) = ...
                    predict_function_coneq{coneq_idx__}(X_predict);
            end
        end
    end

end

function [pred_func,model] = getBestModel(X_MF,Obj_MF)
% judge use single fidelity of mulit fidelity by R^2
%

x_HF_num = size(X_MF{1},1);
[pred_func_MF,model_obj_MF] = interpRadialBasisMultiFidelityPreModel...
    (X_MF{1},Obj_MF{1},[],X_MF{2},Obj_MF{2},[]);
error_MF = ( model_obj_MF.beta./diag(model_obj_MF.H(:,1:x_HF_num)\eye(x_HF_num))+...
    model_obj_MF.alpha./diag(model_obj_MF.H(:,x_HF_num+1:end)\eye(x_HF_num)) )*model_obj_MF.stdD_Y;
Rsq_MF = 1-sum(error_MF.^2)/sum((mean(Obj_MF{1})-Obj_MF{1}).^2);

[pred_func_SF,model_obj_SF] = interpRadialBasisPreModel...
    (X_MF{1},Obj_MF{1});
error_SF = (model_obj_SF.beta./diag(model_obj_SF.inv_radialbasis_matrix))*model_obj_SF.stdD_Y;
Rsq_SF = 1-sum(error_SF.^2)/sum((mean(Obj_MF{1})-Obj_MF{1}).^2);

if Rsq_MF > Rsq_SF
    pred_func = pred_func_MF;
    model = model_obj_MF;
    model.type='MF';
else
    pred_func = pred_func_SF;
    model = model_obj_SF;
    model.type='SF';
end

end

function [predict_function_RBFMF,model_RBFMF] = interpRadialBasisMultiFidelityPreModel...
    (XHF, YHF, varargin)
% multi fildelity radial basis function interp pre model function
% XHF, YHF are x_HF_number x variable_number matrix
% XLF, YLF are x_LF_number x variable_number matrix
% aver_X, stdD_X is 1 x x_HF_number matrix
%
% input:
% XHF, YHF, basis_func_HF(can be []), XLF, YLF, basis_func_LF(can be [])
% XHF, YHF, basis_func_HF(can be []), LF_model
%
% output:
% predict_function, HK_model
%
% reference: [1] LIU Y, WANG S, ZHOU Q, et al. Modified Multifidelity
% Surrogate Model Based on Radial Basis Function with Adaptive Scale Factor
% [J]. Chinese Journal of Mechanical Engineering, 2022, 35(1): 77.
%
% Copyright 2023 Adel
%
[x_HF_number, variable_number] = size(XHF);
switch nargin
    case 4
        basis_func_HF = varargin{1};
        LF_model = varargin{2};

        % check whether LF model exist predict_function
        if ~isfield(LF_model, 'predict_function')
            error('interpRadialBasisMultiFidelityPreModel: low fidelity lack predict function');
        end
    case 6
        basis_func_HF = varargin{1};
        XLF = varargin{2};
        YLF = varargin{3};
        basis_func_LF = varargin{4};

        [x_LF_number, variable_number] = size(XLF);

        % first step
        % construct low fidelity model

        % normalize data
        aver_X = mean(XLF);
        stdD_X = std(XLF);
        aver_Y = mean(YLF);
        stdD_Y = std(YLF);
        idx__ = find(stdD_X == 0);
        if  ~isempty(idx__), stdD_X(idx__) = 1; end
        idx__ = find(stdD_Y == 0);
        if  ~isempty(idx__), stdD_Y(idx__) = 1; end

        %         aver_X = 0;
        %         stdD_X = 1;
        %         aver_Y = 0;
        %         stdD_Y = 1;

        XLF_nomlz = (XLF-aver_X)./stdD_X;
        YLF_nomlz = (YLF-aver_Y)./stdD_Y;

        if isempty(basis_func_LF)
            basis_func_LF = @(r) r.^3;
            %             c = (prod(max(XLF_nomlz) - min(XLF_nomlz))/x_LF_number)^(1/variable_number);
            %             basis_func_LF = @(r) exp(-(r.^2)/c);
        end

        % initialization distance of XLF_nomlz
        XLF_dis = zeros(x_LF_number, x_LF_number);
        for variable_idx = 1:variable_number
            XLF_dis = XLF_dis + ...
                (XLF_nomlz(:, variable_idx) - XLF_nomlz(:, variable_idx)').^2;
        end
        XLF_dis = sqrt(XLF_dis);

        [beta_LF, rdibas_matrix_LF] = interpRadialBasis...
            (XLF_dis, YLF_nomlz, basis_func_LF, x_LF_number);

        % initialization predict function
        predict_function_LF = @(X_predict) interpRadialBasisPredictor...
            (X_predict, XLF_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
            x_LF_number, variable_number, beta_LF, basis_func_LF);

        LF_model.X = XLF;
        LF_model.Y = YLF;
        LF_model.radialbasis_matrix = rdibas_matrix_LF;
        LF_model.beta = beta_LF;

        LF_model.aver_X = aver_X;
        LF_model.stdD_X = stdD_X;
        LF_model.aver_Y = aver_Y;
        LF_model.stdD_Y = stdD_Y;
        LF_model.basis_function = basis_func_LF;

        LF_model.predict_function = predict_function_LF;
    otherwise
        error('interpRadialBasisMultiFidelityPreModel: error input');
end
model_RBFMF.LF_model = LF_model;
predict_function_LF = LF_model.predict_function;

% second step
% construct MFRBF model

% normalize data
aver_X = mean(XHF);
stdD_X = std(XHF);
aver_Y = mean(YHF);
stdD_Y = std(YHF);
idx__ = find(stdD_X == 0);
if ~isempty(idx__), stdD_X(idx__) = 1;end
idx__ = find(stdD_Y == 0);
if ~isempty(idx__), stdD_Y(idx__) = 1;end

% aver_X = 0;
% stdD_X = 1;
% aver_Y = 0;
% stdD_Y = 1;

XHF_nomlz = (XHF - aver_X)./stdD_X;
YHF_nomlz = (YHF - aver_Y)./stdD_Y;

% predict LF value at XHF point
YHF_pred = predict_function_LF(XHF);

% nomalizae
YHF_pred_nomlz = (YHF_pred - aver_Y)./stdD_Y;

if isempty(basis_func_HF)
    basis_func_HF = @(r) r.^3;
    %     c = (prod(max(XHF_nomlz) - min(XHF_nomlz))/x_HF_number)^(1/variable_number);
    %     basis_func_HF = @(r) exp(-(r.^2)/c);
end

% initialization distance of XHF_nomlz
XHF_dis = zeros(x_HF_number, x_HF_number);
for variable_idx = 1:variable_number
    XHF_dis = XHF_dis + ...
        (XHF_nomlz(:, variable_idx) - XHF_nomlz(:, variable_idx)').^2;
end
XHF_dis = sqrt(XHF_dis);

[omega, H, H_hessian, inv_H_hessian] = interpRadialBasisMultiFidelity...
    (XHF_dis, YHF_nomlz, basis_func_HF, x_HF_number, YHF_pred_nomlz);

% initialization predict function
predict_function_RBFMF = @(X_predict) interpRadialBasisMultiFidelityPredictor...
    (X_predict, XHF_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
    x_HF_number, variable_number, omega, basis_func_HF, predict_function_LF);

model_RBFMF.X = XHF;
model_RBFMF.Y = YHF;
model_RBFMF.H = H;
model_RBFMF.H_hessian = H_hessian;
model_RBFMF.inv_H_hessian = inv_H_hessian;
model_RBFMF.omega = omega;
model_RBFMF.alpha = omega(1:x_HF_number);
model_RBFMF.beta = omega(x_HF_number+1:end);

model_RBFMF.aver_X = aver_X;
model_RBFMF.stdD_X = stdD_X;
model_RBFMF.aver_Y = aver_Y;
model_RBFMF.stdD_Y = stdD_Y;
model_RBFMF.basis_function = basis_func_HF;

model_RBFMF.predict_function = predict_function_RBFMF;

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
        rdibas_matrix = rdibas_matrix + eye(x_number)*1e-9;

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
        X_pred_nomlz = (X_pred - aver_X)./stdD_X;

        % calculate distance
        X_dis_pred = zeros(x_pred_num, x_num);
        for vari_idx = 1:vari_num
            X_dis_pred = X_dis_pred + ...
                (X_pred_nomlz(:, vari_idx) - X_nomlz(:, vari_idx)').^2;
        end
        X_dis_pred = sqrt(X_dis_pred);

        % predict variance
        Y_pred = basis_function(X_dis_pred)*beta;

        % normalize data
        Y_pred = Y_pred*stdD_Y + aver_Y;
    end

    function [omega, H, H_hessian, inv_H_hessian] = interpRadialBasisMultiFidelity...
            (X_dis, Y, basis_function, x_number, YHF_pred)
        % interp polynomial responed surface core function
        % calculation beta
        %
        % Copyright 2022 Adel
        %
        rdibas_matrix = basis_function(X_dis);

        % add low fildelity value
        H = [rdibas_matrix.*YHF_pred, rdibas_matrix];
        H_hessian = (H*H');

        % stabilize matrix
        H_hessian = H_hessian + eye(x_number)*1e-6;

        % get inv matrix
        inv_H_hessian = H_hessian\eye(x_number);

        % solve omega
        omega = H'*(inv_H_hessian*Y);

        %         omega = H\Y;
    end

    function [Y_pred] = interpRadialBasisMultiFidelityPredictor...
            (X_pred, X_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
            x_num, vari_num, omega, basis_function, predict_function_LF)
        % radial basis function interpolation predict function
        %
        [x_pred_num, ~] = size(X_pred);

        % normalize data
        X_pred_nomlz = (X_pred - aver_X)./stdD_X;

        % calculate distance
        X_dis_pred = zeros(x_pred_num, x_num);
        for vari_idx = 1:vari_num
            X_dis_pred = X_dis_pred + ...
                (X_pred_nomlz(:, vari_idx) - X_nomlz(:, vari_idx)').^2;
        end
        X_dis_pred = sqrt(X_dis_pred);

        % predict low fildelity value
        Y_pred_LF = predict_function_LF(X_pred);

        % nomalizae
        Y_pred_LF_nomlz = (Y_pred_LF - aver_Y)./stdD_Y;

        % combine two matrix
        rdibas_matrix_pred = basis_function(X_dis_pred);
        H_pred = [rdibas_matrix_pred.*Y_pred_LF_nomlz, rdibas_matrix_pred];

        % predict variance
        Y_pred_nomlz = H_pred*omega;

        % normalize data
        Y_pred = Y_pred_nomlz*stdD_Y + aver_Y;
    end

end

function [predict_function,model_radialbasis] = interpRadialBasisPreModel...
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
idx__ = find(stdD_X == 0);
if ~isempty(idx__),stdD_X(idx__) = 1;end
idx__ = find(stdD_Y == 0);
if ~isempty(idx__),stdD_Y(idx__) = 1;end

% aver_X = 0;
% stdD_X = 1;
% aver_Y = 0;
% stdD_Y = 1;

X_nomlz = (X-aver_X)./stdD_X;
Y_nomlz = (Y-aver_Y)./stdD_Y;

if isempty(basis_function)
    %     c = (prod(max(X_nomlz)-min(X_nomlz))/x_number)^(1/variable_number);
    %     basis_function = @(r) exp(-(r.^2)/c);
    basis_function = @(r) r.^3;
end

% initialization distance of all X
X_dis = zeros(x_number,x_number);
for variable_idx = 1:variable_number
    X_dis = X_dis+(X_nomlz(:,variable_idx)-X_nomlz(:,variable_idx)').^2;
end
X_dis = sqrt(X_dis);

[beta,rdibas_matrix,inv_rdibas_matrix] = interpRadialBasis...
    (X_dis,Y_nomlz,basis_function,x_number);

% initialization predict function
predict_function = @(X_predict) interpRadialBasisPredictor...
    (X_predict,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_number,variable_number,beta,basis_function);

model_radialbasis.X = X;
model_radialbasis.Y = Y;
model_radialbasis.radialbasis_matrix = rdibas_matrix;
model_radialbasis.inv_radialbasis_matrix=inv_rdibas_matrix;
model_radialbasis.beta = beta;

model_radialbasis.aver_X = aver_X;
model_radialbasis.stdD_X = stdD_X;
model_radialbasis.aver_Y = aver_Y;
model_radialbasis.stdD_Y = stdD_Y;
model_radialbasis.basis_function = basis_function;

model_radialbasis.predict_function = predict_function;

% abbreviation:
% num: number,pred: predict,vari: variable
    function [beta,rdibas_matrix,inv_rdibas_matrix] = interpRadialBasis...
            (X_dis,Y,basis_function,x_number)
        % interp polynomial responed surface core function
        % calculation beta
        %
        % Copyright 2022 Adel
        %
        rdibas_matrix = basis_function(X_dis);

        % stabilize matrix
        rdibas_matrix = rdibas_matrix+eye(x_number)*1e-9;

        % get inverse matrix
        inv_rdibas_matrix = rdibas_matrix\eye(x_number);

        % solve beta
        beta = inv_rdibas_matrix*Y;
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
        for vari_idx = 1:vari_num
            X_dis_pred = X_dis_pred+...
                (X_pred_nomlz(:,vari_idx)-X_nomlz(:,vari_idx)').^2;
        end
        X_dis_pred = sqrt(X_dis_pred);

        % predict variance
        Y_pred = basis_function(X_dis_pred)*beta;

        % normalize data
        Y_pred = Y_pred*stdD_Y+aver_Y;
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

function [X_sample,dist_min_nomlz,X_total] = getLatinHypercube...
    (sample_number,variable_number,...
    low_bou,up_bou,X_exist,cheapcon_function)
% generate latin hypercube desgin
%
% more uniform point distribution by simulating particle motion
%
% input:
% sample number(new point to sample),variable_number
% low_bou,up_bou,x_exist_list,cheapcon_function
%
% output:
% X_sample,dist_min_nomlz(min distance of normalize data)
% X_total,include all data in area
%
% Copyright 2023 3 Adel
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

iteration_max = 100;

% check x_exist_list if meet boundary
if ~isempty(X_exist)
    if size(X_exist,2) ~= variable_number
        error('getLatinHypercube: x_exist_list variable_number error');
    end
    %     idx = find(X_exist < low_bou-eps);
    %     idx = [idx;find(X_exists > up_bou+eps)];
    %     if ~isempty(idx)
    %         error('getLatinHypercube: x_exist_list range error');
    %     end
    X_exist_nomlz = (X_exist-low_bou)./(up_bou-low_bou);
else
    X_exist_nomlz = [];
end

exist_number = size(X_exist,1);
total_number = sample_number+exist_number;
if sample_number < 0
    X_total = X_exist;
    X_sample = [];
    dist_min_nomlz = calMinDistance(X_exist_nomlz);
    return;
end

low_bou_nomlz = zeros(1,variable_number);
up_bou_nomlz = ones(1,variable_number);

% obtain initial point
if ~isempty(cheapcon_function)
    % obtian feasible point
    X_quasi_nomlz = [];

    % check if have enough X_supply_nomlz
    iteration = 0;
    while size(X_quasi_nomlz,1) < 10*sample_number && iteration < 500
        X_quasi_nomlz_initial = rand(10*sample_number,variable_number);

        qusai_idx = [];
        for x_idx = 1:size(X_quasi_nomlz_initial,1)
            if cheapcon_function(X_quasi_nomlz_initial(x_idx,:).*(up_bou-low_bou)+low_bou) <= 0
                qusai_idx = [qusai_idx,x_idx];
            end
        end
        X_quasi_nomlz = [X_quasi_nomlz;X_quasi_nomlz_initial(qusai_idx,:)];

        iteration = iteration+1;
    end

    if iteration == 500 && size(X_quasi_nomlz,1) < sample_number
        error('getLatinHypercube: feasible quasi point cannot be found');
    end

    % use fuzzy clustering to get feasible point center
    X_sample_nomlz = clusteringFuzzy(X_quasi_nomlz,sample_number,2);
    X_feas_center_nomlz = X_sample_nomlz;

    %     scatter(X_quasi_nomlz(:,1),X_quasi_nomlz(:,2));
    %     hold on;
    %     scatter(X_sample_nomlz(:,1),X_sample_nomlz(:,2),'red');
    %     hold off;
else
    X_sample_nomlz = rand(sample_number,variable_number);
end

% pic_num = 1;

iteration = 0;
obj_list = zeros(sample_number,1);
gradient_list = zeros(sample_number,variable_number);
while iteration < iteration_max
    % change each x place by newton methods
    for x_idx = 1:sample_number

        % get gradient
        [obj_list(x_idx,1),gradient_list(x_idx,:)] = calParticleEnergy...
            (X_sample_nomlz(x_idx,:),[X_sample_nomlz(1:x_idx-1,:);X_sample_nomlz(x_idx+1:end,:);X_exist_nomlz],...
            sample_number,variable_number,low_bou_nomlz-1/2/sample_number,up_bou_nomlz+1/2/sample_number);

        %         energy_function = @(x) calParticleEnergy...
        %             ([x],[X_sample_nomlz(1:x_idx-1,:);X_sample_nomlz(x_idx+1:end,:);X_exist_nomlz],...
        %             sample_number,variable_number,low_bou_nomlz-1/2/sample_number,up_bou_nomlz+1/2/sample_number);
        %         drawFunction(energy_function,low_bou_nomlz(1:2),up_bou_nomlz(1:2))

        %         [obj,gradient] = differ(energy_function,X_sample_nomlz(x_idx,:));
        %         gradient'-gradient_list(x_idx,:)
    end

    C = (1-iteration/iteration_max)*0.5;

    % updata partical location
    for x_idx = 1:sample_number
        x = X_sample_nomlz(x_idx,:);
        gradient = gradient_list(x_idx,:);

        % check if feasible
        if ~isempty(cheapcon_function)
            con = cheapcon_function(x.*(up_bou-low_bou)+low_bou);
            % if no feasible,move point to close point
            if con > 0
                %                 % search closest point
                %                 dx_center = x-X_feas_center_nomlz;
                %                 [~,idx] = min(norm(dx_center,"inf"));
                %                 gradient = dx_center(idx(1),:);

                gradient = x-X_feas_center_nomlz(x_idx,:);
            end
        end

        gradient = min(gradient,0.5);
        gradient = max(gradient,-0.5);
        x = x-gradient*C;

        boolean = x < low_bou_nomlz;
        x(boolean) = -x(boolean);
        boolean = x > up_bou_nomlz;
        x(boolean) = 2-x(boolean);
        X_sample_nomlz(x_idx,:) = x;
    end

    iteration = iteration+1;

    %     scatter(X_sample_nomlz(:,1),X_sample_nomlz(:,2));
    %     bou = [low_bou_nomlz(1:2);up_bou_nomlz(1:2)];
    %     axis(bou(:));
    %     grid on;
    %
    %     radius = 1;
    %     hold on;
    %     rectangle('Position',[-radius,-radius,2*radius,2*radius],'Curvature',[1 1])
    %     hold off;
    %
    %     drawnow;
    %     F = getframe(gcf);
    %     I = frame2im(F);
    %     [I,map] = rgb2ind(I,256);
    %     if pic_num  =  =  1
    %         imwrite(I,map,'show_trajectory_constrain.gif','gif','Loopcount',inf,'DelayTime',0.1);
    %     else
    %         imwrite(I,map,'show_trajectory_constrain.gif','gif','WriteMode','append','DelayTime',0.1);
    %     end
    %     pic_num = pic_num + 1;
end

% process out of boundary point
for x_idx = 1:sample_number
    x = X_sample_nomlz(x_idx,:);
    % check if feasible
    if ~isempty(cheapcon_function)
        con = cheapcon_function(x.*(up_bou-low_bou)+low_bou);
        % if no feasible,move point to close point
        if con > 0
            % search closest point
            dx_center = x-X_feas_center_nomlz;
            [~,idx] = min(norm(dx_center,"inf"));

            gradient = dx_center(idx(1),:);
        end
    end
    x = x-gradient*C;

    boolean = x < low_bou_nomlz;
    x(boolean) = -x(boolean);
    boolean = x > up_bou_nomlz;
    x(boolean) = 2-x(boolean);
    X_sample_nomlz(x_idx,:) = x;
end
X_sample_nomlz=max(X_sample_nomlz,low_bou_nomlz);
X_sample_nomlz=min(X_sample_nomlz,up_bou_nomlz);

dist_min_nomlz = calMinDistance([X_sample_nomlz;X_exist_nomlz]);
X_sample = X_sample_nomlz.*(up_bou-low_bou)+low_bou;
X_total = [X_sample;X_exist];

    function [obj,gradient] = calParticleEnergy...
            (x,X_surplus,sample_number,variable_number,low_bou,up_bou)
        % function describe distance between X and X_supply
        % x is colume vector and X_surplus is matrix which is num-1 x var
        % low_bou_limit__ and up_bou_limit__ is colume vector
        % variable in colume
        %
        a__ = 10;
        a_bou__ = 10;

        sign__ = ((x > X_surplus)-0.5)*2;

        xi__ = -a__*(x-X_surplus).*sign__;
        psi__ = a_bou__*(low_bou-x);
        zeta__ = a_bou__*(x-up_bou);

        exp_psi__ = exp(psi__);
        exp_zeta__ = exp(zeta__);

        %         sum_xi__ = sum(xi__,2)/variable_number;
        %         exp_sum_xi__ = exp(sum_xi__);
        %         % get obj
        %         obj = sum(exp_sum_xi__,1)+...
        %             sum(exp_psi__+exp_zeta__,2)/variable_number;

        %         exp_xi__ = exp(xi__);
        %         sum_exp_xi__ = sum(exp_xi__,2);
        %         % get obj
        %         obj = sum(sum_exp_xi__,1)/variable_number/sample_number+...
        %             sum(exp_psi__+exp_zeta__,2)/variable_number;

        sum_xi__ = sum(xi__,2)/variable_number;
        exp_sum_xi__ = exp(sum_xi__);
        exp_xi__ = exp(xi__);
        sum_exp_xi__ = sum(exp_xi__,2)/variable_number;
        % get obj
        obj = (sum(sum_exp_xi__,1)+sum(exp_sum_xi__,1))/2/sample_number+...
            sum(exp_psi__+exp_zeta__,2)/variable_number*0.1;

        %         % get gradient
        %         gradient = sum(-a__*sign__.*exp_sum_xi__,1)/variable_number+...
        %             (-a_bou__*exp_psi__+a_bou__*exp_zeta__)/variable_number;

        %         % get gradient
        %         gradient = sum(-a__*sign__.*exp_xi__,1)/variable_number/sample_number+...
        %             (-a_bou__*exp_psi__+a_bou__*exp_zeta__)/variable_number;

        % get gradient
        gradient = (sum(-a__*sign__.*exp_sum_xi__,1)+sum(-a__*sign__.*exp_xi__,1))/2/variable_number/sample_number+...
            (-a_bou__*exp_psi__+a_bou__*exp_zeta__)/variable_number*0.1;

    end

    function distance_min__ = calMinDistance(x_list__)
        % get distance min from x_list
        %
        if isempty(x_list__)
            distance_min__ = [];
            return;
        end

        % sort x_supply_list_initial to decrese distance calculate times
        x_list__ = sortrows(x_list__,1);
        sample_number__ = size(x_list__,1);
        variable_number__ = size(x_list__,2);
        distance_min__ = variable_number__;
        for x_idx__ = 1:sample_number__
            x_curr__ = x_list__(x_idx__,:);
            x_next_idx__ = x_idx__ + 1;
            % first dimension only search in min_distance
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
        vio = calViolation(con,coneq,data_lib.nonlcon_torlance);
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
