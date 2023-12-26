clc;
clear;
close all hidden;

benchmark = Benchmark();

benchmark_type = 'single';
% benchmark_name = 'GP';
% benchmark_name = 'Wei';
% benchmark_name = 'PK';
% benchmark_name = 'EP20';
% benchmark_name = 'Forrester';
% benchmark_name = 'PVD4';
% benchmark_name = 'G01';
% benchmark_name = 'G06';
% benchmark_name = 'G07';
% benchmark_name = 'G18';

% benchmark_name_list = {'PVD4'};
% benchmark_name_list = {'G09'};
benchmark_name_list = {'G01'};

%% test case 
% benchmark_name_list = {'SR7','PVD4','G01','G5MOD','G09','G16','G18'};
% benchmark_name_list = {'G07','G09','G19','G23'};
cheapcon_function = [];

%% single run

% [model,variable_number,low_bou,up_bou,...
%     object_function,A,B,Aeq,Beq,nonlcon_function] = benchmark.getBenchmark(benchmark_type,benchmark_name);
% [x_best,obj_best,NFE,output] = optimalRBFGPCRS...
%     (model,variable_number,low_bou,up_bou,...
%     cheapcon_function,300,500)
% result_x_best = output.result_x_best;
% result_obj_best = output.result_obj_best;
%
% figure(1);
% plot(result_obj_best);

%% repeat run
mkdir('torl0')
for benchmark_idx=1:length(benchmark_name_list)
    benchmark_name=benchmark_name_list{benchmark_idx};

    [model,variable_number,low_bou,up_bou,...
        object_function,A,B,Aeq,Beq,nonlcon_function] = benchmark.getBenchmark(benchmark_type,benchmark_name);
    repeat_number = 25;
    result_obj = zeros(repeat_number,1);
    result_NFE = zeros(repeat_number,1);
    max_NFE = 200;
    for repeat_idx = 1:repeat_number
        [x_best,obj_best,NFE,output] = optimalRBFGPCRS...
            (model,variable_number,low_bou,up_bou,...
            cheapcon_function,max_NFE,300,1e-6,0);

        result_obj(repeat_idx) = obj_best;
        result_NFE(repeat_idx) = NFE;

        data_lib = output.data_lib;
%         plot(data_lib.Obj(data_lib.result_best_idx),'o-')
%         line(1:200,data_lib.Obj,'Marker','o','Color','g')
%         line(1:200,data_lib.Vio,'Marker','o','Color','r')
    end

    fprintf('Obj     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_obj),mean(result_obj),max(result_obj),std(result_obj));
    fprintf('NFE     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_NFE),mean(result_NFE),max(result_NFE),std(result_NFE));
    save(['torl0/',benchmark_name,'_RBF_GPC_RS_',num2str(max_NFE),'.mat']);
end

% mkdir('torl3')
% for benchmark_idx=1:length(benchmark_name_list)
%     benchmark_name=benchmark_name_list{benchmark_idx};
% 
%     [model,variable_number,low_bou,up_bou,...
%         object_function,A,B,Aeq,Beq,nonlcon_function] = benchmark.getBenchmark(benchmark_type,benchmark_name);
%     repeat_number = 10;
%     result_obj = zeros(repeat_number,1);
%     result_NFE = zeros(repeat_number,1);
%     max_NFE = 200;
%     for repeat_idx = 1:repeat_number
%         [x_best,obj_best,NFE,output] = optimalRBFGPCRS...
%             (model,variable_number,low_bou,up_bou,...
%             cheapcon_function,max_NFE,300,1e-6,1e-3);
% 
%         result_obj(repeat_idx) = obj_best;
%         result_NFE(repeat_idx) = NFE;
% 
%         data_lib = output.data_lib;
% %         plot(data_lib.Obj(data_lib.result_best_idx),'o-')
% %         line(1:200,data_lib.Obj,'Marker','o','Color','g')
% %         line(1:200,data_lib.Vio,'Marker','o','Color','r')
%     end
% 
%     fprintf('Obj     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_obj),mean(result_obj),max(result_obj),std(result_obj));
%     fprintf('NFE     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_NFE),mean(result_NFE),max(result_NFE),std(result_NFE));
%     save(['torl3/',benchmark_name,'_RBF_GPC_RS_',num2str(max_NFE),'.mat']);
% end

%% main
function [x_best,obj_best,NFE,output] = optimalRBFGPCRS...
    (model,vari_num,low_bou,up_bou,...
    cheapcon_fcn,....
    NFE_max,iter_max,torl,nonlcon_torl)
% RBF-GPC-RS optimization algorithm
%
% Copyright 2023 4 Adel
%
if nargin < 9 || isempty(nonlcon_torl)
    nonlcon_torl = 1e-3;
    if nargin < 8 || isempty(torl)
        torl = 1e-6;
        if nargin < 7
            iter_max = [];
            if nargin < 6
                NFE_max = [];
            end
        end
    end
end

if nargin < 5
    cheapcon_fcn = [];
end

DRAW_FIGURE_FLAG = 0; % whether draw data
INFORMATION_FLAG = 1; % whether print data
CONVERGENCE_JUDGMENT_FLAG = 0; % whether judgment convergence
WRIRE_FILE_FLAG = 0; % whether write to file

% hyper parameter
sample_num_initial = 6+3*vari_num;
sample_num_restart = sample_num_initial;
sample_num_add = ceil(log(sample_num_initial)/2);
% sample_num_add = 0;
min_bou_interest=1e-3;
max_bou_interest=1e-1;
trial_num = min(100*vari_num,100);

nomlz_value = 10; % max obj when normalize obj,con,coneq
protect_range = 1e-16; % surrogate add point protect range
identiy_torl = 1e-3; % if inf norm of point less than identiy_torlance, point will be consider as same local best

% NFE and iteration setting
if isempty(NFE_max)
    NFE_max = 10+10*vari_num;
end

if isempty(iter_max)
    iter_max = 20+20*vari_num;
end

done = 0;NFE = 0;iteration = 0;

% step 1
% generate initial data lib
X_updata = lhsdesign(sample_num_initial,vari_num,'iterations',50,'criterion','maximin').*(up_bou-low_bou)+low_bou;

data_lib = struct('model',model,'vari_num',vari_num,'low_bou',low_bou,'up_bou',up_bou,...
    'nonlcon_torl',nonlcon_torl,'str_data_file','result_total.txt','write_file_flag',WRIRE_FILE_FLAG,...
    'X',[],'Obj',[],'Con',[],'Coneq',[],'Vio',[],'Ks',[],'value_format','%.8e ','result_best_idx',[]);

% detech expensive constraints and initialize data lib
[data_lib,~,~,~,~,~,~,~,NFE_updata]=dataUpdata(data_lib,X_updata(1,:),0);
NFE = NFE+NFE_updata;
if ~isempty(data_lib.Vio)
    expensive_nonlcon_flag = 1;
else
    expensive_nonlcon_flag = 0;
end

% updata data lib by x_list
[data_lib,~,~,~,~,~,~,~,NFE_updata]=dataUpdata(data_lib,X_updata(2:end,:),0);
NFE = NFE+NFE_updata;

% find fesiable data in current data lib
if expensive_nonlcon_flag
    Bool_feas = data_lib.Vio == 0;
end
Bool_conv = false(sample_num_initial,1);

hyp.mean = 0;
hyp.cov = [0,0];

X_local_best=[];
Obj_local_best=[];
X_potential=[];
Obj_potential=[];
Vio_potential=[];
detect_local_flag=true(1);

fmincon_options = optimoptions('fmincon','Display','none','Algorithm','sqp','ConstraintTolerance',0,'FiniteDifferenceStepSize',1e-5);

result_x_best = zeros(iter_max,vari_num);
result_obj_best = zeros(iter_max,1);

iteration = iteration+1;

while ~done
    % step 2
    % nomalization all data by max obj and to create surrogate model
    [X,Obj,Con,Coneq,Vio,Ks]=dataLoad(data_lib);

    [X_model,Obj_model,Con_model,Coneq_model,Vio_model,Ks_model,obj_max,con_max_list,coneq_max_list,vio_max_list,ks_max_list]=getModelData(X,Obj,Con,Coneq,Vio,Ks,nomlz_value);

    % get local infill point, construct surrogate model
    [object_function_surrogate,nonlcon_function_surrogate,output_model] = getSurrogateFunction...
        (X_model,Obj_model,Con_model,Coneq_model);
    RBF_obj=output_model.RBF_obj;
    RBF_con_list=output_model.RBF_con_list;
    RBF_coneq_list=output_model.RBF_coneq_list;

    if ~isempty(nonlcon_function_surrogate) || ~isempty(cheapcon_fcn)
        constraint_function = @(x) totalconFunction...
            (x,nonlcon_function_surrogate,cheapcon_fcn,[]);
    else
        constraint_function = [];
    end

    % step 3
    if detect_local_flag
        % detech potential local best point
        for x_idx=1:size(X_model,1)
            x_initial=X_model(x_idx,:);
            [x_potential,obj_potential_pred,exit_flag,output_fmincon] = fmincon(object_function_surrogate,x_initial,[],[],[],[],...
                low_bou,up_bou,constraint_function,fmincon_options);

            if exit_flag == 1 || exit_flag == 2
                % check if x_potential have existed
                add_flag=true(1);
                for x_check_idx=1:size(X_potential,1)
                    if sum(abs(X_potential(x_check_idx,:)-x_potential),2)/vari_num < identiy_torl
                        add_flag=false(1);
                        break;
                    end
                end
                for x_check_idx=1:size(X_local_best,1)
                    if sum(abs(X_local_best(x_check_idx,:)-x_potential),2)/vari_num < identiy_torl
                        add_flag=false(1);
                        break;
                    end
                end

                % updata into X_potential
                if add_flag
                    X_potential=[X_potential;x_potential];
                    Obj_potential=[Obj_potential;obj_potential_pred/nomlz_value.*obj_max];
                    [con_potential,coneq_potential]=nonlcon_function_surrogate(x_potential);
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
            [ks_function_surrogate,radbas_model_ks] = interpRadialBasisPreModel(X_model,Ks_model);

            [~,x_idx]=min(Vio_model);
            x_initial=X_model(x_idx,:);
            [x_potential,obj_potential_pred,exit_flag,output_fmincon] = fmincon(ks_function_surrogate,x_initial,[],[],[],[],...
                low_bou,up_bou,cheapcon_fcn,fmincon_options);
            obj_potential_pred=object_function_surrogate(x_potential);

            X_potential=[X_potential;x_potential];
            Obj_potential=[Obj_potential;obj_potential_pred/nomlz_value*obj_max];
            [con_potential,coneq_potential]=nonlcon_function_surrogate(x_potential);
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

            [x_potential,obj_potential_pred,exit_flag,output_fmincon] = fmincon(object_function_surrogate,x_potential,[],[],[],[],...
                low_bou,up_bou,constraint_function,fmincon_options);

            X_potential(x_idx,:)=x_potential;
            Obj_potential(x_idx,:)=obj_potential_pred/nomlz_value*obj_max;
            [con_potential,coneq_potential]=nonlcon_function_surrogate(x_potential);
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
                if sum(abs(X_potential(x_check_idx,:)-x_potential),2)/vari_num < identiy_torl
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
    [data_lib,x_infill,obj_infill,con_infill,coneq_infill,vio_infill,ks_infill,repeat_idx,NFE_updata] = ...
        dataUpdata(data_lib,x_infill,protect_range);
    NFE = NFE+NFE_updata;

    if isempty(x_infill)
        % process error
        x_infill = data_lib.X(repeat_idx,:);
        obj_infill = data_lib.Obj(repeat_idx,:);
        if ~isempty(Con)
            con_infill = data_lib.Con(repeat_idx,:);
        end
        if ~isempty(Coneq)
            coneq_infill = data_lib.Coneq(repeat_idx,:);
        end
        if ~isempty(Vio)
            vio_infill = data_lib.Vio(repeat_idx,:);
        end
    else
        if ~isempty(vio_infill) && vio_infill > 0
            Bool_feas=[Bool_feas;false(1)];
        else
            Bool_feas=[Bool_feas;true(1)];
        end
        Bool_conv=[Bool_conv;false(1)];
    end
    Obj_potential(1,:)=obj_infill;
    Vio_potential(1,:)=vio_infill;

    if DRAW_FIGURE_FLAG && vari_num < 3
        interpVisualize(RBF_obj,low_bou,up_bou);
        line(x_infill(1),x_infill(2),obj_infill./obj_max*nomlz_value,'Marker','o','color','r','LineStyle','none');
    end

    % find best result to record
    [X,Obj,~,~,Vio,~]=dataLoad(data_lib);
    X_unconv = X(~Bool_conv,:);
    Obj_unconv=Obj(~Bool_conv,:);
    if ~isempty(Vio)
        Vio_unconv=Vio(~Bool_conv,:);
    else
        Vio_unconv=[];
    end
    idx = find(Vio_unconv == 0);
    if isempty(idx)
        [vio_best,min_idx] = min(Vio_unconv);
        obj_best = Obj_unconv(min_idx);
        x_best = X_unconv(min_idx,:);
    else
        [obj_best,min_idx] = min(Obj_unconv(idx));
        vio_best = 0;
        x_best = X_unconv(idx(min_idx),:);
    end

    if INFORMATION_FLAG
        fprintf('obj:    %f    violation:    %f    NFE:    %-3d\n',obj_best,vio_best,NFE);
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
    if iteration > iter_max || NFE >= NFE_max
        done = 1;
    end

    % convergence judgment
    if CONVERGENCE_JUDGMENT_FLAG
        if ( ((iteration > 2) && (abs((obj_infill-obj_infill_old)/obj_infill_old) < torl)) && ...
                ((~isempty(vio_infill) && vio_infill == 0) || isempty(vio_infill)) )
            done = 1;
        end
    end

    if ~done
        [X,Obj,~,~,Vio,~]=dataLoad(data_lib);
        % check if converage
        if  ( ((iteration > 2) && (abs((obj_infill-obj_infill_old)/obj_infill_old) < torl)))
            % && ((~isempty(vio_infill) && vio_infill == 0) || isempty(vio_infill)) 
            % resample LHD

            % step 7.1
            X_local_best=[X_local_best;x_infill];
            Obj_local_best=[Obj_local_best;obj_infill];
            X_potential=X_potential(2:end,:);
            Obj_potential=Obj_potential(2:end,:);
            Vio_potential=Vio_potential(2:end,:);

            if isempty(X_potential)
                detect_local_flag=true(1);
            end

            % step 7.2
            % detech converage
            for x_idx = 1:size(X,1)
                if ~Bool_conv(x_idx)
                    x_single_pred = fmincon(object_function_surrogate,X(x_idx,:),[],[],[],[],low_bou,up_bou,nonlcon_function_surrogate,fmincon_options);

                    converage_flag=false(1);

                    for x_check_idx=1:size(X_local_best,1)
                        if sum(abs(X_local_best(x_check_idx,:)-x_single_pred),2)/vari_num < identiy_torl
                            converage_flag=true(1);
                            break;
                        end
                    end

                    if converage_flag
                        % if converage to local minimum, set to infeasible
                        Bool_conv(x_idx) = true(1);
                    end
                end
            end

            % step 7.3
            % use GPC to limit do not converage to exist local best
            if ~all(Bool_conv)
                Class = -1*ones(size(X,1),1);
                Class(Bool_conv) = 1; % cannot go into converage area

                [GPC_predict_function,GPC_model] = classifyGaussProcess(X,Class,hyp);
                con_GPC_fcn = @(x) conGPCFunction(x,GPC_predict_function);
            else
                con_GPC_fcn=[];
            end

            % step 7.4
            % resample latin hypercubic and updata into data lib
            try
                X_add = getLatinHypercube(min(floor(sample_num_restart),NFE_max-NFE),vari_num,...
                    low_bou,up_bou,X,con_GPC_fcn);
            catch
                X_add=lhsdesign(min(floor(sample_num_restart),NFE_max-NFE),vari_num).*(up_bou-low_bou)+low_bou;
            end

            [data_lib,X_add,~,~,~,Vio_add,~,~,NFE_updata] = ...
                dataUpdata(data_lib,X_add,protect_range);
            NFE = NFE+NFE_updata;
            Bool_feas = [Bool_feas;Vio_add==0];
            Bool_conv = [Bool_conv;false(size(X_add,1),1)];

            if DRAW_FIGURE_FLAG && vari_num < 3
                classifyVisualization(GPC_model,low_bou,up_bou);
                line(X(base_boolean,1),X(base_boolean,2),'Marker','o','color','k','LineStyle','none');
            end
        else
            % step 5.1
            % check if improve
            improve = 0;
            if isempty(repeat_idx)
                Bool_comp=(~Bool_conv)&Bool_feas;
                Bool_comp(end)=false(1);
                if expensive_nonlcon_flag
                    min_vio = min(Vio(~Bool_conv(1:end-1)));
                    min_obj = min(Obj(Bool_comp));

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
                    min_obj = min(Obj(~Bool_conv));

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
                % construct GPC
                train_num=min(size(data_lib.X,1),11*vari_num-1+25);
                [pred_func_GPC,model_GPC,x_pareto_center,hyp]=trainFilter(data_lib,x_infill,hyp,train_num,expensive_nonlcon_flag,Bool_conv);

                % step 5.3
                % identify interest area
                con_GPC_fcn=@(x) conGPCFunction(x,pred_func_GPC);
                center_point=fmincon(con_GPC_fcn,x_pareto_center,[],[],[],[],low_bou,up_bou,cheapcon_fcn,fmincon_options);

                bou_interest=abs(center_point-x_infill);
                bou_interest=max(min_bou_interest.*(up_bou-low_bou),bou_interest);
                bou_interest=min(max_bou_interest.*(up_bou-low_bou),bou_interest);
                low_bou_interest=x_infill-bou_interest;
                up_bou_interest=x_infill+bou_interest;
                low_bou_interest=max(low_bou_interest,low_bou);
                up_bou_interest=min(up_bou_interest,up_bou);

                % generate trial point
                trial_point=repmat(x_infill,trial_num,1);
                for variable_idx = 1:vari_num
                    trial_point(:,variable_idx)=trial_point(:,variable_idx)+...
                        normrnd(0,bou_interest(variable_idx),[trial_num,1]);
                end
                trial_point=max(trial_point,low_bou);
                trial_point=min(trial_point,up_bou);

                Bool_negetive=pred_func_GPC(trial_point) == -1;
                if sum(Bool_negetive) < sample_num_add
                    value=con_GPC_fcn(trial_point);
                    thres=quantile(value,0.25);
                    Bool_negetive=value<thres;
                end
                trial_point=trial_point(Bool_negetive,:);

                % step 5.4
                % select point
                if size(trial_point,1) <= min(sample_num_add,NFE_max-NFE)
                    X_add=trial_point;
                else
                    max_dist=0;
                    iteration_select=1;
                    while iteration_select < 100
                        select_idx=randperm(size(trial_point,1),min(sample_num_add,NFE_max-NFE));
                        dist=calMinDistanceIter(trial_point(select_idx,:),X);
                        if max_dist < dist
                            X_add=trial_point(select_idx,:);
                            max_dist=dist;
                        end

                        iteration_select=iteration_select+1;
                    end
                end

                if DRAW_FIGURE_FLAG && vari_num < 3
                    classifyVisualization(model_GPC,low_bou,up_bou);
                    line(trial_point(:,1),trial_point(:,2),'Marker','o','color','k','LineStyle','none');
                    line(X_add(:,1),X_add(:,2),'Marker','o','color','g','LineStyle','none');
                end
            else
                X_add=[];
            end
        end

        % step 7
        % updata data lib
        [data_lib,X_add,~,~,~,Vio_add,~,~,NFE_updata] = ...
            dataUpdata(data_lib,X_add,protect_range);
        NFE = NFE+NFE_updata;
        Bool_feas = [Bool_feas;Vio_add == 0];
        Bool_conv = [Bool_conv;false(size(X_add,1),1)];

        % forced interrupt
        if iteration > iter_max || NFE >= NFE_max
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
x_best = data_lib.X(data_lib.result_best_idx(end),:);
obj_best = data_lib.Obj(data_lib.result_best_idx(end),:);

result_x_best = result_x_best(1:iteration-1,:);
result_obj_best = result_obj_best(1:iteration-1);

output.result_x_best = result_x_best;
output.result_obj_best = result_obj_best;

output.x_local_best = X_local_best;
output.obj_local_best = Obj_local_best;

output.data_lib = data_lib;

    function [con,coneq] = conGPCFunction(x,GPC_predict_function)
        % function to obtain probability predict function
        %
        [~,~,con] = GPC_predict_function(x);
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
function [X_model,Obj_model,Con_model,Coneq_model,Vio_model,Ks_model,obj_max,con_max_list,coneq_max_list,vio_max_list,ks_max_list]=getModelData(X,Obj,Con,Coneq,Vio,Ks,nomlz_obj)

X_model = X;
obj_max = max(abs(Obj),[],1);
Obj_model=Obj/obj_max*nomlz_obj;
if ~isempty(Con)
    con_max_list = max(abs(Con),[],1);
    Con_model = Con./con_max_list*nomlz_obj;
else
    con_max_list=[];
    Con_model = [];
end
if ~isempty(Coneq)
    coneq_max_list = max(abs(Coneq),[],1);
    Coneq_model = Coneq./coneq_max_list*nomlz_obj;
else
coneq_max_list=[];
    Coneq_model = [];
end
if ~isempty(Vio)
    vio_max_list = max(abs(Vio),[],1);
    Vio_model = Vio./vio_max_list*nomlz_obj;
else
    vio_max_list=[];
    Vio_model = [];
end
if ~isempty(Ks)
    ks_max_list = max(abs(Ks),[],1);
    Ks_model = Ks./ks_max_list*nomlz_obj;
else
    ks_max_list=[];
    Ks_model = [];
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
    (x_list,obj_list,con_list,coneq_list)
% base on lib_data to create radialbasis model and function
% if input model,function will updata model
% object_function is single obj output
% nonlcon_function is normal nonlcon_function which include con,coneq
% con is colume vector,coneq is colume vector
% var_function is same
%
basis_function = @(r) r.^3;

[predict_function_obj,RBF_obj] = interpRadialBasisPreModel...
    (x_list,obj_list,basis_function);

if ~isempty(con_list)
    predict_function_con = cell(size(con_list,2),1);
    RBF_con_list = cell(size(con_list,2),1);
    for con_idx = 1:size(con_list,2)
        [predict_function_con{con_idx},RBF_con_list{con_idx}] = interpRadialBasisPreModel...
            (x_list,con_list(:,con_idx),basis_function);
    end
else
    predict_function_con = [];
    RBF_con_list = [];
end

if ~isempty(coneq_list)
    predict_function_coneq = cell(size(coneq_list,2),1);
    RBF_coneq_list = cell(size(coneq_list,2),1);
    for coneq_idx = 1:size(coneq_list,2)
        [predict_function_coneq{coneq_idx},RBF_coneq_list{coneq_idx}] = interpRadialBasisPreModel...
            (x_list,coneq_list(:,coneq_idx),basis_function);
    end
else
    predict_function_coneq = [];
    RBF_coneq_list = [];
end

object_function_surrogate = @(X_predict) objectFunctionSurrogate(X_predict,predict_function_obj);
if isempty(predict_function_con) && isempty(predict_function_coneq)
    nonlcon_function_surrogate = [];
else
    nonlcon_function_surrogate = @(X_predict) nonlconFunctionSurrogate(X_predict,predict_function_con,predict_function_coneq);
end

output.RBF_obj=RBF_obj;
output.RBF_con_list=RBF_con_list;
output.RBF_coneq_list=RBF_coneq_list;

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
idx__ = find(stdD_X == 0);
if ~isempty(idx__),stdD_X(idx__) = 1;end
idx__ = find(stdD_Y == 0);
if ~isempty(idx__),stdD_Y(idx__) = 1;end
X_nomlz = (X-aver_X)./stdD_X;
Y_nomlz = (Y-aver_Y)./stdD_Y;

if isempty(basis_function)
    %     c = (prod(max(X_nomlz)-min(X_nomlz))/x_number)^(1/variable_number);
    %     basis_function = @(r) exp(-(r.^2)/c);
    basis_function = @(r) r.^2;
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

radialbasis_model.X = X;
radialbasis_model.Y = Y;
radialbasis_model.radialbasis_matrix = rdibas_matrix;
radialbasis_model.inv_radialbasis_matrix=inv_rdibas_matrix;
radialbasis_model.beta = beta;

radialbasis_model.aver_X = aver_X;
radialbasis_model.stdD_X = stdD_X;
radialbasis_model.aver_Y = aver_Y;
radialbasis_model.stdD_Y = stdD_Y;
radialbasis_model.basis_function = basis_function;

radialbasis_model.predict_function = predict_function;

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
function [X_sample,dist_min_nomlz,X_total] = getLatinHypercube...
    (sample_number,variable_number,...
    low_bou,up_bou,X_exist,cheapcon_function)
% generate latin hypercube desgin
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

%% data library function
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
    file_data = fopen(data_lib.str_data_file,'a');
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
        if distance_min < data_lib.vari_num*protect_range
            % distance to exist point of point to add is small than protect_range
            repeat_idx = [repeat_idx;min_idx];
            continue;
        end
    end

    [obj,con,coneq] = data_lib.model(x); % eval value
    NFE = NFE+1;

    con = con(:)';
    coneq = coneq(:)';
    % calculate vio
    if isempty(con) && isempty(coneq)
        vio = [];
        ks = [];
    else
        vio = calViolation(con,coneq,data_lib.nonlcon_torl);
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
        fprintf(file_data,'%d ',repmat('%.8e ',1,data_lib.vari_num));
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
            if obj <= data_lib.Obj(data_lib.result_best_idx(end))
                data_lib.result_best_idx = [data_lib.result_best_idx;size(data_lib.X,1)];
            else
                data_lib.result_best_idx = [data_lib.result_best_idx;data_lib.result_best_idx(end)];
            end
        else
            if vio <= data_lib.Vio(data_lib.result_best_idx(end))
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
