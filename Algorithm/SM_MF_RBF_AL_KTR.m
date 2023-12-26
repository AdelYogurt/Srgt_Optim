clc;
clear;
close all hidden;

benchmark=BenchmarkFunction();

benchmark_type = 'single';
multi_fidelity = 2;
% benchmark_name = 'GP';
% benchmark_name = 'Wei';
% benchmark_name = 'PK';
% benchmark_name = 'EP20';
% benchmark_name = 'Forrester';
% benchmark_name = 'PVD4';
% benchmark_name = 'G01';
% benchmark_name = 'G06';
benchmark_name = 'G18';

[model_function,variable_number,low_bou,up_bou,...
    object_function,A,B,Aeq,Beq,nonlcon_function] = benchmark.getBenchmark(benchmark_type,benchmark_name,multi_fidelity);

cheapcon_function = [];

%% single run

% [x_best,fval_best,NFE,output] = optimalMFRBFALKTR...
%     (model_function,variable_number,low_bou,up_bou,...
%     cheapcon_function,300,500)
% result_x_best = output.result_x_best;
% result_fval_best = output.result_fval_best;
% 
% figure(1);
% plot(result_fval_best);

%% repeat run

repeat_number = 10;
result_fval = zeros(repeat_number,1);
result_NFE = zeros(repeat_number,1);
max_NFE = 200;
for repeat_index = 1:repeat_number
    [x_best,fval_best,NFE,output] = optimalMFRBFALKTR...
        (model_function,variable_number,low_bou,up_bou,...
        cheapcon_function,max_NFE,300,1e-6,1e-3);

    result_fval(repeat_index) = fval_best;
    result_NFE(repeat_index) = NFE;
end

fprintf('Fval     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_fval),mean(result_fval),max(result_fval),std(result_fval));
fprintf('NFE     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_NFE),mean(result_NFE),max(result_NFE),std(result_NFE));
save([benchmark_name,'_MFRBF_MFAL_KTR','.mat']);

%% main
function [x_best,fval_best,NFE,output] = optimalMFRBFALKTR...
    (model_function_data,variable_number,low_bou,up_bou,...
    cheapcon_function,....
    NFE_max,iteration_max,torlance,nonlcon_torlance)
% MFRBF-MFAL-KTR optimization algorithm
%
% input:
% model_function_data(type: struct, include high and low fidelity, cross_ratio)
%
% Copyright 2023 4 Adel
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

DRAW_FIGURE_FLAG = 1; % whether draw data
INFORMATION_FLAG = 1; % whether print data
CONVERGENCE_JUDGMENT_FLAG = 0; % whether judgment convergence
WRIRE_FILE_FLAG = 0; % whether write to file

% hyper parameter
sample_number_initial = 6+3*variable_number;
sample_number_restart = sample_number_initial;
% sample_number_add = ceil(log(6+4*variable_number));
sample_number_add = 1;
min_bou_interest=1e-3;

nomlz_fval = 10; % max fval when normalize fval,con,coneq
protect_range = 1e-5; % surrogate add point protect range
identiy_torlance = 1e-3; % if inf norm of point less than identiy_torlance, point will be consider as same local best

% NFE and iteration setting
if isempty(NFE_max)
    NFE_max = 10+10*variable_number;
end

if isempty(iteration_max)
    iteration_max = 20+20*variable_number;
end

done = 0;NFE = 0;iteration = 0;

% step 0
% local model function
HF_model=model_function_data.model_function_HF;
LF_model=model_function_data.model_function_LF;

% step 1
% generate initial data llibrary
%     [~,x_updata_list,~] = getLatinHypercube...
%         (sample_number_initial,variable_number,[],low_bou,up_bou,cheapcon_function);
X_updata = lhsdesign(sample_number_initial,variable_number,'iterations',50,'criterion','maximin').*(up_bou-low_bou)+low_bou;

data_library = DataLibrary(HF_model,variable_number,low_bou,up_bou,...
    nonlcon_torlance,[],WRIRE_FILE_FLAG);

% detech expensive constraints and initialize data library
[~,~,~,~,~,~,~,NFE_updata]=data_library.dataUpdata(X_updata(1,:),0);
NFE = NFE+NFE_updata;
if ~isempty(data_library.vio_list)
    expensive_nonlcon_flag = 1;
else
    expensive_nonlcon_flag = 0;
end

% updata data library by x_list
[~,~,~,~,~,~,~,NFE_updata]=data_library.dataUpdata(X_updata(2:end,:),0);
NFE = NFE+NFE_updata;

% find fesiable data in current data library
if expensive_nonlcon_flag
    Bool_feas = data_library.vio_list == 0;
else
    Bool_feas = true(sample_number_initial,1);
end
Bool_conv = false(sample_number_initial,1);

hyp.mean = 0;
hyp.cov = [0,0];

conv_con_GPC_function=[];
ks_hyp=0;

X_local_best=[];
Fval_local_best=[];
X_potential=[];
Fval_potential=[];
Vio_potential=[];
detect_local_flag=true(1);

fmincon_options = optimoptions('fmincon','Display','none','Algorithm','sqp','ConstraintTolerance',0,'FiniteDifferenceStepSize',1e-5);

result_x_best = zeros(iteration_max,variable_number);
result_fval_best = zeros(iteration_max,1);

iteration = iteration+1;

while ~done
    % step 2
    % nomalization all data by max fval and to create surrogate model
    [X,Fval,Con,Coneq,Vio,Ks]=data_library.dataLoad;

    X_model = X;
    fval_max = max(abs(Fval),[],1);
    Fval_model=Fval/fval_max*nomlz_fval;
    if ~isempty(Con)
        con_max_list = max(abs(Con),[],1);
        Con_model = Con./con_max_list*nomlz_fval;
    else
        Con_model = [];
    end
    if ~isempty(Coneq)
        coneq_max_list = max(abs(Coneq),[],1);
        Coneq_model = Coneq./coneq_max_list*nomlz_fval;
    else
        Coneq_model = [];
    end
    if ~isempty(Vio)
        vio_max_list = max(abs(Vio),[],1);
        Vio_model = Ks./vio_max_list*nomlz_fval;
    else
        Vio_model = [];
    end
    if ~isempty(Ks)
        ks_max_list = max(abs(Ks),[],1);
        Ks_model = Ks./ks_max_list*nomlz_fval;
    else
        Ks_model = [];
    end

    % get local infill point, construct surrogate model
    [object_function_surrogate,nonlcon_function_surrogate,output_model] = getSurrogateFunction...
        (X_model,Fval_model,Con_model,Coneq_model);
    radbas_model_fval=output_model.radbas_model_fval;
    radbas_model_con_list=output_model.radbas_model_con_list;
    radbas_model_coneq_list=output_model.radbas_model_coneq_list;
    %     [ks_function_surrogate,~,output_model] = getSurrogateFunction...
    %         (X_model,Ks_model,[],[]);
    %     radbas_model_ks=output_model.radbas_model_fval;
    [ks_function_surrogate,kriging_model_ks] = interpKrigingPreModel(X_model,Ks_model,ks_hyp);

    if ~isempty(nonlcon_function_surrogate) || ~isempty(cheapcon_function)
        constraint_function = @(x) totalconFunction...
            (x,nonlcon_function_surrogate,cheapcon_function,[]);
    else
        constraint_function = [];
    end

    % step 3
    if detect_local_flag
        % detech potential local best point
        for x_index=1:size(X_model,1)
            x_initial=X_model(x_index,:);
            [x_potential,fval_potential_pred,exit_flag,output] = fmincon(object_function_surrogate,x_initial,[],[],[],[],...
                low_bou,up_bou,constraint_function,fmincon_options);

            if exit_flag == 1 || exit_flag == 2
                % check if x_potential have existed
                add_flag=true(1);
                for x_check_index=1:size(X_potential,1)
                    if sum(abs(X_potential(x_check_index,:)-x_potential),2)/variable_number < identiy_torlance
                        add_flag=false(1);
                        break;
                    end
                end
                for x_check_index=1:size(X_local_best,1)
                    if sum(abs(X_local_best(x_check_index,:)-x_potential),2)/variable_number < identiy_torlance
                        add_flag=false(1);
                        break;
                    end
                end

                % updata into X_potential
                if add_flag
                    X_potential=[X_potential;x_potential];
                    Fval_potential=[Fval_potential;fval_potential_pred/nomlz_fval.*fval_max];
                    [con_potential,coneq_potential]=nonlcon_function_surrogate(x_potential);
                    if ~isempty(con_potential)
                        con_potential=con_potential/nomlz_fval.*con_max_list;
                    end
                    if ~isempty(coneq_potential)
                        coneq_potential=coneq_potential/nomlz_fval.*coneq_max_list;
                    end
                    Vio_potential=[Vio_potential;calViolation(con_potential,coneq_potential,nonlcon_torlance)];
                end
            end
        end

        if isempty(X_potential)
            [~,x_index]=min(Vio_model);
            x_initial=X_model(x_index,:);
            [x_potential,fval_potential_pred,exit_flag,output] = fmincon(ks_function_surrogate,x_initial,[],[],[],[],...
                low_bou,up_bou,cheapcon_function,fmincon_options);
            fval_potential_pred=object_function_surrogate(x_potential);

            X_potential=[X_potential;x_potential];
            Fval_potential=[Fval_potential;fval_potential_pred/nomlz_fval*fval_max];
            [con_potential,coneq_potential]=nonlcon_function_surrogate(x_potential);
            if ~isempty(con_potential)
                con_potential=con_potential/nomlz_fval.*con_max_list;
            end
            if ~isempty(coneq_potential)
                coneq_potential=coneq_potential/nomlz_fval.*coneq_max_list;
            end
            Vio_potential=[Vio_potential;calViolation(con_potential,coneq_potential,nonlcon_torlance)];
        end

        % sort X potential by Vio
        [Vio_potential,index]=sort(Vio_potential);
        Fval_potential=Fval_potential(index,:);
        X_potential=X_potential(index,:);

        % sort X potential by Fval
        flag=find(Vio_potential == 0, 1, 'last' );
        if isempty(flag) % mean do not have fesaible point
            [Fval_potential,index]=sort(Fval_potential);
            Vio_potential=Vio_potential(index,:);
            X_potential=X_potential(index,:);
        else
            [Fval_potential(1:flag,:),index_feas]=sort(Fval_potential(1:flag,:));
            [Fval_potential(flag+1:end,:),index_infeas]=sort(Fval_potential(flag+1:end,:));
            index=[index_feas;index_infeas+flag];

            Vio_potential=Vio_potential(index,:);
            X_potential=X_potential(index,:);
        end

        detect_local_flag=false(1);
    else
        % updata X potential
        for x_index=1:size(X_potential,1)
            x_potential=X_potential(x_index,:);

%             if Vio_potential(x_index,:) == 0
                [x_potential,fval_potential_pred,exit_flag,output] = fmincon(object_function_surrogate,x_potential,[],[],[],[],...
                    low_bou,up_bou,constraint_function,fmincon_options);
%             else
%                 [x_potential,~,exit_flag,output] = fmincon(ks_function_surrogate,x_potential,[],[],[],[],...
%                     low_bou,up_bou,cheapcon_function,fmincon_options);
%                 fval_potential_pred=object_function_surrogate(x_potential);
%             end

            X_potential(x_index,:)=x_potential;
            Fval_potential(x_index,:)=fval_potential_pred/nomlz_fval*fval_max;
            [con_potential,coneq_potential]=nonlcon_function_surrogate(x_potential);
            if ~isempty(con_potential)
                con_potential=con_potential/nomlz_fval.*con_max_list;
            end
            if ~isempty(coneq_potential)
                coneq_potential=coneq_potential/nomlz_fval.*coneq_max_list;
            end
            Vio_potential(x_index,:)=calViolation(con_potential,coneq_potential,nonlcon_torlance);
        end

        % merge X potential
        % Upward merge
        for x_index=size(X_potential,1):-1:1
            x_potential=X_potential(x_index,:);

            % check if x_potential have existed
            merge_flag=false(1);
            for x_check_index=1:x_index-1
                if sum(abs(X_potential(x_check_index,:)-x_potential),2)/variable_number < identiy_torlance
                    merge_flag=true(1);
                    break;
                end
            end

            % updata into X_potential
            if merge_flag
                X_potential(x_index,:)=[];
                Fval_potential(x_index,:)=[];
                Vio_potential(x_index,:)=[];
            end
        end
    end

    % select best potential point as x_infill
    x_infill=X_potential(1,:);
    fval_infill_pred=Fval_potential(1,:);

    % updata infill point
    [x_infill,fval_infill,con_infill,coneq_infill,vio_infill,ks_infill,repeat_index,NFE_updata] = ...
        data_library.dataUpdata(x_infill,protect_range);
    NFE = NFE+NFE_updata;

    if isempty(x_infill)
        % process error
        x_infill = data_library.x_list(repeat_index,:);
        fval_infill = data_library.fval_list(repeat_index,:);
        if ~isempty(Con)
            con_infill = data_library.con_list(repeat_index,:);
        end
        if ~isempty(Coneq)
            coneq_infill = data_library.coneq_list(repeat_index,:);
        end
        if ~isempty(Vio)
            vio_infill = data_library.vio_list(repeat_index,:);
        end
    else
        if ~isempty(vio_infill) && vio_infill > 0
            Bool_feas=[Bool_feas;false(1)];
        else
            Bool_feas=[Bool_feas;true(1)];
        end
        Bool_conv=[Bool_conv;false(1)];
    end
    Fval_potential(1,:)=fval_infill;
    Vio_potential(1,:)=vio_infill;

    if DRAW_FIGURE_FLAG && variable_number < 3
        interpVisualize(radbas_model_fval,low_bou,up_bou);
        line(x_infill(1),x_infill(2),fval_infill./fval_max*nomlz_fval,'Marker','o','color','r','LineStyle','none');
    end

    % find best result to record
    [X,Fval,Con,Coneq,Vio,Ks]=data_library.dataLoad;
    X_unconv=X(~Bool_conv,:);
    Fval_unconv=Fval(~Bool_conv,:);
    if ~isempty(Con)
        Con_unconv=Con(~Bool_conv,:);
    else
        Con=[];
    end
    if ~isempty(Coneq)
        Coneq_unconv=Coneq(~Bool_conv,:);
    else
        Coneq_unconv=[];
    end
    [x_best,fval_best,con_best,coneq_best] = findMinRaw...
        (X_unconv,Fval_unconv,Con_unconv,Coneq_unconv,...
        cheapcon_function,nonlcon_torlance);
    vio_best = calViolation(con_best,coneq_best,nonlcon_torlance);

    if INFORMATION_FLAG
        fprintf('fval:    %f    violation:    %f    NFE:    %-3d\n',fval_best,vio_best,NFE);
        %         fprintf('iteration:          %-3d    NFE:    %-3d\n',iteration,NFE);
        %         fprintf('x:          %s\n',num2str(x_infill));
        %         fprintf('value:      %f\n',fval_infill);
        %         fprintf('violation:  %s  %s\n',num2str(con_infill),num2str(coneq_infill));
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
        if ( ((iteration > 2) && (abs((fval_infill-fval_infill_old)/fval_infill_old) < torlance)) && ...
                ((~isempty(vio_infill) && vio_infill == 0) || isempty(vio_infill)) )
            done = 1;
        end
    end

    if ~done
        % check if converage
        if (isempty(conv_con_GPC_function) || ((~isempty(conv_con_GPC_function) && (conv_con_GPC_function(x_infill)) < -1e-6)))...
                && ( ((iteration > 2) && (abs((fval_infill-fval_infill_old)/fval_infill_old) < torlance)) && ...
                ((~isempty(vio_infill) && vio_infill == 0) || isempty(vio_infill)) )
            % step 6
            % resample LHD

            X_local_best=[X_local_best;x_infill];
            Fval_local_best=[Fval_local_best;fval_infill];
            X_potential=X_potential(2:end,:);
            Fval_potential=Fval_potential(2:end,:);
            Vio_potential=Vio_potential(2:end,:);

            if isempty(X_potential)
                detect_local_flag=true(1);
            end

            % step 6.1
            % detech converage
            for x_index = 1:size(X,1)
                if ~Bool_conv(x_index)
                    x_single_pred = fmincon(object_function_surrogate,X(x_index,:),[],[],[],[],low_bou,up_bou,nonlcon_function_surrogate,fmincon_options);

                    converage_flag=false(1);

                    for x_check_index=1:size(X_local_best,1)
                        if sum(abs(X_local_best(x_check_index,:)-x_single_pred),2)/variable_number < identiy_torlance
                            converage_flag=true(1);
                            break;
                        end
                    end

                    if converage_flag
                        % if converage to local minimum, set to infeasible
                        Bool_conv(x_index) = true(1);
                    end
                end
            end

            % step 6.2
            % use GPC to limit do not converage to exist local best
            if ~all(Bool_conv)
                class_list = -1*ones(size(X,1),1);
                class_list(Bool_conv) = 1; % cannot go into converage area

                [GPC_predict_function,GPC_model] = classifyGaussProcess(X,class_list,hyp);
                conv_con_GPC_function = @(x) conGPCFunction(x,GPC_predict_function);
            else
                conv_con_GPC_function=[];
            end
            
            % step 6.3
            % resample latin hypercubic and updata into data library
            x_updata = getLatinHypercube(min(floor(sample_number_restart),NFE_max-NFE-1),variable_number,...
                low_bou,up_bou,X,conv_con_GPC_function);

            [x_updata,fval_updata,con_updata,coneq_updata,vio_updata,ks_updata,repeat_index,NFE_updata] = ...
                data_library.dataUpdata(x_updata,protect_range);
            NFE = NFE+NFE_updata;
            Bool_feas = [Bool_feas;ks_updata==0];
            Bool_conv = [Bool_conv;false(size(x_updata,1),1)];

            conv_con_GPC_function=[];

            if DRAW_FIGURE_FLAG && variable_number < 3
                classifyVisualization(GPC_model,low_bou,up_bou);
                line(X(base_boolean,1),X(base_boolean,2),'Marker','o','color','k','LineStyle','none');
            end
        else
            % step 4
            % check if improve
            improve = 0;
            if isempty(repeat_index)
                Bool_comp=(~Bool_conv)&Bool_feas;
                Bool_comp(end)=false(1);
                if expensive_nonlcon_flag
                    min_vio = min(Vio(~Bool_conv(1:end-1)));
                    min_fval = min(Fval(Bool_comp));

                    % if all point is infeasible,violation of point infilled is
                    % less than min violation of all point means improve.if
                    % feasible point exist,fval of point infilled is less than min
                    % fval means improve
                    if vio_infill == 0 || vio_infill < min_vio
                        if ~isempty(min_fval)
                            if fval_infill < min_fval
                                % improve, continue local search
                                improve = 1;
                            end
                        else
                            % improve, continue local search
                            improve = 1;
                        end
                    end
                else
                    min_fval = min(Fval(Bool_comp));

                    % fval of point infilled is less than min fval means improve
                    if fval_infill < min_fval
                        % imporve, continue local search
                        improve = 1;
                    end
                end
            end

            if iteration > 2
                improve_value=((fval_infill_old-fval_infill)/(fval_infill_old-fval_infill_pred));
            end

            % step 5
            % if fval no improve, use GPC to identify interest area
            % than, imporve interest area surrogate quality
            if ~improve
                [X,Fval,Con,Coneq,Vio,Ks]=data_library.dataLoad;

                % step 5.1
                % construct GPC
                if expensive_nonlcon_flag
                    % base on filter to decide which x should be choose
                    pareto_index_list = getParetoFront([Fval(~Bool_feas),Ks(~Bool_feas)]);
                    index = 1:size(X,1);
                    pareto_index_list=index(pareto_index_list);

                    x_pareto_center=sum(X(pareto_index_list,:,:),1)/length(pareto_index_list);

                    class_list = ones(size(X,1),1);
                    class_list(pareto_index_list) = -1;
                    class_list(Bool_feas) = -1; % can go into feasiable area
                    class_list(Bool_conv) = 1; % cannot go into convarage area

                    [GPC_predict_function,GPC_model] = classifyGaussProcess(X,class_list,hyp);

                else
                    fval_threshold = prctile(Fval(~Bool_feas),50-40*sqrt(NFE/NFE_max));
                    class_list = ones(size(X,1),1);
                    class_list(Fval < fval_threshold) = -1;
                    class_list(Bool_feas) = -1; % can go into feasiable area
                    class_list(Bool_conv) = 1; % cannot go into convarage area

                    [GPC_predict_function,GPC_model] = classifyGaussProcess(X,class_list,hyp);
                end

                % step 5.2
                % identify interest area
                con_GPC_function=@(x) conGPCFunction(x,GPC_predict_function);
                center_point=fmincon(con_GPC_function,x_pareto_center,[],[],[],[],low_bou,up_bou,cheapcon_function,fmincon_options);

                bou_interest=abs(center_point-x_infill);
                bou_interest=max(min_bou_interest.*(up_bou-low_bou),bou_interest);
                low_bou_interest=x_infill-bou_interest;
                up_bou_interest=x_infill+bou_interest;
                low_bou_interest=max(low_bou_interest,low_bou);
                up_bou_interest=min(up_bou_interest,up_bou);

                % step 5.3
                [X_local,Fval_local,Con_local,Coneq_local,Vio_local,Ks_local]=data_library.dataLoad(low_bou_interest,up_bou_interest);
                low_bou_local=low_bou_interest;
                up_bou_local=up_bou_interest;

                [X_local,index]=proFilterX(X_local,1e-3);
                Fval_local=Fval_local(index,:);
                if ~isempty(Con_local)
                    Con_local=Con_local(index,:);
                end
                if ~isempty(Coneq_local)
                    Coneq_local=Coneq_local(index,:);
                end

                if size(X_local,1) < 10
                    [X_local,Fval_local,Con_local,Coneq_local,Vio_local,Ks_local]=data_library.dataLoad();
                    low_bou_local=low_bou;
                    up_bou_local=up_bou;

                    [X_local,index]=proFilterX(X_local,1e-3);
                    Fval_local=Fval_local(index,:);
                    if ~isempty(Con_local)
                        Con_local=Con_local(index,:);
                    end
                    if ~isempty(Coneq_local)
                        Coneq_local=Coneq_local(index,:);
                    end

                end

                interp_number=min(50,size(X_local,1)-1);
                X_local=X_local(end-interp_number:end,:);
                Fval_local=Fval_local(end-interp_number:end,:);
                if ~isempty(Con_local)
                    Con_local=Con_local(end-interp_number:end,:);
                end
                if ~isempty(Coneq_local)
                    Coneq_local=Coneq_local(end-interp_number:end,:);
                end

                x_updata=optiSurrogateQuality...
                    (variable_number,low_bou_interest,up_bou_interest,sample_number_add,...
                    low_bou_local,up_bou_local,X_local,Fval_local,object_function_surrogate,Con_local,Coneq_local);

                [x_updata,fval_updata,con_updata,coneq_updata,vio_updata,ks_updata,repeat_index,NFE_updata] = ...
                    data_library.dataUpdata(x_updata,protect_range);
                NFE = NFE+NFE_updata;
                Bool_feas = [Bool_feas;vio_updata == 0];
                Bool_conv = [Bool_conv;false(size(x_updata,1),1)];
            end
        end
    end

    fval_best_old = fval_best;

    fval_infill_old = fval_infill;
    con_infill_old = con_infill;
    coneq_infill_old = coneq_infill;
    vio_infill_old = vio_infill;
end

% find best result to record
[X,Fval,Con,Coneq]=data_library.dataLoad();
[x_best,fval_best,con_best,coneq_best] = findMinRaw...
    (X,Fval,Con,Coneq,...
    cheapcon_function,nonlcon_torlance);

result_x_best = result_x_best(1:iteration-1,:);
result_fval_best = result_fval_best(1:iteration-1);

output.result_x_best = result_x_best;
output.result_fval_best = result_fval_best;
output.data_library = data_library;
output.x_local_best = X_local_best;
output.fval_local_best = Fval_local_best;

%     function fval = meritFunction(x,object_function_surrogate,x_list,variable_number,up_bou,low_bou,wF,wD)
%         % function to consider surrogate fval and variance
%         %
%         fval_pred = object_function_surrogate(x);
%
%         fval_dist = -atan(min(sum(((x-x_list)./(up_bou-low_bou)/variable_number).^2,2))*200);
%
%         fval = fval_pred*wF+fval_dist*wD;
%     end

    function fval = meritFunction...
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
        for vari_index = 1:variable_number
            dis = dis+(x_trial_list(:,vari_index)-x_list(:,vari_index)').^2;
        end
        D = min(sqrt(dis),[],2);
        D_min = min(D); D_max = max(D);
        D = (D_max-D)./(D_max-D_min);

        fval = w_F*F+w_D*D;
    end

    function [con,coneq] = conGPCFunction(x,GPC_predict_function)
        % function to obtain probability predict function
        %
        [~,~,con] = GPC_predict_function(x);
        coneq = [];
    end

    function fval = distanceFunction(x,x_list,variable_number,up_bou,low_bou)
        fval = -(min(sum(abs((x-x_list)./(up_bou-low_bou)/variable_number),2)));
    end

end

%% auxiliary function
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
if ~isempty(cheapcon_function)
    for x_index = 1:size(x_list,1)
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

function [x_list,fval_list,con_list,coneq_list,vio_list,index_list] = rankData...
    (x_list,fval_list,con_list,coneq_list,...
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
% infeasible data rank by violation,feasible data rank by fval
feasi_boolean_list = vio_list <= 0;
all = 1:x_number;
feasi_index_list = all(feasi_boolean_list);
infeasi_index_list = all(~feasi_boolean_list);
[~,index_list] = sort(fval_list(feasi_index_list));
feasi_index_list = feasi_index_list(index_list);
[~,index_list] = sort(vio_list(infeasi_index_list));
infeasi_index_list = infeasi_index_list(index_list);
index_list = [feasi_index_list,infeasi_index_list];

% rank by index_list
x_list = x_list(index_list,:);
fval_list = fval_list(index_list);
if ~isempty(con_list)
    con_list = con_list(index_list,:);
end
if ~isempty(coneq_list)
    coneq_list = coneq_list(index_list,:);
end
vio_list = vio_list(index_list);

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

function X_add=optiSurrogateQuality...
    (variable_number,low_bou_add,up_bou_add,add_number,...
    low_bou,up_bou,X,Fval,object_function_surrogate,Con,Coneq)
% find point local to imporve surrogate quality
%
X_add_initial=lhsdesign(add_number,variable_number);
basis_function = @(r) r.^3;

fmincon_options = optimoptions('fmincon','Display','none','Algorithm','sqp','ConstraintTolerance',0,'FiniteDifferenceStepSize',1e-5);

X_nomlz=(X-low_bou)./(up_bou-low_bou);

% initialization distance of all X
x_number=size(X_nomlz,1);
X_dis = zeros(x_number+add_number,x_number+add_number);
for variable_index = 1:variable_number
    X_dis(1:x_number,1:x_number) = X_dis(1:x_number,1:x_number)+(X_nomlz(:,variable_index)-X_nomlz(:,variable_index)').^2;
end
X_dis = sqrt(X_dis);

object_function_quality=@(x) objectFunctionQuality...
    (x,x_number,variable_number,low_bou,up_bou,add_number,X_dis,X_nomlz,Fval,object_function_surrogate,Con,Coneq,basis_function);
X_add_initial=reshape(X_add_initial',add_number*variable_number,1)';
low_bou_add=repmat((low_bou_add-low_bou)./(up_bou-low_bou),1,add_number);
up_bou_add=repmat((up_bou_add-low_bou)./(up_bou-low_bou),1,add_number);

object_function_quality(X_add_initial);

[X_add,fval,exit_flag,output]=fmincon(object_function_quality,X_add_initial,[],[],[],[],low_bou_add,up_bou_add,[],fmincon_options);
X_add=reshape(X_add',variable_number,add_number)'.*(up_bou-low_bou)+low_bou;

    function fval_quality=objectFunctionQuality...
            (X_add,x_num,vari_num,low_bou,up_bou,add_num,X_dis,X_nomlz,Fval,object_function_surrogate,Con,Coneq,basis_function)
        X_add=reshape(X_add',vari_num,add_num)';

        for vari_index = 1:vari_num
            X_dis(1:x_num,x_num+1:x_num+add_num) = X_dis(1:x_num,x_num+1:x_num+add_num)...
                +(X_nomlz(:,vari_index)-X_add(:,vari_index)').^2;
            X_dis(x_num+1:x_num+add_num,x_num+1:x_num+add_num) = X_dis(x_num+1:x_num+add_num,x_num+1:x_num+add_num)...
                +(X_add(:,vari_index)-X_add(:,vari_index)').^2;
        end
        X_dis(1:x_num,x_num+1:x_num+add_num)=sqrt(X_dis(1:x_num,x_num+1:x_num+add_num));
        X_dis(x_num+1:x_num+add_num,x_num+1:x_num+add_num)=sqrt(X_dis(x_num+1:x_num+add_num,x_num+1:x_num+add_num));
        X_dis(x_num+1:x_num+add_num,1:x_num)=X_dis(1:x_num,x_num+1:x_num+add_num)';

        rdibas_matrix=basis_function(X_dis);

        % calculate error
        inv_rdibas_matrix = rdibas_matrix\eye(x_num+add_num);
        beta = inv_rdibas_matrix*[Fval;object_function_surrogate(X_add.*(up_bou-low_bou)+low_bou)];
        GRSE_sq=sum((beta./diag(inv_rdibas_matrix)).^2);

        fval_quality=GRSE_sq;
    end
end

function [x_list_filter,index]=proFilterX(x_list,range)
x_list_filter=[];
index=[];

[x_number,variable_number]=size(x_list);

for x_index=1:x_number
    x=x_list(x_index,:);

    overlap_flag=false(1);
    for filter_index=1:size(x_list_filter,1)
        if sum(abs(x_list_filter(filter_index,:)-x))/variable_number < range
            overlap_flag=true(1);
            break;
        end
    end

    if ~overlap_flag
        x_list_filter=[x_list_filter;x];
        index=[index;x_index];
    end
end
end

%% machine learning
function [predict_function,CGPMF_model] = classifyGaussProcessMultiFidelity...
    (XHF,ClassHF,XLF,ClassLF,hyp)
% generate multi fidelity gaussian process classifier model
% version 6,this version is assembly of gpml-3.6 EP method
% X,XHF,XLF is x_number x variable_number matirx
% Class,ClassH,ClassLF is x_number x 1 matrix
% low_bou,up_bou is 1 x variable_number matrix
%
% input:
% XHF,ClassHF,XLF,ClassLF,hyp(mean,cov(lenD,etaD,lenL,etaL,rho))
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
X = [XHF;XLF];
Class = [ClassHF;ClassLF];
[x_number,variable_number] = size(X);
x_HF_number = size(XHF,1);
x_LF_number = size(XLF,1);
if nargin < 5
    hyp.mean = 0;
    hyp.cov = zeros(1,variable_number*2+3);
end

% normalization data
aver_X = mean(X);
stdD_X = std(X);
index__ = find(stdD_X  ==  0);
if  ~isempty(index__),stdD_X(index__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;

object_function = @(x) objectNLLMFGPC(x,{@infEP},{@meanConst},{@calCovMF},{@likErf},X_nomlz,Class);
hyp_x = [hyp.mean,hyp.cov];

% [fval,gradient] = object_function(hyp_x)
% [fval_differ,gradient_differ] = differ(object_function,hyp_x)

low_bou_hyp = -4*ones(1,2*variable_number+4);
up_bou_hyp = 4*ones(1,2*variable_number+4);
hyp_x = fmincon(object_function,hyp_x,[],[],[],[],low_bou_hyp,up_bou_hyp,[],...
    optimoptions('fmincon','Display','iter','SpecifyObjectiveGradient',true,...
    'MaxFunctionEvaluations',20,'OptimalityTolerance',1e-6));

hyp.mean = hyp_x(1);
hyp.cov = hyp_x(2:end);
hyp.lik = [];
post = infEP(hyp,{@meanConst},{@calCovMF},{@likErf},X_nomlz,Class);
predict_function = @(x_pred) classifyGaussPredictor...
    (x_pred,hyp,{@meanConst},{@calCovMF},{@likErf},post,X_nomlz,aver_X,stdD_X);

% output model
X = {XHF,XLF};
Class = {ClassHF,ClassLF};
CGPMF_model.X = X;
CGPMF_model.Class = Class;
CGPMF_model.aver_X = aver_X;
CGPMF_model.stdD_X = stdD_X;
CGPMF_model.predict_function = predict_function;
CGPMF_model.hyp = hyp;
CGPMF_model.post = post;

    function [fval,gradient] = objectNLLMFGPC(x,inf,mean,cov,lik,X,Y)
        hyp_iter.mean = x(1);
        hyp_iter.cov = x(2:end);
        hyp_iter.lik = [];

        if nargout < 2
            [~,nlZ] = feval(inf{:},hyp_iter,mean,cov,lik,X,Y);
            fval = nlZ;
        elseif nargout < 3
            [~,nlZ,dnlZ] = feval(inf{:},hyp_iter,mean,cov,lik,X,Y);
            fval = nlZ;
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
            nact = id(end);          % set counter to index of last processed data point
        end

        possibility = exp(possibility);
        class = ones(pred_num,1);
        index_list = possibility < 0.5;
        class(index_list) = -1;
    end

    function [K,dK_dcov] = calCovMF(cov,X,Z)
        % obtain covariance of x
        % cov: lenD,etaD,lenL,etaL,rho
        % len equal to 1/len_origin.^2
        %
        % % k = eta*exp(-sum(x_dis*theta)/vari_num);
        %
        [x_num,vari_num] = size(X);

        lenD = exp(cov(1:vari_num));
        etaD = exp(cov(vari_num+1));
        lenL = exp(cov(vari_num+1+(1:vari_num)));
        etaL = exp(cov(2*(vari_num+1)));
        rho = exp(cov(end));

        if nargin > 2 && nargout < 2 && ~isempty(Z)
            if strcmp(Z,'diag')
                K = rho*rho*etaL+etaD;
                return
            end
        end

        % predict
        if nargin > 2 && nargout < 2 && ~isempty(Z)
            [z_num,vari_num] = size(Z);
            % initializate square of X inner distance sq/ vari_num
            sq_dis_v = zeros(x_num,z_num,vari_num);
            for len_index = 1:vari_num
                sq_dis_v(:,:,len_index) = (X(:,len_index)-Z(:,len_index)').^2/vari_num;
            end

            % exp of x__x with D
            exp_disD = zeros(x_HF_number,z_num);
            for len_index = 1:vari_num
                exp_disD = exp_disD+...
                    sq_dis_v(1:x_HF_number,:,len_index)*lenD(len_index);
            end
            exp_disD = exp(-exp_disD);

            % exp of x__x with L
            exp_disL = zeros(x_num,z_num);
            for len_index = 1:vari_num
                exp_disL = exp_disL+...
                    sq_dis_v(1:x_num,:,len_index)*lenL(len_index);
            end
            exp_disL = exp(-exp_disL);

            K = exp_disL;
            K(1:x_HF_number,:) = rho*rho*etaL*K(1:x_HF_number,:)+etaD*exp_disD;
            K(x_HF_number+1:end,:) = rho*etaL*K(x_HF_number+1:end,:);
        else
            % initializate square of X inner distance sq/ vari_num
            sq_dis_v = zeros(x_num,x_num,vari_num);
            for len_index = 1:vari_num
                sq_dis_v(:,:,len_index) = (X(:,len_index)-X(:,len_index)').^2/vari_num;
            end

            % exp of x__x with H
            exp_disD = zeros(x_num);
            for len_index = 1:vari_num
                exp_disD(1:x_HF_number,1:x_HF_number) = exp_disD(1:x_HF_number,1:x_HF_number)+...
                    sq_dis_v(1:x_HF_number,1:x_HF_number,len_index)*lenD(len_index);
            end
            exp_disD(1:x_HF_number,1:x_HF_number) = exp(-exp_disD(1:x_HF_number,1:x_HF_number));
            KD = etaD*exp_disD;

            % exp of x__x with L
            exp_disL = zeros(x_num);
            for len_index = 1:vari_num
                exp_disL = exp_disL+...
                    sq_dis_v(1:end,1:end,len_index)*lenL(len_index);
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
                dK_dcov = cell(1,2*vari_num+3);

                % len D
                for len_index = 1:vari_num
                    dK_dlenD = zeros(x_num);
                    dK_dlenD(1:x_HF_number,1:x_HF_number) = -KD(1:x_HF_number,1:x_HF_number).*...
                        sq_dis_v(1:x_HF_number,1:x_HF_number,len_index)*lenD(len_index);
                    dK_dcov{len_index} = dK_dlenD;
                end

                % eta D
                dK_dcov{vari_num+1} = KD;

                % len L
                for len_index = 1:vari_num
                    dK_dlenL = -KL.*sq_dis_v(:,:,len_index)*lenL(len_index);
                    dK_dcov{(vari_num+1)+len_index} = dK_dlenL;
                end

                % eta L
                dK_dcov{2*(vari_num+1)} = KL;

                % rho
                dK_drho = zeros(x_num);
                dK_drho(1:x_HF_number,1:x_HF_number) = ...
                    2*rho*rho*eta_exp_disL(1:x_HF_number,1:x_HF_number);
                dK_drho(1:x_HF_number,(x_HF_number+1):end) = ...
                    rho*eta_exp_disL(1:x_HF_number,(x_HF_number+1):end);
                dK_drho((x_HF_number+1):end,1:x_HF_number) = ...
                    dK_drho(1:x_HF_number,(x_HF_number+1):end)';
                dK_dcov{end} = dK_drho;
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
        tol = 1e-4; max_sweep = 100; min_sweep = 2;     % tolerance to stop EP iterations

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
        % The function is based on index 5725 in Hart et al. and gsl_sf_log_erfc_e.
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
index__ = find(stdD_X == 0);
if  ~isempty(index__),stdD_X(index__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;

% if x_number equal 1,clustering cannot done
if x_number == 1
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
                1/sum((X_center_dis_sq(classify_index,x_index)./X_center_dis_sq(:,x_index)).^(1/(m-1)));
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

%% surrogate model
function [object_function_surrogate,nonlcon_function_surrogate,output] = getSurrogateFunction...
    (x_list,fval_list,con_list,coneq_list)
% base on library_data to create radialbasis model and function
% if input model,function will updata model
% object_function is single fval output
% nonlcon_function is normal nonlcon_function which include con,coneq
% con is colume vector,coneq is colume vector
% var_function is same
%
basis_function = @(r) r.^3;

[predict_function_fval,radbas_model_fval] = interpRadialBasisPreModel...
    (x_list,fval_list,basis_function);

if ~isempty(con_list)
    predict_function_con = cell(size(con_list,2),1);
    radbas_model_con_list = cell(size(con_list,2),1);
    for con_index = 1:size(con_list,2)
        [predict_function_con{con_index},radbas_model_con_list{con_index}] = interpRadialBasisPreModel...
            (x_list,con_list(:,con_index),basis_function);
    end
else
    predict_function_con = [];
    radbas_model_con_list = [];
end

if ~isempty(coneq_list)
    predict_function_coneq = cell(size(coneq_list,2),1);
    radbas_model_coneq_list = cell(size(coneq_list,2),1);
    for coneq_index = 1:size(coneq_list,2)
        [predict_function_coneq{coneq_index},radbas_model_con_list{coneq_index}] = interpRadialBasisPreModel...
            (x_list,coneq_list(:,coneq_index),basis_function);
    end
else
    predict_function_coneq = [];
    radbas_model_coneq_list = [];
end

object_function_surrogate = @(X_predict) objectFunctionSurrogate(X_predict,predict_function_fval);
if isempty(predict_function_con) && isempty(predict_function_coneq)
    nonlcon_function_surrogate = [];
else
    nonlcon_function_surrogate = @(X_predict) nonlconFunctionSurrogate(X_predict,predict_function_con,predict_function_coneq);
end

output.radbas_model_fval=radbas_model_fval;
output.radbas_model_con_list=radbas_model_con_list;
output.radbas_model_coneq_list=radbas_model_coneq_list;

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

function [predict_function_MFRBF, MFRBF_model] = interpRadialBasisMultiFidelityPreModel...
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
            error('interpHieraKrigingPreModel: low fidelity lack predict function');
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
        index__ = find(stdD_X == 0);
        if  ~isempty(index__), stdD_X(index__) = 1; end
        index__ = find(stdD_Y == 0);
        if  ~isempty(index__), stdD_Y(index__) = 1; end

        XLF_nomlz = (XLF-aver_X)./stdD_X;
        YLF_nomlz = (YLF-aver_Y)./stdD_Y;

        if isempty(basis_func_LF)
            c = (prod(max(XLF_nomlz) - min(XLF_nomlz))/x_LF_number)^(1/variable_number);
            basis_func_LF = @(r) exp(-(r.^2)/c);
        end

        % initialization distance of XLF_nomlz
        XLF_dis = zeros(x_LF_number, x_LF_number);
        for variable_index = 1:variable_number
            XLF_dis = XLF_dis + ...
                (XLF_nomlz(:, variable_index) - XLF_nomlz(:, variable_index)').^2;
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
        error('interpHieraKrigingPreModel: error input');
end
MFRBF_model.LF_model = LF_model;
predict_function_LF = LF_model.predict_function;

% predict LF value at XHF point
YHF_pred = predict_function_LF(XHF);

% nomalizae
YHF_pred_nomlz = (YHF_pred - aver_Y)./stdD_Y;

% second step
% construct MFRBF model

% normalize data
aver_X = mean(XHF);
stdD_X = std(XHF);
aver_Y = mean(YHF);
stdD_Y = std(YHF);
index__ = find(stdD_X == 0);
if ~isempty(index__), stdD_X(index__) = 1;end
index__ = find(stdD_Y == 0);
if ~isempty(index__), stdD_Y(index__) = 1;end
XHF_nomlz = (XHF - aver_X)./stdD_X;
YHF_nomlz = (YHF - aver_Y)./stdD_Y;

if isempty(basis_func_HF)
    c = (prod(max(XHF_nomlz) - min(XHF_nomlz))/x_HF_number)^(1/variable_number);
    basis_func_HF = @(r) exp(-(r.^2)/c);
end

% initialization distance of XHF_nomlz
XHF_dis = zeros(x_HF_number, x_HF_number);
for variable_index = 1:variable_number
    XHF_dis = XHF_dis + ...
        (XHF_nomlz(:, variable_index) - XHF_nomlz(:, variable_index)').^2;
end
XHF_dis = sqrt(XHF_dis);

[beta_HF, rdibas_matrix_HF] = interpMultiRadialBasis...
    (XHF_dis, YHF_nomlz, basis_func_HF, x_HF_number, YHF_pred_nomlz);

% initialization predict function
predict_function_MFRBF = @(X_predict) interpMultiRadialBasisPredictor...
    (X_predict, XHF_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
    x_HF_number, variable_number, beta_HF, basis_func_HF, predict_function_LF);

MFRBF_model.X = XHF;
MFRBF_model.Y = YHF;
MFRBF_model.radialbasis_matrix = rdibas_matrix_HF;
MFRBF_model.beta = beta_HF;

MFRBF_model.aver_X = aver_X;
MFRBF_model.stdD_X = stdD_X;
MFRBF_model.aver_Y = aver_Y;
MFRBF_model.stdD_Y = stdD_Y;
MFRBF_model.basis_function = basis_func_HF;

MFRBF_model.predict_function = predict_function_MFRBF;

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
        rdibas_matrix = rdibas_matrix + eye(x_number)*1e-6;

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
        for vari_index = 1:vari_num
            X_dis_pred = X_dis_pred + ...
                (X_pred_nomlz(:, vari_index) - X_nomlz(:, vari_index)').^2;
        end
        X_dis_pred = sqrt(X_dis_pred);

        % predict variance
        Y_pred = basis_function(X_dis_pred)*beta;

        % normalize data
        Y_pred = Y_pred*stdD_Y + aver_Y;
    end

    function [beta, rdibas_matrix] = interpMultiRadialBasis...
            (X_dis, Y, basis_function, x_number, YHF_pred)
        % interp polynomial responed surface core function
        % calculation beta
        %
        % Copyright 2022 Adel
        %
        rdibas_matrix = basis_function(X_dis);

        % stabilize matrix
        rdibas_matrix = rdibas_matrix + eye(x_number)*1e-6;

        % add low fildelity value
        H = [rdibas_matrix.*YHF_pred, rdibas_matrix];

        % solve beta
        beta = H'*((H*H')\Y);
    end

    function [Y_pred] = interpMultiRadialBasisPredictor...
            (X_pred, X_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
            x_num, vari_num, beta, basis_function, predict_function_LF)
        % radial basis function interpolation predict function
        %
        [x_pred_num, ~] = size(X_pred);

        % normalize data
        X_pred_nomlz = (X_pred - aver_X)./stdD_X;

        % calculate distance
        X_dis_pred = zeros(x_pred_num, x_num);
        for vari_index = 1:vari_num
            X_dis_pred = X_dis_pred + ...
                (X_pred_nomlz(:, vari_index) - X_nomlz(:, vari_index)').^2;
        end
        X_dis_pred = sqrt(X_dis_pred);

        % predict low fildelity value
        Y_pred_LF = predict_function_LF(X_pred);

        % nomalizae
        Y_pred_LF_nomlz = (Y_pred_LF - aver_Y)./stdD_Y;

        % combine two matrix
        rdibas_matrix_pred = basis_function(X_dis_pred);
        H = [rdibas_matrix_pred.*Y_pred_LF_nomlz, rdibas_matrix_pred];

        % predict variance
        Y_pred = H*beta;

        % normalize data
        Y_pred = Y_pred*stdD_Y + aver_Y;
    end

end

%% LHD
% function [X_sample,dist_min_nomlz,X_total] = getLatinHypercube...
%     (sample_number,variable_number,...
%     low_bou,up_bou,X_exist,cheapcon_function)
% % generate latin hypercube desgin
% %
% % more uniform point distribution by simulating particle motion
% %
% % input:
% % sample number(new point to sample),variable_number
% % low_bou,up_bou,x_exist_list,cheapcon_function
% %
% % output:
% % X_sample,dist_min_nomlz(min distance of normalize data)
% % X_total,include all data in area
% %
% % Copyright 2023 3 Adel
% %
% if nargin < 6
%     cheapcon_function = [];
%     if nargin < 5
%         X_exist = [];
%         if nargin < 4
%             up_bou = ones(1,variable_number);
%             if nargin < 3
%                 low_bou = zeros(1,variable_number);
%                 if nargin < 2
%                     error('getLatinHypercube: lack variable_number');
%                 end
%             end
%         end
%     end
% end
%
% iteration_max = 100;
%
% % check x_exist_list if meet boundary
% if ~isempty(X_exist)
%     if size(X_exist,2) ~= variable_number
%         error('getLatinHypercube: x_exist_list variable_number error');
%     end
%     index = find(X_exist < low_bou-eps);
%     index = [index,find(X_exist > up_bou+eps)];
%     if ~isempty(index)
%         error('getLatinHypercube: x_exist_list range error');
%     end
%     X_exist_nomlz = (X_exist-low_bou)./(up_bou-low_bou);
% else
%     X_exist_nomlz = [];
% end
%
% exist_number = size(X_exist,1);
% total_number = sample_number+exist_number;
% if sample_number < 0
%     X_total = X_exist;
%     X_sample = [];
%     dist_min_nomlz = getMinDistance(X_exist_nomlz);
%     return;
% end
%
% low_bou_nomlz = zeros(1,variable_number);
% up_bou_nomlz = ones(1,variable_number);
%
% % obtain initial point
% if ~isempty(cheapcon_function)
%     % obtian feasible point
%     X_quasi_nomlz = [];
%
%     % check if have enough X_supply_nomlz
%     iteration = 0;
%     while size(X_quasi_nomlz,1) < 10*sample_number && iteration < 500
%         X_quasi_nomlz_initial = rand(10*sample_number,variable_number);
%
%         qusai_index = [];
%         for x_index = 1:size(X_quasi_nomlz_initial,1)
%             if cheapcon_function(X_quasi_nomlz_initial(x_index,:).*(up_bou-low_bou)+low_bou) <= 0
%                 qusai_index = [qusai_index,x_index];
%             end
%         end
%         X_quasi_nomlz = [X_quasi_nomlz;X_quasi_nomlz_initial(qusai_index,:)];
%
%         iteration = iteration+1;
%     end
%
%     if iteration == 500 && size(X_quasi_nomlz,1) < sample_number
%         error('getLatinHypercube: feasible quasi point cannot be found');
%     end
%
%     % use fuzzy clustering to get feasible point center
%     X_sample_nomlz = clusteringFuzzy(X_quasi_nomlz,sample_number,2);
%     X_feas_center_nomlz = X_sample_nomlz;
%
% %     scatter(X_quasi_nomlz(:,1),X_quasi_nomlz(:,2));
% %     hold on;
% %     scatter(X_sample_nomlz(:,1),X_sample_nomlz(:,2),'red');
% %     hold off;
% else
%     X_sample_nomlz = rand(sample_number,variable_number);
% end
%
% % pic_num = 1;
%
% iteration = 0;
% fval_list = zeros(sample_number,1);
% gradient_list = zeros(sample_number,variable_number);
% while iteration < iteration_max
%     % change each x place by newton methods
%     for x_index = 1:sample_number
%
%         % get gradient
%         [fval_list(x_index,1),gradient_list(x_index,:)] = calParticleEnergy...
%             (X_sample_nomlz(x_index,:),[X_sample_nomlz(1:x_index-1,:);X_sample_nomlz(x_index+1:end,:);X_exist_nomlz],...
%             sample_number,variable_number,low_bou_nomlz-1/2/sample_number,up_bou_nomlz+1/2/sample_number);
%
% %         energy_function = @(x) calParticleEnergy...
% %             ([x],[X_sample_nomlz(1:x_index-1,:);X_sample_nomlz(x_index+1:end,:);X_exist_nomlz],...
% %             sample_number,variable_number,low_bou_nomlz-1/2/sample_number,up_bou_nomlz+1/2/sample_number);
% %         drawFunction(energy_function,low_bou_nomlz(1:2),up_bou_nomlz(1:2))
%
% %         [fval,gradient] = differ(energy_function,X_sample_nomlz(x_index,:));
% %         gradient'-gradient_list(x_index,:)
%     end
%
%     C = (1-iteration/iteration_max)*0.5;
%
%     % updata partical location
%     for x_index = 1:sample_number
%         x = X_sample_nomlz(x_index,:);
%         gradient = gradient_list(x_index,:);
%
%         % check if feasible
%         if ~isempty(cheapcon_function)
%             con = cheapcon_function(x.*(up_bou-low_bou)+low_bou);
%             % if no feasible,move point to close point
%             if con > 0
%                 %                 % search closest point
%                 %                 dx_center = x-X_feas_center_nomlz;
%                 %                 [~,index] = min(norm(dx_center,"inf"));
%                 %                 gradient = dx_center(index(1),:);
%
%                 gradient = x-X_feas_center_nomlz(x_index,:);
%             end
%         end
%
%         gradient = min(gradient,0.5);
%         gradient = max(gradient,-0.5);
%         x = x-gradient*C;
%
%         boolean = x < low_bou_nomlz;
%         x(boolean) = -x(boolean);
%         boolean = x > up_bou_nomlz;
%         x(boolean) = 2-x(boolean);
%         X_sample_nomlz(x_index,:) = x;
%     end
%
%     iteration = iteration+1;
%
% %     scatter(X_sample_nomlz(:,1),X_sample_nomlz(:,2));
% %     bou = [low_bou_nomlz(1:2);up_bou_nomlz(1:2)];
% %     axis(bou(:));
% %     grid on;
% %
% %     radius = 1;
% %     hold on;
% %     rectangle('Position',[-radius,-radius,2*radius,2*radius],'Curvature',[1 1])
% %     hold off;
% %
% %     drawnow;
% %     F = getframe(gcf);
% %     I = frame2im(F);
% %     [I,map] = rgb2ind(I,256);
% %     if pic_num  =  =  1
% %         imwrite(I,map,'show_trajectory_constrain.gif','gif','Loopcount',inf,'DelayTime',0.1);
% %     else
% %         imwrite(I,map,'show_trajectory_constrain.gif','gif','WriteMode','append','DelayTime',0.1);
% %     end
% %     pic_num = pic_num + 1;
% end
%
% % process out of boundary point
% for x_index = 1:sample_number
%     x = X_sample_nomlz(x_index,:);
%     % check if feasible
%     if ~isempty(cheapcon_function)
%         con = cheapcon_function(x);
%         % if no feasible,move point to close point
%         if con > 0
%             % search closest point
%             dx_center = x-X_feas_center_nomlz;
%             [~,index] = min(norm(dx_center,"inf"));
%
%             gradient = dx_center(index(1),:);
%         end
%     end
%     x = x-gradient*C;
%
%     boolean = x < low_bou_nomlz;
%     x(boolean) = -x(boolean);
%     boolean = x > up_bou_nomlz;
%     x(boolean) = 2-x(boolean);
%     X_sample_nomlz(x_index,:) = x;
% end
%
% dist_min_nomlz = getMinDistance([X_sample_nomlz;X_exist_nomlz]);
% X_sample = X_sample_nomlz.*(up_bou-low_bou)+low_bou;
% X_total = [X_sample;X_exist];
%
%     function [fval,gradient] = calParticleEnergy...
%             (x,X_surplus,sample_number,variable_number,low_bou,up_bou)
%         % function describe distance between X and X_supply
%         % x is colume vector and X_surplus is matrix which is num-1 x var
%         % low_bou_limit__ and up_bou_limit__ is colume vector
%         % variable in colume
%         %
%         a__ = 10;
%         a_bou__ = 10;
%
%         sign__ = ((x > X_surplus)-0.5)*2;
%
%         xi__ = -a__*(x-X_surplus).*sign__;
%         psi__ = a_bou__*(low_bou-x);
%         zeta__ = a_bou__*(x-up_bou);
%
%         exp_psi__ = exp(psi__);
%         exp_zeta__ = exp(zeta__);
%
% %         sum_xi__ = sum(xi__,2)/variable_number;
% %         exp_sum_xi__ = exp(sum_xi__);
% %         % get fval
% %         fval = sum(exp_sum_xi__,1)+...
% %             sum(exp_psi__+exp_zeta__,2)/variable_number;
%
% %         exp_xi__ = exp(xi__);
% %         sum_exp_xi__ = sum(exp_xi__,2);
% %         % get fval
% %         fval = sum(sum_exp_xi__,1)/variable_number/sample_number+...
% %             sum(exp_psi__+exp_zeta__,2)/variable_number;
%
%         sum_xi__ = sum(xi__,2)/variable_number;
%         exp_sum_xi__ = exp(sum_xi__);
%         exp_xi__ = exp(xi__);
%         sum_exp_xi__ = sum(exp_xi__,2)/variable_number;
%         % get fval
%         fval = (sum(sum_exp_xi__,1)+sum(exp_sum_xi__,1))/2/sample_number+...
%             sum(exp_psi__+exp_zeta__,2)/variable_number*0.1;
%
% %         % get gradient
% %         gradient = sum(-a__*sign__.*exp_sum_xi__,1)/variable_number+...
% %             (-a_bou__*exp_psi__+a_bou__*exp_zeta__)/variable_number;
%
% %         % get gradient
% %         gradient = sum(-a__*sign__.*exp_xi__,1)/variable_number/sample_number+...
% %             (-a_bou__*exp_psi__+a_bou__*exp_zeta__)/variable_number;
%
%         % get gradient
%         gradient = (sum(-a__*sign__.*exp_sum_xi__,1)+sum(-a__*sign__.*exp_xi__,1))/2/variable_number/sample_number+...
%             (-a_bou__*exp_psi__+a_bou__*exp_zeta__)/variable_number*0.1;
%
%     end
%
%     function distance_min__ = getMinDistance(x_list__)
%         % get distance min from x_list
%         %
%         if isempty(x_list__)
%             distance_min__ = [];
%             return;
%         end
%
%         % sort x_supply_list_initial to decrese distance calculate times
%         x_list__ = sortrows(x_list__,1);
%         sample_number__ = size(x_list__,1);
%         variable_number__ = size(x_list__,2);
%         distance_min__ = variable_number__;
%         for x_index__ = 1:sample_number__
%             x_curr__ = x_list__(x_index__,:);
%             x_next_index__ = x_index__ + 1;
%             % first dimension only search in min_distance
%             search_range__ = variable_number__;
%             while x_next_index__ <= sample_number__ &&...
%                     (x_list__(x_next_index__,1)-x_list__(x_index__,1))^2 ...
%                     < search_range__
%                 x_next__ = x_list__(x_next_index__,:);
%                 distance_temp__ = sum((x_next__-x_curr__).^2);
%                 if distance_temp__ < distance_min__
%                     distance_min__ = distance_temp__;
%                 end
%                 if distance_temp__ < search_range__
%                     search_range__ = distance_temp__;
%                 end
%                 x_next_index__ = x_next_index__+1;
%             end
%         end
%         distance_min__ = sqrt(distance_min__);
%     end
%
% end

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
% low_bou,up_bou,x_exist_list,cheapcon_function
%
% output:
% X_sample,dist_min_nomlz(min distance of normalize data)
% X_total,include all data in area
%
% reference: [1] LONG T, LI X, SHI R, et al., Gradient-Free
% Trust-Region-Based Adaptive Response Surface Method for Expensive
% Aircraft Optimization[J]. AIAA Journal, 2018, 56(2): 862-73.
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

iteration_max = 100*sample_number;

% check x_exist_list if meet boundary
if ~isempty(X_exist)
    if size(X_exist,2) ~= variable_number
        error('getLatinHypercube: x_exist_list variable_number error');
    end
    X_exist_nomlz = (X_exist-low_bou)./(up_bou-low_bou);
else
    X_exist_nomlz = [];
end

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
        error('getLatinHypercube: feasible quasi point cannot be found');
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
        if isempty(x_list__)
            distance_min__ = [];
            return;
        end

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
