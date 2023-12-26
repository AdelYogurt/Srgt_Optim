clc;
clear;
close all hidden;

benchmark_function=BenchmarkFunction();

data_library_name='optimalSurrogate_MO_K_SVM_FC_PSO_result';

object_function=@(x) benchmark_function.multiZDT3Object(x);
variable_number=10;
low_bou=zeros(1,variable_number);
up_bou=ones(1,variable_number);
nonlcon_function=[];
cheapcon_function=[];

% object_function=@(x) benchmark_function.multiTNKObject(x);
% variable_number=2;
% low_bou=zeros(1,2);
% up_bou=ones(1,2)*pi;
% nonlcon_function=@(x) benchmark_function.multiTNKNonlcon(x);
% cheapcon_function=[];

delete([data_library_name,'.txt']);
delete('result_total.txt');

[x_pareto,fval_pareto,NFE,output]=optimalSurrogateMOKSVMFCPSO...
    (object_function,variable_number,low_bou,up_bou,nonlcon_function,...
    cheapcon_function,[]);

line(fval_pareto(:,1),fval_pareto(:,2));

%% main function
function [x_pareto,fval_pareto,NFE,output]=optimalSurrogateMOKSVMFCPSO...
    (object_function,variable_number,low_bou,up_bou,nonlcon_function,...
    cheapcon_function,model_function,....
    NFE_max,iteration_max,torlance,nonlcon_torlance)
% surrogate base multi objective optimal method version 4
% constrain proccess by multiply constrain improve 
% use SVM to get front field
% FS_FCM to get global best, pareto point as personal best
% x_list is x_number x variable_number matrix
% object_function format is [fval(colume vector)]
% both nonlcon_function and cheapcon_function format is [con(colume vector),coneq(colume vector)]
% model_function should output fval, format is [fval(colume vector),con(colume vector),coneq(colume vector)]
% con or coneq can be colume vector if there was more than one constrain
%
% Copyright Adel 2022.12
%
if nargin < 11 || isempty(nonlcon_torlance)
    nonlcon_torlance=1e-3;
    if nargin < 10 || isempty(torlance)
        torlance=1e-3;
        if nargin < 9
            iteration_max=[];
            if nargin < 8
                NFE_max=[];
            end
        end
    end
end

if nargin < 7
    model_function=[];
    if nargin < 6
        cheapcon_function=[];
        if nargin < 5
            nonlcon_function=[];
        end
    end
end

low_bou=low_bou(:)';
up_bou=up_bou(:)';

DRAW_FIGURE_FLAG=0; % whether draw SVM data
DRAW_PARETO_FLAG=0; % whether draw pareto data
INFORMATION_FLAG=1; % whether print data
CONVERGENCE_JUDGMENT_FLAG=0; % whether judgment convergence

if isempty(iteration_max)
    iteration_max=500;
end

% parameter
sample_number_initial=10*variable_number;
particle_number=20*variable_number;
sample_number_iteration=variable_number;
sample_number_data=50*particle_number;
lead_number=5*variable_number;
global_number=5;

penalty_SVM=10;

% kernal_function_FS_FCM=@(sq) exp(-sq/2/0.1^2);
m=2; % clustering parameter
vec_abs_max=((up_bou-low_bou)/2)*0.5;

pareto_torlance=0.01;
protect_range=1e-5;

data_library_name='optimalSurrogate_MO_K_SVM_FC_PSO_result';
file_result=fopen('result_total.txt','a');
fprintf(file_result,'%s\n',datetime);
fclose(file_result);
clear('file_result');

done=0;NFE=0;iteration=0;

% if do not input model_function, generate model_function
if isempty(model_function)
    model_function=@(x) modelFunction(x,object_function,nonlcon_function);
end

iteration=iteration+1;

% step 0
% load exist data
[x_list,fval_list,con_list,coneq_list]=dataLibraryLoad(data_library_name,low_bou,up_bou);
sample_number_initial=sample_number_initial-size(fval_list,1);

% step 1
% use latin hypercube method to get initial particle list
[~,x_updata_list,~]=getLatinHypercube...
    (sample_number_initial,variable_number,[],low_bou,up_bou,cheapcon_function);

x_particle_list=rand(particle_number,variable_number);
vec_list=(rand(particle_number,variable_number)-0.5).*vec_abs_max;

% detech expensive constraints
if ~isempty(x_updata_list)
    [fval_list,con_list,coneq_list]=dataLibraryUpdata...
        (data_library_name,model_function,x_updata_list(1,:));NFE=NFE+1;
    x_updata_list=x_updata_list(2:end,:);
else
    [x_list,fval_list,con_list,coneq_list]=dataLibraryLoad(data_library_name,low_bou,up_bou);
end
fval_number=size(fval_list,2);
con_number=size(con_list,2);
coneq_number=size(coneq_list,2);
if con_number > 0 || coneq_number > 0
    expensive_nonlcon_flag=1;
else
    expensive_nonlcon_flag=0;
end

index_lead_list=[];

result_pareto=struct('x_pareto',[],'fval_pareto',[]);
result_pareto=repmat(result_pareto,iteration_max,1);

% NFE setting
if isempty(NFE_max)
    if expensive_nonlcon_flag == 1
        NFE_max=100*variable_number;
    else
        NFE_max=50*variable_number;
    end
end

% import data from data library
[x_list,fval_list,con_list,coneq_list]=dataLibraryLoad...
    (data_library_name,low_bou,up_bou);

kriging_model_fval=[];
kriging_model_con=[];
kriging_model_coneq=[];
while ~done
    % step 3
    % updata data library by x_list
    [x_updata_list,fval_updata_list,con_updata_list,coneq_updata_list,NFE_updata]=dataLibraryUpdataProtect...
        (data_library_name,model_function,x_updata_list,...
        x_list,fval_list,con_list,coneq_list,...
        low_bou,up_bou,protect_range);NFE=NFE+NFE_updata;
    
    x_list=[x_list;x_updata_list];
    fval_list=[fval_list;fval_updata_list];
    con_list=[con_list;con_updata_list];
    coneq_list=[coneq_list;coneq_updata_list];
    
    % nomalization con average
    fval_max=mean(abs(fval_list),1);
    fval_nomlz_list=fval_list;
    if ~isempty(con_list)
        con_max_list=mean(abs(con_list),1);
        con_nomlz_list=con_list./con_max_list*10;
    else
        con_max_list=[];
        con_nomlz_list=[];
    end
    if ~isempty(coneq_list)
        coneq_max_list=mean(abs(coneq_list),1);
        coneq_nomlz_list=coneq_list./coneq_max_list*10;
    else
        coneq_max_list=[];
        coneq_nomlz_list=[];
    end
    
    % step 4
    % generate kriging model use normalization fval
    % if x_list more than 11*D-1+25, select 11*D-1+25 to construct model
    if size(x_list,1) > (11*variable_number-1+25)
        % select by pareto front
        [pareto_index_list,~]=getParetoFront...
            (fval_list,con_list,coneq_list,...
            pareto_torlance);
        
        % rand select from remainder
        index_list=1:size(x_list,1);
        index_list(pareto_index_list)=[];
        index_list=index_list(randi(length(index_list),...
            [1,(11*variable_number-1+25-length(pareto_index_list))]));
        
        x_list_model=[x_list(pareto_index_list,:);x_list(index_list,:)];
        fval_nomlz_model_list=[fval_nomlz_list(pareto_index_list,:);fval_nomlz_list(index_list,:)];
        if con_number > 0
            con_nomlz_model_list=[con_nomlz_list(pareto_index_list,:);con_nomlz_list(index_list,:)];
        else
            con_nomlz_model_list=[];
        end
        if coneq_number > 0
            coneq_nomlz_model_list=[coneq_nomlz_list(pareto_index_list,:);coneq_nomlz_list(index_list,:)];
        else
            coneq_nomlz_list=[];
        end       
        
    else
        x_list_model=x_list;
        fval_nomlz_model_list=fval_nomlz_list;
        con_nomlz_model_list=con_nomlz_list;
        coneq_nomlz_model_list=coneq_nomlz_list;
    end
    [kriging_model_fval,kriging_model_con,kriging_model_coneq,output_kriging]=getKrigingModel...
        (x_list_model,fval_nomlz_model_list,con_nomlz_model_list,coneq_nomlz_model_list,...
        kriging_model_fval,kriging_model_con,kriging_model_coneq);
    object_function_surrogate=output_kriging.object_function_surrogate;
    object_function_variance=output_kriging.object_function_variance;
    nonlcon_function_surrogate=output_kriging.nonlcon_function_surrogate;
    nonlcon_function_variance=output_kriging.nonlcon_function_variance;
    
    if DRAW_FIGURE_FLAG && variable_number < 3
%         drawFunction2(object_function_variance,low_bou,up_bou);
    end
    
    % step 6
    % find best result to record
    [pareto_index_list,feasible_index_list]=getParetoFront...
        (fval_list,con_list,coneq_list,...
        pareto_torlance);
    x_pareto=x_list(pareto_index_list,:);
    fval_pareto=fval_list(pareto_index_list,:);
    if con_number > 0
        con_pareto=con_list(pareto_index_list,:);
    else
        con_pareto=[];
    end
    if coneq_number > 0
        coneq_pareto=coneq_list(pareto_index_list,:);
    else
        coneq_pareto=[];
    end
    
    if INFORMATION_FLAG
        fprintf('iteration:          %-3d    NFE:    %-3d\n',iteration,NFE);
        for pareto_index=1:size(x_pareto,1)
            fprintf('x_pareto:          %s\n',num2str(x_pareto(pareto_index,:)));
        end
        fprintf('\n');
        for pareto_index=1:size(x_pareto,1)
            fprintf('value_pareto:      %s\n',num2str(fval_pareto(pareto_index,:)));
        end
        fprintf('\n');
        if expensive_nonlcon_flag
            if con_number > 0
                for pareto_index=1:size(x_pareto,1)
                    fprintf('con_pareto:    %s\n',...
                        num2str(con_pareto(pareto_index,:)));
                end
            end
            fprintf('\n');
            for pareto_index=1:size(x_pareto,1)
                if coneq_number > 0
                    fprintf('coneq_pareto:  %s\n',...
                        num2str(num2str(coneq_pareto(pareto_index,:))));
                end
            end
            fprintf('\n');
        end
        fprintf('\n');
    end
    
    result_pareto(iteration).x_pareto=x_pareto;
    result_pareto(iteration).fval_pareto=fval_pareto;
    iteration=iteration+1;
    
    % forced interrupt
    if iteration > iteration_max || NFE >= NFE_max
        done=1;
    end
    
    % convergence judgment
    if CONVERGENCE_JUDGMENT_FLAG
        if (iteration > 1 && ...
                abs((fval_potential-fval_potential_old)/fval_potential_old) < torlance)
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
    
    % step 7 select max fitness
    index_lead_list=unique([index_lead_list;pareto_index_list]);
    if length(index_lead_list) > lead_number
        fval_lead_list=fval_list(index_lead_list,:);
        
        % evaluate multi objective fitness
        fitness_list=zeros(1,length(index_lead_list));
        for x_index=1:length(index_lead_list)
            select_index=1:length(index_lead_list);
            select_index(x_index)=[];
            
            fitness_list(x_index)=1-max(...
                min(fval_lead_list(x_index,:)-fval_lead_list(select_index,:),[],2),[],1);
        end
        
        [~,index_list]=sort(fitness_list,'descend');
        index_lead_list=index_lead_list(index_list(1:lead_number));
    end
    
    % step 8 construct SVM
    fval_label=zeros(size(x_list,1),1);
    fval_label(index_lead_list)=1;
    
    % use filter and train SVM
    [SVM_predict_function,SVM_model]=classifySupportVectorMachine...
        (x_list,fval_label,penalty_SVM,low_bou,up_bou);
    
    % candidate x list which probably in domain region
    x_data_list=lhsdesign(sample_number_data,variable_number).*...
        (up_bou-low_bou)+low_bou;
    x_candidate_list=[];
    for x_index=1:sample_number_data
        if  SVM_predict_function(x_data_list(x_index,:)')==1
            x_candidate_list=[x_candidate_list;x_data_list(x_index,:)];
        end
    end
    x_candidate_list=[x_candidate_list;x_pareto];
    iteration_temp=0;
    if size(x_candidate_list,1) < global_number*2
        center_point=sum(x_candidate_list,1)/size(x_candidate_list,1);
        d_bou_temp=(max(x_candidate_list)-min(x_candidate_list))*2;
        low_bou_temp=center_point-d_bou_temp;
        up_bou_temp=center_point+d_bou_temp;
        x_data_list_temp=rand(sample_number_data,variable_number).*...
            (up_bou_temp-low_bou_temp)+low_bou_temp;
        for x_index=1:sample_number_data
            if  SVM_predict_function(x_data_list_temp(x_index,:)')==1
                x_candidate_list=[x_candidate_list;x_data_list_temp(x_index,:)];
            end
        end
    end
    if size(x_candidate_list,1) < global_number*2
        x_candidate_list=[x_candidate_list;
            center_point+...
            (rand(global_number*2-size(x_candidate_list,1),variable_number)-0.5).*d_bou_temp];
    end
    x_candidate_list=max(x_candidate_list,low_bou);
    x_candidate_list=min(x_candidate_list,up_bou);
    
    % step 9 calculate clustering center as global_best
    if ~isempty(x_candidate_list)
        FC_model=classifyFuzzyClustering...
            (x_candidate_list,global_number,low_bou,up_bou,m);
        global_best_list=FC_model.center_list;
    end
    
    % select min distance x_candidate as personal best
    personal_best_list=zeros(particle_number,variable_number);
    for paritcal_index=1:particle_number
%         if ~isempty(x_candidate_list)
%             distance=sum(abs(x_particle_list(paritcal_index,:)-x_candidate_list),2);
%             [~,index]=min(distance);
%             personal_best_list(paritcal_index,:)=x_candidate_list(index(1),:);
%         else
            distance=sum(abs(x_particle_list(paritcal_index,:)-x_pareto),2);
            [~,index]=min(distance);
            personal_best_list(paritcal_index,:)=x_pareto(index(1),:);
%         end
    end
    
    % base on center list as global best to updata PSO particl
    %     W=rand(particle_number,variable_number)*0.4+0.1;
    %     Cp=(rand(particle_number,variable_number)*1.5+0.5);
    W=rand()*0.4+0.1;
    Cp=(rand()*1.5+0.5)*rand();
    vec_list=W.*vec_list+Cp.*(personal_best_list-x_particle_list);
    
    % global_number cluster particle
    FC_model=classifyFuzzyClustering...
        (x_particle_list,global_number,low_bou,up_bou,m);
    x_class_list=FC_model.x_class_list;
    
    % add x_center velocity
    for paritcal_index=1:particle_number
%         Cg=rand(particle_number,variable_number)*1.5+0.5;
        Cg=(rand()*1.5+0.5)*rand();
        vec_list(paritcal_index,:)=...
            vec_list(paritcal_index,:)+...
            Cg.*(global_best_list(x_class_list(paritcal_index),:)-...
            x_particle_list(paritcal_index,:));
    end
    vec_list=min(vec_list,vec_abs_max);
    vec_list=max(vec_list,-vec_abs_max);
    x_particle_list=x_particle_list+vec_list;
    
    x_particle_list=differEvolutionMutation...
        (low_bou,up_bou,x_particle_list,0.05);
    
    index=find(x_particle_list < low_bou);
    vec_list(index)=-vec_list(index);
    x_particle_list=max(x_particle_list,low_bou);
    index=find(x_particle_list > up_bou);
    vec_list(index)=-vec_list(index);
    x_particle_list=min(x_particle_list,up_bou);
    
    % process cheapcon_function, if pratical infeasible, rand select two and add middle
    if ~isempty(cheapcon_function)
        for x_index=1:particle_number
            while cheapcon_function(x_particle_list(x_index,:))
               index=randi(size(x_pareto,1));
               x_particle_list(x_index,:)=...
                   x_pareto(index)+((rand(1,variable_number)-0.5)*0.1).*(up_bou-low_bou)+low_bou;
            end
        end
    end
    
    %     line(x_particle_list(:,1),x_particle_list(:,2),...
    %         'Marker','o','LineStyle','none','Color','green');
    
    % step 9 evaluate particle EHVI
    if expensive_nonlcon_flag
        if con_number > 0
            con_predict_list=zeros(particle_number,con_number);
            con_predict_var_list=zeros(particle_number,con_number);
        end
        if coneq_number > 0
            coneq_predict_list=zeros(particle_number,coneq_number);
            coneq_predict_var_list=zeros(particle_number,coneq_number);
        end
        
        % classify particle to feasible and infeasible
        feasible_index_list=[];
        infeasible_index_list=[];
        for x_index=1:particle_number
            % evaluate all predict con, coneq
            x=x_particle_list(x_index,:);
            [con,coneq]=nonlcon_function_surrogate(x);
            
            if max([con,coneq]) > nonlcon_torlance
                infeasible_index_list=[infeasible_index_list;x_index];
                [con_var,coneq_var]=nonlcon_function_variance(x);
                if con_number > 0
                    con_predict_list(x_index,:)=con;
                    con_predict_var_list(x_index,:)=con_var;
                end
                if coneq_number > 0
                    coneq_predict_list(x_index,:)=coneq;
                    coneq_predict_var_list(x_index,:)=coneq_var;
                end
            else
                feasible_index_list=[feasible_index_list;x_index];
            end
        end
        
        feasiable_updata_number=ceil(sample_number_iteration*(length(feasible_index_list))/...
            (length(feasible_index_list)+length(infeasible_index_list)));
        infeasiable_updata_number=sample_number_iteration-feasiable_updata_number;
        x_updata_list=[];
        
        if ~isempty(feasible_index_list)
            [EHVI_list,fval_particle_list]=calHVLCB...
                (x_particle_list(feasible_index_list,:),object_function_surrogate,object_function_variance,...
                fval_number,fval_pareto,1);
            
            % step 10 evaluate particle crowded to make particl more uniform distribution
            crowd_list=calParetoCrowd(fval_particle_list,fval_pareto);
            crowd_list=crowd_list/sum(crowd_list);
            EHVI_list=EHVI_list.*crowd_list;
            
            [~,index_list]=sort(EHVI_list,'descend');
            feasible_index_list=feasible_index_list(index_list);
            EHVI_list=EHVI_list(index_list,:);
            
            % EHVI only use large than 0
            index_list=find(EHVI_list > 0);
            if length(index_list) > sample_number_iteration
                x_updata_list=[x_updata_list;x_particle_list(feasible_index_list(...
                    index_list(1:sample_number_iteration)),:)];
            else
                x_updata_list=[x_updata_list;x_particle_list(feasible_index_list(...
                    index_list),:)];
            end
        else
            % calculate CI
            % instead of using predict function
            CI_list=ones(length(infeasible_index_list),1);
            for x_index=1:length(infeasible_index_list)
                if con_number > 0
                    for con_index=1:con_number
                        CI_list(x_index)=CI_list(x_index)*normcdf(-(con_predict_list(infeasible_index_list(x_index),con_index))...
                            /sqrt(con_predict_var_list(infeasible_index_list(x_index),con_index)));
                    end
                end
            end
            
            % select first sample number to add in data
            [~,index_list]=sort(CI_list,'descend');
            infeasible_index_list=infeasible_index_list(index_list);
            CI_list=CI_list(index_list,:);
            
            % CI use all
            if length(infeasible_index_list) > sample_number_iteration
                x_updata_list=[x_updata_list;x_particle_list(infeasible_index_list(...
                    index_list(1:sample_number_iteration)),:)];
            else
                x_updata_list=[x_updata_list;x_particle_list(infeasible_index_list,:);];
            end
        end
        
    else
        [EHVI_list,fval_particle_list]=calHVLCB...
            (x_particle_list,object_function_surrogate,object_function_variance,...
            fval_number,fval_pareto,2);
        
        % step 10 evaluate particle crowded to make particl more uniform distribution
        crowd_list=calParetoCrowd(fval_particle_list,fval_pareto);
        crowd_list=crowd_list/sum(crowd_list);
        EHVI_list=EHVI_list.*crowd_list;
        
        % EHVI only use large than 0
        index_list=find(EHVI_list > 0);
        x_updata_list=x_particle_list(index_list,:);
        EHVI_list=EHVI_list(index_list,:);
        
        % select first sample_number_iteration sample number to add in data
        [~,index_list]=sort(EHVI_list,'descend');
        if length(index_list) > sample_number_iteration
            x_updata_list=x_updata_list(index_list(1:sample_number_iteration),:);
        else
            x_updata_list=x_updata_list(index_list,:);
        end
    end
    
    EHVI_function=@(x_particle_list) -calEHVI...
        (x_particle_list,object_function_surrogate,object_function_variance,...
        nonlcon_function_surrogate,nonlcon_function_variance,...
        fval_number,con_number,coneq_number,fval_pareto);
    
    HVLCB_function=@(x_particle_list) -calHVLCB...
        (x_particle_list,object_function_surrogate,object_function_variance,...
        fval_number,fval_pareto,2);
    
    CI_function=@(x_particle_list) -calCI...
        (x_particle_list,nonlcon_function_surrogate,nonlcon_function_variance,...
        con_number,coneq_number,con_nomlz_list,coneq_nomlz_list);
    
    % local search
    x_initial_fmincon=global_best_list(randi(global_number),:);
    totalcon_function=@(x) totalconFunction(x,nonlcon_function_surrogate,cheapcon_function);
        fmincon_option=optimoptions('fmincon','FunctionTolerance',1e-2,...
            'display','none','FiniteDifferenceStepSize',1e-5,'Algorithm','sqp');
    [x_LCB,HVLCB_LCB,exit_flag,~]=fmincon...
        (HVLCB_function,x_initial_fmincon,[],[],[],[],low_bou,up_bou,totalcon_function,fmincon_option);
    if ~isempty(cheapcon_function) && cheapcon_function(x_LCB) > 0
        x_LCB=[];
    end
    
    x_updata_list=[x_updata_list;x_LCB];
    
    if DRAW_FIGURE_FLAG
        classifySupportVectorMachineVisualization...
            (SVM_model,low_bou,up_bou);
        line(x_particle_list(:,1),x_particle_list(:,2),...
            'Marker','o','LineStyle','none','Color','green');
        line(x_updata_list(:,1),x_updata_list(:,2),...
            'Marker','o','LineStyle','none','Color','black');
    end
    
    if DRAW_PARETO_FLAG
        figure(2)
        [~,index_list]=sort(fval_pareto(:,1));
        fval_pareto_draw=fval_pareto(index_list,:);
        line(fval_pareto_draw(:,1),fval_pareto_draw(:,2),'Marker','o');
        drawnow;
    end
    %     drawFunction(CI_function,low_bou,up_bou)
    %     drawFunction(HVLCB_function,low_bou,up_bou)
    %     drawFunction(EHVI_function,low_bou,up_bou)
    
%     if mod((iteration-1),20) == 0
%         save(['data_',num2str((iteration-1)),'.mat']);
%     end
end
result_pareto=result_pareto(1:(iteration-1));

[~,index_list]=sort(fval_pareto(:,1));
fval_pareto=fval_pareto(index_list,:);
x_pareto=x_pareto(index_list,:);
output.result_pareto=result_pareto;

    function [fval,con,coneq]=modelFunction...
            (x,object_function,nonlcon_function)
        % model function, concertrate fval, con, coneq into one function
        %
        if nargin < 3 || isempty(nonlcon_function)
            con=[];
            coneq=[];
        else
            [con,coneq]=nonlcon_function(x);
        end
        fval=object_function(x);
    end
    function [con,coneq]=totalconFunction(x,nonlcon_function,cheapcon_function)
        con=[];
        coneq=[];
        if ~isempty(nonlcon_function)
            [expencon,expenconeq]=nonlcon_function(x);
            con=[con;expencon];
            coneq=[coneq;expenconeq];
        end
        if ~isempty(cheapcon_function)
            [expencon,expenconeq]=cheapcon_function(x);
            con=[con;expencon];
            coneq=[coneq;expenconeq];
        end
    end

    function [x_updata_list,fval_updata_list,con_updata_list,coneq_updata_list,NFE_updata]=dataLibraryUpdataProtect...
            (data_library_name,model_function,x_add_list,...
            x_list,fval_list,con_list,coneq_list,...
            low_bou,up_bou,protect_range)
        % function updata data with same_point_avoid protect
        % return fval
        % all list is x_number x variable_number matrix
        %
        variable_number__=size(x_list,2);
        NFE_updata=0;
        x_updata_list=[];fval_updata_list=[];con_updata_list=[];coneq_updata_list=[];
        for x_index__=1:size(x_add_list,1)
            x_updata__=x_add_list(x_index__,:);
            
            % check x_potential if exist in data library
            % if not, updata data libraray
            distance__=sum(((x_updata__-x_list)./(low_bou-up_bou)).^2,2);
            [~,min_index__]=min(distance__);
            if distance__(min_index__) < variable_number__*protect_range^2
                x_updata__=x_list(min_index__,:);
                fval_updata__=fval_list(min_index__,:);
                con_updata__=[];
                coneq_updata__=[];
                if ~isempty(con_list)
                    con_updata__=(con_list(min_index__,:));
                end
                if ~isempty(coneq_list)
                    coneq_updata__=(coneq_list(min_index__,:));
                end
            else
                [fval_updata__,con_updata__,coneq_updata__]=dataLibraryUpdata...
                    (data_library_name,model_function,x_updata__);NFE_updata=NFE_updata+1;
                x_updata_list=[x_updata_list;x_updata__];
                fval_updata_list=[fval_updata_list;fval_updata__];
                con_updata_list=[con_updata_list;con_updata__];
                coneq_updata_list=[coneq_updata_list;coneq_updata__];
            end
        end
    end

    function X_new=differEvolutionCrossover(low_bou,up_bou,X,V,C_R)
        [x_number__,variable_number__]=size(X);
        X_new=X;
        for x_index__=1:x_number__
            rand_number=rand(1,variable_number__);
            v_index__=randi(size(V,1));
            index__=find(rand_number < C_R);
            X_new(x_index__,index__)=V(v_index__,index__);
            X_new(x_index__,:)=max(X_new(x_index__,:),low_bou);
            X_new(x_index__,:)=min(X_new(x_index__,:),up_bou);
        end
    end
    function X_new=differEvolutionMutation(low_bou,up_bou,X,C_M)
        [x_number__,variable_number__]=size(X);
        X_new=X;
        for x_index__=1:x_number__
            rand_number=rand(1,variable_number__);
            index__=find(rand_number < C_M);
            X_new(x_index__,index__)=rand(1,length(index__)).*...
                (up_bou(index__)-low_bou(index__))+low_bou(index__);
            X_new(x_index__,:)=max(X_new(x_index__,:),low_bou);
            X_new(x_index__,:)=min(X_new(x_index__,:),up_bou);
        end
    end
end

%% pareto front
function [pareto_index_list,feasible_index_list]=getParetoFront...
    (fval_nomlz_list,con_nomlz_list,coneq_nomlz_list,...
    pareto_torlance)
% distinguish pareto front of exist point
% use initial pareto definition(contrain compare first and fval compare)
%
if nargin < 4
    pareto_torlance=0;
end

pareto_index_list=[];% index of filter point list
feasible_index_list=[];% feasible point list

% select no domain filter
for x_index=1:size(fval_nomlz_list,1)
    con_nomlz=[];
    if ~isempty(con_nomlz_list)
        con_nomlz=max(con_nomlz_list(x_index,:));
    end
    coneq_nomlz=[];
    if ~isempty(coneq_nomlz_list)
        coneq_nomlz=max(abs(coneq_nomlz_list(x_index,:)));
    end
    con_nomlz_max=max([con_nomlz;coneq_nomlz;0]); % con is large than 0 or 0
    
    if (con_nomlz_max <= pareto_torlance)
        feasible_index_list=[feasible_index_list;x_index];
    end
    
    % notice con_nomlz_max is greater than or equal to zero
    % notice con_nomlz_pareto_max is greater than or equal to zero
    pareto_index=1;
    add_filter_flag=1;
    while pareto_index <= length(pareto_index_list)
        % compare x with exit pareto front point
        x_pareto_index=pareto_index_list(pareto_index,:);
        
        % contain constraint of x_filter
        con_pareto_nomlz=[];
        if ~isempty(con_nomlz_list)
            con_pareto_nomlz=max(con_nomlz_list(x_pareto_index,:));
        end
        coneq_pareto_nomlz=[];
        if ~isempty(coneq_nomlz_list)
            coneq_pareto_nomlz=max(coneq_nomlz_list(x_pareto_index,:));
        end
        con_nomlz_pareto_max=max([con_pareto_nomlz;coneq_pareto_nomlz;0]);
        
        % compare x with x_pareto
        % if this x is domain by pareto point, reject it
        if con_nomlz_max > con_nomlz_pareto_max
            % x constrain is large than pareto
            add_filter_flag=0;
            break;
        elseif con_nomlz_max <= pareto_torlance && con_nomlz_pareto_max <= pareto_torlance
            % both is feasiable point
            judge=fval_nomlz_list(x_index,:)+pareto_torlance >= fval_nomlz_list(x_pareto_index,:);
            if ~sum(~judge)
                add_filter_flag=0;
                break;
            end
        end
        
        % if better or equal than exit pareto point, reject pareto point
        if con_nomlz_pareto_max > con_nomlz_max
            % pareto constrain is large than x
            pareto_index_list(pareto_index)=[];
            pareto_index=pareto_index-1;
        elseif con_nomlz_max <= pareto_torlance && con_nomlz_pareto_max <= pareto_torlance
            judge=fval_nomlz_list(x_index,:) <= fval_nomlz_list(x_pareto_index,:)+pareto_torlance;
            if ~sum(~judge)
                pareto_index_list(pareto_index)=[];
                pareto_index=pareto_index-1;
            end
        end
        
        pareto_index=pareto_index+1;
    end
    
    % add into pareto list if possible
    if add_filter_flag
        pareto_index_list=[pareto_index_list;x_index];
    end
end
end

function [pareto_fitness_list,pareto_fval_list]=calParetoFitness...
    (x_list,object_function_surrogate,object_function_variance,...
    nonlcon_function_surrogate,nonlcon_function_variance,...
    fval_number,con_number,coneq_number,fval_min,fval_pareto)
% calculate
%
[x_number,variable_number]=size(x_list);

% evaluate all predict fval
pareto_fval_list=zeros(x_number,fval_number);
fval_var_list=zeros(x_number,fval_number);
if con_number > 0
    con_list=zeros(x_number,con_number);
    con_var_list=zeros(x_number,con_number);
end
if coneq_number > 0
    coneq_list=zeros(x_number,coneq_number);
    coneq_var_list=zeros(x_number,coneq_number);
end
for x_index=1:x_number
    x=x_list(x_index,:);
    pareto_fval_list(x_index,:)=object_function_surrogate(x)';
    fval_var_list(x_index,:)=object_function_variance(x)';
    if ~isempty(nonlcon_function_surrogate)
        [con,coneq]=nonlcon_function_surrogate(x);
        [con_var,coneq_var]=nonlcon_function_variance(x);
        if con_number > 0
            con_list(x_index,:)=con;
            con_var_list(x_index,:)=con_var;
        end
        if coneq_number > 0
            coneq_list(x_index,:)=coneq;
            coneq_var_list(x_index,:)=coneq_var;
        end
    end
end

LCB_list=pareto_fval_list-fval_var_list;
fval_nomlz_list=(fval_min-pareto_fval_list)./sqrt(fval_var_list);
EI_list=(fval_min-pareto_fval_list).*normcdf(fval_nomlz_list)+...
    pareto_fval_list.*normpdf(fval_nomlz_list);
if con_number > 0
    con_max=max(min(con_list,[],1),0);
    CI_list=sum(...
        normpdf((con_max-con_list)./con_var_list),2);
end
if coneq_number > 0
    %
end

% calculate G
% instead of using predict function
% constraints use
G_list=zeros(x_number,1);
for x_index=1:x_number
    % G(x)=1-max(min(fval_i-fval_j,con_i-con_j,coneq_i-coneq_j))
    
    min_HV=min(LCB_list(x_index,:)-fval_pareto,[],2);
    if con_number > 0 && con_list(x_index) < 0
        min_HV=min(min_HV,min(CI_list(x_index,:)-CI_list(search_index,:),[],2),2);
    end
    if coneq_number > 0
        
    end
    G_list(x_index)=1-max(min_HV);
end

% pareto fitness
G_max=max(G_list);
pareto_fitness_list=zeros(x_number,1);
for x_index=1:x_number
    if G_list(x_index) < 1
        pareto_fitness_list(x_index)=...
            (1-G_list(x_index))+2*G_max;
    else
        pareto_fitness_list(x_index)=G_list(x_index);
    end
end

end

%% multi-objective indicator
function crowd_list=calParetoCrowd(fval_list,fval_exit_list)
% calculate crowded the same as NSGA-II
%

[x_number,variable_number]=size(fval_list);
% calculate crowd
index_list=sort(fval_list(:,1));
crowd_list=zeros(x_number,1);

for x_index=1:x_number
    search_index=1:x_number;
    search_index(x_index)=[];
    
    fval_distance=min(abs(fval_list(x_index,:)-[fval_list(search_index,:);fval_exit_list]),[],1);
    
    crowd_list(x_index)=sum(fval_distance)/variable_number;
end

end
function [EHVI_list,fval_list]=calEHVI...
    (x_list,object_function_surrogate,object_function_variance,...
    nonlcon_function_surrogate,nonlcon_function_variance,...
    fval_number,con_number,coneq_number,fval_pareto)
% calculate 
%
[x_number,variable_number]=size(x_list);

% evaluate all predict fval
fval_list=zeros(x_number,fval_number);
fval_var_list=zeros(x_number,fval_number);
if con_number > 0
    con_list=zeros(x_number,con_number);
    con_var_list=zeros(x_number,con_number);
end
if coneq_number > 0
    coneq_list=zeros(x_number,coneq_number);
    coneq_var_list=zeros(x_number,coneq_number);
end
for x_index=1:x_number
    x=x_list(x_index,:);
    fval_list(x_index,:)=object_function_surrogate(x);
    fval_var_list(x_index,:)=object_function_variance(x);
    if ~isempty(nonlcon_function_surrogate)
        [con,coneq]=nonlcon_function_surrogate(x);
        [con_var,coneq_var]=nonlcon_function_variance(x);
        if con_number > 0
            con_list(x_index,:)=con;
            con_var_list(x_index,:)=con_var;
        end
        if coneq_number > 0
            coneq_list(x_index,:)=coneq;
            coneq_var_list(x_index,:)=coneq_var;
        end
    end
end

% calculate EHVI
% instead of using predict function
% constraints use
EHVI_list=zeros(x_number,1);
[~,index_list]=sort(fval_pareto(:,1));
fval_pareto=fval_pareto(index_list,:);
for x_index=1:x_number
    aver1=fval_list(x_index,1);
    svar1=sqrt(fval_var_list(x_index,1));
    aver2=fval_list(x_index,2);
    svar2=sqrt(fval_var_list(x_index,2));
    
    if size(fval_pareto,1) > 1
        EHVI=0;
        for j_index=1:size(fval_pareto,1)-2
            nomlz_b_j1=(fval_pareto(j_index+1,2)-aver2)/svar2;
            nomlz_b_j=(fval_pareto(j_index,2)-aver2)/svar2;
%             for i_index=1:j_index-1
                nomlz_a_i1=(fval_pareto((1:j_index-1)+1,1)-aver1)/svar1;
                EHVI=EHVI+...
                    sum((svar1*normpdf(nomlz_a_i1)+(fval_pareto((1:j_index-1)+1,1)-aver1).*normcdf(nomlz_a_i1)).*...
                    (fval_pareto((1:j_index-1),2)-fval_pareto((1:j_index-1)+1,2))*...
                    (normcdf(nomlz_b_j)-normcdf(nomlz_b_j1)));
%             end
            nomlz_a_j1=(fval_pareto(j_index+1,1)-aver1)/svar1;
            EHVI=EHVI+...
                (svar1*normpdf(nomlz_a_j1)+(fval_pareto(j_index+1,1)-aver1).*normcdf(nomlz_a_j1))*...
                (fval_pareto(j_index,2)-fval_pareto(j_index+1,2))*normcdf(nomlz_b_j);
        end
        j_index=size(fval_pareto,1)-1;
        nomlz_b_j=(fval_pareto(j_index,2)-aver2)/svar2;
%         for i_index=1:j_index-1
            nomlz_a_i1=(fval_pareto((1:j_index-1)+1,1)-aver1)/svar1;
            EHVI=EHVI+...
                sum((svar1*normpdf(nomlz_a_i1)+(fval_pareto((1:j_index-1)+1,1)-aver1).*normcdf(nomlz_a_i1)).*...
                (fval_pareto((1:j_index-1),2)-fval_pareto((1:j_index-1)+1,2))*...
                (normcdf(nomlz_b_j)));
%         end
        nomlz_a_j1=(fval_pareto(j_index+1,1)-aver1)/svar1;
        EHVI=EHVI+...
            (svar1*normpdf(nomlz_a_j1)+(fval_pareto(j_index+1,1)-aver1).*normcdf(nomlz_a_j1))*...
            (svar2*normpdf(nomlz_b_j)+(fval_pareto(j_index,2)-aver2)*normcdf(nomlz_b_j));
    else
        nomlz_a_j=(fval_pareto(1,1)-aver1)/svar1;
        nomlz_b_j=(fval_pareto(1,2)-aver2)/svar2;
        EHVI=(svar1*normpdf(nomlz_a_j)+(fval_pareto(1,1)-aver1)*normcdf(nomlz_a_j))*...
            (svar2*normpdf(nomlz_b_j)+(fval_pareto(1,2)-aver2)*normcdf(nomlz_b_j));
    end
    if con_number > 0
        for con_index=1:con_number
            EHVI=EHVI*normcdf((-con_list(con_index))/sqrt(con_var_list(con_index)));
        end
    end
    EHVI_list(x_index)=EHVI;
end

end
function [HVLCB_list,fval_list,fval_var_list]=calHVLCB...
    (x_list,object_function_surrogate,object_function_variance,...
    fval_number,fval_pareto,weight)
% calculate hyoervolume LCB
%
[x_number,~]=size(x_list);

% evaluate all predict fval
fval_list=zeros(x_number,fval_number);
fval_var_list=zeros(x_number,fval_number);
for x_index=1:x_number
    x=x_list(x_index,:);
    fval_list(x_index,:)=object_function_surrogate(x);
    fval_var_list(x_index,:)=object_function_variance(x);
end

% calculate HVLCB
% instead of using predict function
HVLCB_list=zeros(x_number,1);
[~,index_list]=sort(fval_pareto(:,1));
fval_pareto=fval_pareto(index_list,:);
for x_index=1:x_number
    aver1=fval_list(x_index,1);
    stdD1=sqrt(fval_var_list(x_index,1));
    aver2=fval_list(x_index,2);
    stdD2=sqrt(fval_var_list(x_index,2));
    
    llb1=aver1-weight*stdD1;
    llb2=aver2-weight*stdD2;
    
    if size(fval_pareto,1) > 1
        HVLCB=0;
        if llb1 < fval_pareto(1,1) && llb2 > fval_pareto(1,2)
            HVLCB=(fval_pareto(1,1)-llb1)*(llb2-fval_pareto(1,2));
        elseif llb1 > fval_pareto(end,1) && llb2 < fval_pareto(end,2)
            HVLCB=(llb1-fval_pareto(end,1))*(fval_pareto(end,2)-llb2);
        else
            for i_index=1:size(fval_pareto,1)-1
                if llb1 >= fval_pareto(i_index+1,1) ||...
                        llb2 >= fval_pareto(i_index,2)
                elseif fval_pareto(i_index+1,2) < llb2
                    HVLCB=HVLCB+...
                        (fval_pareto(i_index+1,1)-llb1)*...
                        (fval_pareto(i_index,2)-llb2);
                else
                    HVLCB=HVLCB+...
                        (fval_pareto(i_index+1,1)-llb1)*...
                        (fval_pareto(i_index,2)-fval_pareto(i_index+1,2));
                end
            end
        end
    else
        if (fval_pareto(1,1) < llb1) && (fval_pareto(1,2) < llb2)
            HVLCB=0;
        else
            HVLCB=abs(fval_pareto(1,1)-llb1)*abs(fval_pareto(1,2)-llb2);
        end
    end
    HVLCB_list(x_index)=HVLCB;
end

end
function [CI_list,con_list,con_var_list,coneq_list,coneq_var_list]=calCI...
    (x_list,nonlcon_function_surrogate,nonlcon_function_variance,...
    con_number,coneq_number,con_nomlz_list,coneq_nomlz_list)
% calculate constrain imporve probility
%
[x_number,~]=size(x_list);

% evaluate all predict con, coneq
if con_number > 0
    con_list=zeros(x_number,con_number);
    con_var_list=zeros(x_number,con_number);
end
if coneq_number > 0
    coneq_list=zeros(x_number,coneq_number);
    coneq_var_list=zeros(x_number,coneq_number);
end
for x_index=1:x_number
    x=x_list(x_index,:);
    if ~isempty(nonlcon_function_surrogate)
        [con,coneq]=nonlcon_function_surrogate(x);
        [con_var,coneq_var]=nonlcon_function_variance(x);
        if con_number > 0
            con_list(x_index,:)=con;
            con_var_list(x_index,:)=con_var;
        end
        if coneq_number > 0
            coneq_list(x_index,:)=coneq;
            coneq_var_list(x_index,:)=coneq_var;
        end
    end
end

% calculate CI
% instead of using predict function
CI_list=zeros(x_number,1);
for x_index=1:x_number
    CI=1;
    if con_number > 0
        for con_index=1:con_number
            CI=CI*normcdf(-(con_list(con_index))/sqrt(con_var_list(con_index)));
        end
    end
    CI_list(x_index)=CI;
end

end

%% auxiliary function
function [x_best,fval_best,con_best,coneq_best]=findMinRaw...
    (x_list,fval_list,con_list,coneq_list,...
    cheapcon_function,nonlcon_torlance)
% find min fval in raw data
% x_list, rank is variable
% con_list, rank is con
% coneq_list, rank is coneq
% function will find min fval in con==0
% if there was not feasible x, find min consum
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
        [con,coneq]=cheapcon_function(x_list(x_index,:)');
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
    x_best=x_list(index_best,:)';
    if ~isempty(con_list)
        con_best=con_list(index_best,:)';
    end
    if ~isempty(coneq_list)
        coneq_best=coneq_list(index_best,:)';
    end
else
    % min consum
    [~,index_best]=min(max_nonlcon_list);
    fval_best=fval_list(index_best);
    x_best=x_list(index_best,:)';
    if ~isempty(con_list)
        con_best=con_list(index_best,:)';
    end
    if ~isempty(coneq_list)
        coneq_best=coneq_list(index_best,:)';
    end
end
end
function [x_list,fval_list,con_list,coneq_list]=rankData...
    (x_list,fval_list,con_list,coneq_list,...
    cheapcon_function,nonlcon_torlance)
% rank data base on feasibility rule
% infeasible is rank by sum of constraint
%
if nargin < 6 || isempty(nonlcon_torlance)
    nonlcon_torlance=0;
end
if nargin < 5
    cheapcon_function=[];
end

[x_number,~]=size(x_list);
con_sum_list=zeros(x_number,1);
for x_index=1:x_number
    if ~isempty(con_list)
        con_sum_list(x_index)=con_sum_list(x_index)+sum(max(con_list(x_index,:)-nonlcon_torlance,0));
    end
    if ~isempty(coneq_list)
        con_sum_list(x_index)=con_sum_list(x_index)+sum(abs(coneq_list(x_index,:))-nonlcon_torlance);
    end
end

% add cheap con
for x_index=1:size(x_list,1)
    if ~isempty(cheapcon_function)
        [con,coneq]=cheapcon_function(x_list(x_index,:)');
        con_sum_list(x_index)=con_sum_list(x_index)+...
            sum(max(con,0))+sum(max(abs(coneq),0));
    end
end

% get index_list to rank x by rule
index_list=1; % ranked index of x_list
if con_sum_list(1)==0
    infeasible_index=2;
else
    infeasible_index=1;
end
for x_index=2:x_number
    if con_sum_list(x_index)==0
        % local data
        local_index=1;
        while local_index <= (infeasible_index-1) &&...
                fval_list(x_index) > fval_list(index_list(local_index))
            local_index=local_index+1;
        end
        index_list=[index_list(1:local_index-1,:);...
            x_index;index_list(local_index:end,:)];
        infeasible_index=infeasible_index+1;
    else
        % local data
        local_index=infeasible_index;
        while local_index <= size(index_list,1) &&...
                con_sum_list(x_index) > con_sum_list(index_list(local_index))
            local_index=local_index+1;
        end
        index_list=[index_list(1:local_index-1,:);...
            x_index;index_list(local_index:end,:)];
    end
end

% rank by index_list
x_list=x_list(index_list,:);
fval_list=fval_list(index_list);
if ~isempty(con_list)
    con_list=con_list(index_list,:);
end
if ~isempty(coneq_list)
    coneq_list=coneq_list(index_list,:);
end
end

%% surrogate model
function [kriging_model_fval,kriging_model_con,kriging_model_coneq,output]=getKrigingModel...
    (x_list,fval_list,con_list,coneq_list,...
    kriging_model_fval,kriging_model_con,kriging_model_coneq)
% base on library_data to create kriging model and function
% if input model, function will updata model
% object_function is multi fval output
% nonlcon_function is normal nonlcon_function which include con, coneq
% con is colume vector, coneq is colume vector
% var_function is same
%
if size(x_list,1) ~= size(fval_list,1)
    error('getKrigingModel: x_list size no equal fval_list size')
end

predict_function_fval=cell(size(fval_list,2),1);
if isempty(kriging_model_fval)
    kriging_model_fval=struct('X',[],'Y',[],...
        'fval_regression',[],'covariance',[],'inv_covariance',[],...
        'hyp',[],'beta',[],'gama',[],'sigma_sq',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],...
        'predict_function',[]);
    kriging_model_fval=repmat(kriging_model_fval,1,[size(fval_list,2)]);
    for fval_index=1:size(fval_list,2)
        [predict_function_fval{fval_index},kriging_model_fval(fval_index)]=interpKrigingPreModel...
            (x_list,fval_list(:,fval_index));
    end
else
    for fval_index=1:size(fval_list,2)
        [predict_function_fval{fval_index},kriging_model_fval(fval_index)]=interpKrigingPreModel...
            (x_list,fval_list(:,fval_index),kriging_model_fval(fval_index).hyp);
    end
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
        fval=zeros(size(X_predict,1),length(predict_function_fval));
        fval_var=zeros(size(X_predict,1),length(predict_function_fval));
        for fval_index__=1:length(predict_function_fval)
            [fval(:,fval_index__),fval_var(:,fval_index__)]=...
                predict_function_fval{fval_index__}(X_predict);
        end
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
% prepare model, optimal theta and calculation parameter
% X, Y are x_number x variable_number matrix
% aver_X,stdD_X is 1 x x_number matrix
% theta beta gama sigma_sq is normalizede, so predict y is normalize
% theta=exp(hyp)
%
% input initial data X, Y, which are real data
%
% output is a kriging model, include predict_function...
% X, Y, base_function_list
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
if  ~isempty(index__),  stdD_X(index__)=1; end
index__=find(stdD_Y == 0);
if  ~isempty(index__),  stdD_Y(index__)=1; end
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
fval_reg_nomlz=(reg_function(X)-0)./1;

% optimal to get hyperparameter
fmincon_option=optimoptions('fmincon','Display','none',...
    'OptimalityTolerance',1e-2,...
    'FiniteDifferenceStepSize',1e-5,...,
    'MaxIterations',10,'SpecifyObjectiveGradient',false);
low_bou_hyp=-4*ones(1,variable_number);
up_bou_hyp=4*ones(1,variable_number);
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
% num: number, pred: predict, vari: variable, hyp: hyper parameter
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
        fval_reg_pred_nomlz=(fval_reg_pred-0)./1;
        
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

%% FCM
function FC_model=classifyFuzzyClustering...
    (X,classify_number,low_bou,up_bou,m)
% get fuzzy cluster model
% X is x_number x variable_number matrix
% center_list is classify_number x variable_number matrix
%
if nargin < 4
    up_bou=[];
    if nargin < 3
        low_bou=[];
    end
end
iteration_max=100;
torlance=1e-3;

[x_number,variable_number]=size(X);

% nomalization data
if isempty(low_bou)
    low_bou=min(X,[],1);
else
    low_bou=low_bou(:)';
end
if isempty(up_bou)
    up_bou=max(X,[],1);
else
    up_bou=up_bou(:)';
end
X_nomlz=(X-low_bou)./(up_bou-low_bou);

% if x_number equal 1, clustering cannot done
if x_number==1
    FC_model.X=X;
    FC_model.X_normalize=X_nomlz;
    FC_model.center_list=X;
    FC_model.fval_loss_list=[];
    FC_model.x_class_list=ones(x_number,1);
    return;
end

U=zeros(x_number,classify_number);
center_list=rand(classify_number,variable_number)*0.5;
iteration=0;
done=0;
fval_loss_list=zeros(iteration_max,1);

% get X_center_dis_sq and classify x to each x center
% classify_number x x_number matrix
X_center_dis_sq=zeros(x_number,classify_number);
x_class_list=zeros(x_number,1);
for x_index=1:x_number
    for classify_index=1:classify_number
        temp=(X_nomlz(x_index,:)-center_list(classify_index,:));
        X_center_dis_sq(x_index,classify_index)=temp*temp';
    end
    [~,x_class_list(x_index)]=min(X_center_dis_sq(x_index,:));
end

while ~done
    % updata classify matrix U
    % classify matrix U is x weigth of each center
    for classify_index=1:classify_number
        for x_index=1:x_number
            U(x_index,classify_index)=...
                1/sum((X_center_dis_sq(x_index,classify_index)./X_center_dis_sq(x_index,:)).^(1/(m-1)));
        end
    end
    
    % updata center_list
    center_list_old=center_list;
    for classify_index=1:classify_number
        center_list(classify_index,:)=...
            sum((U(:,classify_index)).^m.*X_nomlz,1)./...
            sum((U(:,classify_index)).^m,1);
    end
    
    % updata X_center_dis_sq
    for x_index=1:x_number
        for classify_index=1:classify_number
            temp=(X_nomlz(x_index,:)-center_list(classify_index,:));
            X_center_dis_sq(x_index,classify_index)=temp*temp';
        end
        [~,x_class_list(x_index)]=min(X_center_dis_sq(x_index,:));
    end
    
%     plot(center_list(:,1),center_list(:,2));
    
    % forced interrupt
    if iteration > iteration_max
        done=1;
    end
    
    % convergence judgment
    if sum(sum(center_list_old-center_list).^2) < torlance
        done=1;
    end
    
    iteration=iteration+1;
    fval_loss_list(iteration)=sum(sum(U.^m.*X_center_dis_sq));
end
fval_loss_list(iteration+1:end)=[];
center_list=center_list.*(up_bou-low_bou)+low_bou;

FC_model.X=X;
FC_model.X_normalize=X_nomlz;
FC_model.center_list=center_list;
FC_model.fval_loss_list=fval_loss_list;
FC_model.x_class_list=x_class_list;

end

%% SVM
function [predict_function,SVM_model]=classifySupportVectorMachine...
    (X,class,C,low_bou,up_bou,kernel_function)
% generate support vector machine model version 0
% version 0 use fmincon to get alpha
% only support binary classification, 0 and 1
% X, Y is x_number x variable_number matrix
% C is penalty factor, default is empty
% kernel_function default is gauss kernal function
% kernel_function should be @(x1,x2) ...
%
if nargin < 6
    kernel_function=[];
    if nargin < 3
        C=[];
    end
end

[x_number,variable_number]=size(X);

% normalization data
if nargin < 5
    up_bou=max(X);
    if nargin < 4
        low_bou=min(X);
    end
end
X_nomlz=(X-low_bou)./(up_bou-low_bou);

% transfer class into y
Y=class;
for x_index=1:x_number
    if Y(x_index) == 0
        Y(x_index)=-1;
    end
end

% default kernal function
if isempty(kernel_function)
    sigma=-100*log(1/sqrt(x_number))/variable_number^2;
    kernel_function=@(x1,x2) exp(-((x1-x2)'*(x1-x2))*sigma);
end

% initialization kernal function process X_cov
X_cov=zeros(x_number);
for rank_index=1:x_number
    for colume_index=1:rank_index-1
        X_cov(rank_index,colume_index)=...
            X_cov(colume_index,rank_index);
    end
    for colume_index=rank_index:x_number
        X_cov(rank_index,colume_index)=...
            kernel_function(X_nomlz(rank_index,:)',X_nomlz(colume_index,:)');
    end
end

% min SVM object function to get alpha
object_function_SVM=@(alpha) -objectFunctionSVM(alpha,X_cov,Y);
alpha_initial=ones(x_number,1)*0.5;
low_bou_fmincon=0*ones(x_number,1);
if isempty(C) || C==0
    up_bou_fmincon=[];
else
    up_bou_fmincon=C*ones(x_number,1);
end
Aeq=Y';
fmincon_options=optimoptions('fmincon','Display','none','Algorithm','sqp');
alpha=fmincon(object_function_SVM,alpha_initial,...
    [],[],Aeq,0,low_bou_fmincon,up_bou_fmincon,[],fmincon_options);

% obtain other paramter
w=sum(alpha.*Y.*X_nomlz);
index_list=find(alpha > 1e-6);

alpha_Y=alpha.*Y;
alpha_Y_cov=X_cov*alpha_Y;
b=sum(Y(index_list)-alpha_Y_cov(index_list))/length(index_list);

% generate predict function
predict_function=@(x) classifySupportVectorMachinePredictor...
    (x,X_nomlz,Y,alpha,b,low_bou,up_bou,kernel_function);

% output model
SVM_model.X=X;
SVM_model.class=class;
SVM_model.Y=Y;
SVM_model.X_nomlz=X_nomlz;
SVM_model.low_bou=low_bou;
SVM_model.up_bou=up_bou;
SVM_model.alpha=alpha;
SVM_model.w=w;
SVM_model.b=b;
SVM_model.kernel_function=kernel_function;
SVM_model.predict_function=predict_function;

    function fval=objectFunctionSVM(alpha,X_inner_product,Y)
        % support vector machine maximum object function
        %
        alpha=alpha(:);
        alpha_Y__=alpha.*Y;
        fval=sum(alpha)-alpha_Y__'*X_inner_product*alpha_Y__/2;
    end
    function [predict_class,predict_fval]=classifySupportVectorMachinePredictor...
            (x,X_nomlz,Y,alpha,b,low_bou,up_bou,kernel_function)
        % predict value of x is 1 or -1
        % x input is colume vector
        %
        x=x(:);
        
        x_number__=size(X_nomlz,1);
        x_nomlz=(x-low_bou')./(up_bou'-low_bou');
        X_inner_product__=zeros(x_number__,1);
        for x_index__=1:x_number__
            X_inner_product__(x_index__)=...
                kernel_function(X_nomlz(x_index__,:)',x_nomlz);
        end
        predict_fval=sum(alpha.*Y.*X_inner_product__)+b;
        predict_fval=1/(1+exp(-predict_fval));
        if predict_fval > 0.5
            predict_class=1;
        else
            predict_class=0;
        end
    end
end

%% data library
function [fval_list,con_list,coneq_list]=dataLibraryUpdata...
    (data_library_name,model_function,x_list)
% updata data library
% updata format:
% variable_number, fval_number, con_number, coneq_number
% x, fval, con, coneq
%
fval_list=[];
con_list=[];
coneq_list=[];
[x_number,variable_number]=size(x_list);

if ~strcmp(data_library_name(end-3:end),'.txt')
    data_library_name=[data_library_name,'.txt'];
end

% store format
x_format_base='%.8e ';
fval_format_base='%.8e ';
x_format=repmat(x_format_base,1,variable_number);
file_optimalSurrogate_output=fopen(data_library_name,'a');
file_result=fopen('result_total.txt','a');

% updata format:
% variable_number, fval_number, con_number, coneq_number
% x, fval, con, coneq
for x_index=1:x_number
    x=x_list(x_index,:);
    [fval,con,coneq]=model_function(x);
    fval=fval(:);
    con=con(:);
    coneq=coneq(:);
    fval_list=[fval_list;fval(:)'];
    con_list=[con_list;con(:)'];
    coneq_list=[coneq_list;coneq(:)'];
    
    % write data to txt_optimalSurrogateSADEKTS
    fprintf(file_optimalSurrogate_output,'%d ',variable_number);
    fprintf(file_optimalSurrogate_output,'%d ',length(fval));
    fprintf(file_optimalSurrogate_output,'%d ',length(con));
    fprintf(file_optimalSurrogate_output,'%d ',length(coneq));
    
    fprintf(file_optimalSurrogate_output,x_format,x);
    fval_format=repmat(fval_format_base,1,length(fval));
    fprintf(file_optimalSurrogate_output,fval_format,fval);
    fval_format=repmat(fval_format_base,1,length(con));
    fprintf(file_optimalSurrogate_output,fval_format,con);
    fval_format=repmat(fval_format_base,1,length(coneq));
    fprintf(file_optimalSurrogate_output,fval_format,coneq);
    fprintf(file_optimalSurrogate_output,'\n');
    
    % write data to txt_result
    fprintf(file_result,'%d ',variable_number);
    fprintf(file_result,'%d ',length(fval));
    fprintf(file_result,'%d ',length(con));
    fprintf(file_result,'%d ',length(coneq));
    
    fprintf(file_result,x_format,x);
    fval_format=repmat(fval_format_base,1,length(fval));
    fprintf(file_result,fval_format,fval);
    fval_format=repmat(fval_format_base,1,length(con));
    fprintf(file_result,fval_format,con);
    fval_format=repmat(fval_format_base,1,length(coneq));
    fprintf(file_result,fval_format,coneq);
    fprintf(file_result,'\n');
end

fclose(file_optimalSurrogate_output);
clear('file_optimalSurrogate_output');
fclose(file_result);
clear('file_result');
end

function [x_list,fval_list,con_list,coneq_list]=dataLibraryLoad...
    (data_library_name,low_bou,up_bou)
% load data from data library
% low_bou, up_bou is range of data
% updata format:
% variable_number, fval_number, con_number, coneq_number
% x, fval, con, coneq
%
if nargin < 3
    up_bou=inf;
    if nargin < 2
        low_bou=-inf;
        if nargin < 1
            error('dataLibraryLoad: lack data_library_name');
        end
    end
end

if ~strcmp(data_library_name(end-3:end),'.txt')
    data_library_name=[data_library_name,'.txt'];
end

% updata format:
% variable_number, fval_number, con_number, coneq_number
% x, fval, con, coneq
if exist(data_library_name,'file')==2
    data_list=importdata(data_library_name);
    if ~isempty(data_list)
        % search whether exist point
        x_list=[];
        fval_list=[];
        con_list=[];
        coneq_list=[];
        
        for data_index=1:size(data_list,1)
            data=data_list(data_index,:);
            
            variable_number=data(1);
            fval_number=data(2);
            con_number=data(3);
            coneq_number=data(4);
            
            base=5;
            x=data(base:base+variable_number-1);
            judge=sum(x < low_bou)+sum(x > up_bou);
            if ~judge
                x_list=[x_list;x];
                base=base+variable_number;
                fval_list=[fval_list;data(base:base+fval_number-1)];
                base=base+fval_number;
                con=data(base:base+con_number-1);
                if ~isempty(con)
                    con_list=[con_list;con];
                end
                base=base+con_number;
                coneq=data(base:base+coneq_number-1);
                if ~isempty(coneq)
                    coneq_list=[coneq_list;coneq];
                end
            end
        end
    else
        x_list=[];
        fval_list=[];
        con_list=[];
        coneq_list=[];
    end
else
    x_list=[];
    fval_list=[];
    con_list=[];
    coneq_list=[];
end
end

%% LHD
function [X,X_new,distance_min_nomlz]=getLatinHypercube...
    (sample_number,variable_number,X_exist,...
    low_bou,up_bou,cheapcon_function)
% generate sample sequence latin hypercube
% election sequential method is used(sample and iteration)
%
% sample number is total point in area
% default low_bou is 0, up_bou is 1, cheapcon_function is []
% low_bou and up_bou is colume vector
% X_exist, X, supply_X is x_number x variable_number matrix
% X_exist should meet bou
%
% reference:
% [1]LONG T, LI X, SHI R, et al., Gradient-Free Trust-Region-Based
% Adaptive Response Surface Method for Expensive Aircraft Optimization[J].
% AIAA Journal, 2018, 56(2): 862-73.
%
% Copyright 2022 Adel
%
if nargin < 6
    cheapcon_function=[];
    if nargin < 5
        if nargin < 3
            if nargin < 2
                error('getLatinHypercube: lack variable_number');
            end
        end
        low_bou=zeros(variable_number,1);
        up_bou=ones(variable_number,1);
    end
end

% check x_exist_list if meet boundary
if ~isempty(X_exist)
    if size(X_exist,2) ~= variable_number
        error('getLatinHypercube: x_exist_list variable_number error');
    end
    index=find(X_exist < low_bou);
    index=[index,find(X_exist > up_bou)];
    if ~isempty(index)
        error('getLatinHypercube: x_exist_list range error');
    end
    X_exist_nomlz=(X_exist-low_bou)./(up_bou-low_bou);
else
    X_exist_nomlz=[];
end

if sample_number <= 0
    X=[];
    X_new=[];
    distance_min_nomlz=[];
    return;
end

iteration_max=1000*variable_number;
x_new_number=sample_number-size(X_exist,1);
if x_new_number <= 0
    X=X_exist;
    X_new=[];
    distance_min_nomlz=getMinDistance(X_exist_nomlz);
    return;
end

% get quasi-feasible point
x_initial_number=100*x_new_number;
if ~isempty(cheapcon_function)
    X_supply_quasi_nomlz=[];
    % check if have enough X_supply_nomlz
    while size(X_supply_quasi_nomlz,1) < 100*x_new_number
        X_supply_initial_nomlz=rand(x_initial_number,variable_number);
        x_index=1;
        while x_index <= size(X_supply_initial_nomlz,1)
            x_supply=X_supply_initial_nomlz(x_index,:).*(up_bou-low_bou)+low_bou;
            if cheapcon_function(x_supply) > 0
                X_supply_initial_nomlz(x_index,:)=[];
            else
                x_index=x_index+1;
            end
        end
        X_supply_quasi_nomlz=[X_supply_quasi_nomlz;X_supply_initial_nomlz];
    end
else
    X_supply_quasi_nomlz=rand(x_initial_number,variable_number);
end

% iterate and get final x_supply_list
iteration=0;
x_supply_quasi_number=size(X_supply_quasi_nomlz,1);
distance_min_nomlz=0;
X_new_nomlz=[];
while iteration <= iteration_max
    % random select x_new_number X to X_trial_nomlz
    x_select_index=randperm(x_supply_quasi_number,x_new_number);
    
    % get distance min itertion X_
    distance_min_iteration=getMinDistanceIter...
        (X_supply_quasi_nomlz(x_select_index,:),X_exist_nomlz);
    
    % if distance_min_iteration is large than last time
    if distance_min_iteration > distance_min_nomlz
        distance_min_nomlz=distance_min_iteration;
        X_new_nomlz=X_supply_quasi_nomlz(x_select_index,:);
    end
    
    iteration=iteration+1;
end
X_new=X_new_nomlz.*(up_bou-low_bou)+low_bou;
X=[X_new;X_exist];
distance_min_nomlz=getMinDistance([X_new_nomlz;X_exist_nomlz]);

    function distance_min__=getMinDistance(x_list__)
        % get distance min from x_list
        %
        % sort x_supply_list_initial to decrese distance calculate times
        x_list__=sortrows(x_list__,1);
        [sample_number__,variable_number__]=size(x_list__);
        distance_min__=variable_number__;
        for x_index__=1:sample_number__
            x_curr__=x_list__(x_index__,:);
            x_next_index__=x_index__ + 1;
            % only search in min_distance(x_list had been sort)
            search_range__=variable_number__;
            while x_next_index__ <= sample_number__ &&...
                    (x_list__(x_next_index__,1)-x_list__(x_index__,1))^2 ...
                    < search_range__
                x_next__=x_list__(x_next_index__,:);
                distance_temp__=sum((x_next__-x_curr__).^2);
                if distance_temp__ < distance_min__
                    distance_min__ = distance_temp__;
                end
                if distance_temp__ < search_range__
                    search_range__ = distance_temp__;
                end
                x_next_index__=x_next_index__+1;
            end
        end
        distance_min__=sqrt(distance_min__);
    end
    function distance_min__=getMinDistanceIter...
            (x_list__,x_exist_list__)
        % get distance min from x_list
        %
        % sort x_supply_list_initial to decrese distance calculate times
        x_list__=sortrows(x_list__,1);
        [sample_number__,variable_number__]=size(x_list__);
        distance_min__=variable_number__;
        for x_index__=1:sample_number__
            x_curr__=x_list__(x_index__,:);
            x_next_index__=x_index__ + 1;
            % only search in min_distance(x_list had been sort)
            search_range__=variable_number__;
            while x_next_index__ <= sample_number__ &&...
                    (x_list__(x_next_index__,1)-x_list__(x_index__,1))^2 ...
                    < search_range__
                x_next__=x_list__(x_next_index__,:);
                distance_temp__=sum((x_next__-x_curr__).^2);
                if distance_temp__ < distance_min__
                    distance_min__ = distance_temp__;
                end
                if distance_temp__ < search_range__
                    search_range__ = distance_temp__;
                end
                x_next_index__=x_next_index__+1;
            end
            for x_exist_index=1:size(x_exist_list__,1)
                x_next__=x_exist_list__(x_exist_index,:);
                distance_temp__=sum((x_next__-x_curr__).^2);
                if distance_temp__ < distance_min__
                    distance_min__ = distance_temp__;
                end
            end
        end
        distance_min__=sqrt(distance_min__);
    end
end
