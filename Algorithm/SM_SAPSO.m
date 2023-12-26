clc;
clear;
close all hidden;

data_library_name='optimalSurrogate_SA_DE_result';

variable_number=2;
object_function=@(x) functionPKObject(x);
A=[];
B=[];
Aeq=[];
Beq=[];
low_bou=-3*ones(variable_number,1);
up_bou=3*ones(variable_number,1);
nonlcon_function=[];
cheapcon_function=[];

% variable_number=10;
% object_function=@(x) functionA10Object(x);
% A=[];
% B=[];
% Aeq=[];
% Beq=[];
% low_bou=zeros(variable_number,1);
% up_bou=ones(variable_number,1);
% nonlcon_function=[];
% cheapcon_function=[];

% variable_number=2;
% object_function=@functionG06Object;
% A=[];
% B=[];
% Aeq=[];
% Beq=[];
% low_bou=[13;0];
% up_bou=[100;100];
% nonlcon_function=@functionG06Nonlcon;
% cheapcon_function=[];
% model_function=[];

% variable_number=4;
% object_function=@(x) functionPVD4Object(x);
% object_function_low=@(x) functionPVD4ObjectLow(x);
% A=[-1,0,0.0193,0;
%     0,-1,0.00954,0;];
% B=[0;0];
% Aeq=[];
% Beq=[];
% low_bou=[0;0;0;0];
% up_bou=[1;1;50;240];
% nonlcon_function=@(x) functionPVD4Nonlcon(x);
% cheapcon_function=@(x) cheapconFunction(x,A,B,Aeq,Beq);
% model_function=[];

% fmincon_options=optimoptions('fmincon','Algorithm','sqp');
% x_intial=rand(variable_number,1).*(up_bou-low_bou)+low_bou;
% [x_best,fval_best,~,output]=fmincon...
%     (object_function,x_intial,A,B,Aeq,Beq,low_bou,up_bou,nonlcon_function,fmincon_options)

delete([data_library_name,'.txt']);
delete('result_total.txt');

[x_best,fval_best,NFE,output]=optimalSurrogateSADEKTS...
    (object_function,variable_number,low_bou,up_bou,nonlcon_function,...
    cheapcon_function,[],[],[])
result_x_best=output.result_x_best;
result_fval_best=output.result_fval_best;

figure(1);
plot(result_fval_best);

figure(2);
[x_list,fval_list,con_list,coneq_list]=dataLibraryLoad...
    (data_library_name,low_bou,up_bou);
scatter3(x_list(:,1),x_list(:,2),fval_list);
xlabel('X');
ylabel('Y');
zlabel('Z');

function [con,coneq]=cheapconFunction(x,A,B,Aeq,Beq,cheapcon_function)
% convert A, B, Aeq, Beq to total cheapcon function
%
if nargin < 6
    cheapcon_function=[];
    if nargin < 5
        Beq=[];
        if nargin < 4
            Aeq=[];
            if nargin < 3
                B=[];
                if nargin < 2
                    A=[];
                end
            end
        end
    end
end
con=[];
coneq=[];
if ~isempty(A)
    if isempty(B)
        con=[con;A*x];
    else
        con=[con;A*x-B];
    end
end
if ~isempty(Aeq)
    if isempty(Beq)
        coneq=[coneq;Aeq*x];
    else
        coneq=[coneq;Aeq*x-Beq];
    end
end
if ~isempty(cheapcon_function)
    [lincon,linconeq]=cheapcon_function(x);
    con=[con;lincon];
    coneq=[coneq;linconeq];
end
end

function [x_best,fval_best,NFE,output]=optimalSurrogateSADEKTS...
    (object_function,variable_number,low_bou,up_bou,nonlcon_function,...
    cheapcon_function,model_function,....
    NFE_max,iteration_max,torlance,nonlcon_torlance)
% Surrogate-Assisted Differential Evolution method version 0
% use differ evolution
% all function exchange x should be colume vector
% x_list is x_number x variable_number matrix
% both nonlcon_function and cheapcon_function format is [con,coneq]
% model_function should output fval, format is [fval,con,coneq]
% con or coneq can be colume vector if there was more than one constrain
%
% Copyright 2022 Adel
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

if nargin < 8
    NFE_max=[];
    if nargin < 7
        cheapcon_function=[];
        if nargin < 6
            nonlcon_function=[];
        end
    end
end
DRAW_FIGURE_FLAG=0; % whether draw data
INFORMATION_FLAG=1; % whether print data
CONVERGENCE_JUDGMENT_FLAG=0; % whether judgment convergence

if isempty(iteration_max)
    iteration_max=100;
end

% parameter
population_number=min(100,10*variable_number);
RBF_number=max(100,(variable_number+1)*(variable_number+2)/2);

scaling_factor=0.8; % F
cross_rate=0.8;

protect_range=1e-4;

data_library_name='optimalSurrogate_SA_DE_result';
file_result = fopen('result_total.txt','a');
fprintf(file_result,'%s\n',datetime);
fclose(file_result);
clear('file_result');

done=0;NFE=0;iteration=0;

% if do not input model_function, generate model_function
if nargin < 7 || isempty(model_function)
    model_function=@(x) modelFunction(x,object_function,nonlcon_function);
end

iteration=iteration+1;

% step 2
% generate initial sample x_list
[~,x_updata_list,~]=getLatinHypercube...
    (population_number,variable_number,[],low_bou,up_bou,cheapcon_function);

% detech expensive constraints
if ~isempty(x_updata_list)
    [~,con,coneq]=dataLibraryUpdata...
        (data_library_name,model_function,x_updata_list(1,:));NFE=NFE+1;
    x_updata_list=x_updata_list(2:end,:);
else
    [~,con,coneq]=dataLibraryLoad(data_library_name,low_bou,up_bou);
end
if ~isempty(con) || ~isempty(coneq)
    expensive_nonlcon_flag=1;
else
    expensive_nonlcon_flag=0;
end

% NFE setting
if expensive_nonlcon_flag
    if isempty(NFE_max)
        NFE_max=50*variable_number;
    end
    iteration_max=50*variable_number;
else
    if isempty(NFE_max)
        NFE_max=20*variable_number;
    end
    iteration_max=20*variable_number;
end

result_x_best=zeros(iteration_max,variable_number);
result_fval_best=zeros(iteration_max,1);

% import data from data library
[x_list,fval_list,con_list,coneq_list]=dataLibraryLoad...
    (data_library_name,low_bou,up_bou);

% updata data library by x_list
[fval_updata_list,con_updata_list,coneq_updata_list]=dataLibraryUpdata...
    (data_library_name,model_function,x_updata_list);NFE=NFE+size(x_updata_list,1);
x_list=[x_list;x_updata_list];
fval_list=[fval_list;fval_updata_list];
con_list=[con_list;con_updata_list];
coneq_list=[coneq_list;coneq_updata_list];

search_flag=0; % global search or local search, 0 is global and 1 is local
while ~done
    infor_search_flag=search_flag;
    % nomalization con with average
    fval_max=mean(abs(fval_list),1);
    fval_nomlz_list=fval_list./fval_max;
    if ~isempty(con_list)
        con_max_list=mean(abs(con_list),1);
        con_nomlz_list=con_list./con_max_list;
    else
        con_max_list=[];
        con_nomlz_list=[];
    end
    if ~isempty(coneq_list)
        coneq_max_list=mean(abs(coneq_list),1);
        coneq_nomlz_list=coneq_list./coneq_max_list;
    else
        coneq_max_list=[];
        coneq_nomlz_list=[];
    end
    
    if search_flag == 0
        % global search
        [x_global_infill,feasiable_index_list]=searchGlobal...
            (x_list,fval_nomlz_list,con_nomlz_list,coneq_nomlz_list,...
            variable_number,low_bou,up_bou,cheapcon_function,nonlcon_torlance,...
            population_number,scaling_factor,cross_rate,...
            expensive_nonlcon_flag,DRAW_FIGURE_FLAG);
        
        [x_global_infill,fval_global_infill,con_global_infill,coneq_global_infill,NFE_p]=dataLibraryUpdataProtect...
            (data_library_name,model_function,x_global_infill',...
            x_list,fval_list,con_list,coneq_list,...
            low_bou,up_bou,protect_range);NFE=NFE+NFE_p;
        
        x_list=[x_list;x_global_infill];
        fval_list=[fval_list;fval_global_infill];
        con_list=[con_list;con_global_infill];
        coneq_list=[coneq_list;coneq_global_infill];
        
        % whether impove pupolation, if imporve, continue global
        search_flag=1;
        if isempty(feasiable_index_list)
            min_con=min(con_list,[],1);
            min_coneq=min(abs(coneq_list),[],1);
            if ~isempty(con_list)
                if ~sum(con_global_infill > min_con) % imporve
                    search_flag=0;
                end
            end
            if isempty(coneq_list)
                if ~sum(abs(coneq_global_infill) > min_coneq) % imporve
                    search_flag=0;
                end
            end
        else
            min_fval=min(fval_list,[],1);
            if fval_global_infill < min_fval  % imporve
                search_flag=0;
            end
        end
    else
        % local search
        x_local_infill=searchLocal...
            (x_list,fval_nomlz_list,con_nomlz_list,coneq_nomlz_list,...
            variable_number,low_bou,up_bou,cheapcon_function,nonlcon_torlance,...
            population_number,RBF_number,...
            DRAW_FIGURE_FLAG);
        
        [x_local_infill,fval_local_infill,con_local_infill,coneq_local_infill,NFE_p]=dataLibraryUpdataProtect...
            (data_library_name,model_function,x_local_infill',...
            x_list,fval_list,con_list,coneq_list,...
            low_bou,up_bou,protect_range);NFE=NFE+NFE_p;
        
        x_list=[x_list;x_local_infill];
        fval_list=[fval_list;fval_local_infill];
        con_list=[con_list;con_local_infill];
        coneq_list=[coneq_list;coneq_local_infill];
        
        % whether impove pupolation, if imporve, continue local
        search_flag=0;
        if isempty(feasiable_index_list)
            min_con=min(con_list,[],1);
            min_coneq=min(abs(coneq_list),[],1);
            if ~isempty(con_list)
                if ~sum(con_global_infill > min_con) % imporve
                    search_flag=1;
                end
            end
            if isempty(coneq_list)
                if ~sum(abs(coneq_global_infill) > min_coneq) % imporve
                    search_flag=1;
                end
            end
        else
            min_fval=min(fval_list,[],1);
            if fval_global_infill < min_fval  % imporve
                search_flag=1;
            end
        end
    end
    
    % find best result to record
    [x_best,fval_best,con_best,coneq_best]=findMinRaw...
        (x_list,fval_list,con_list,coneq_list,...
        cheapcon_function,nonlcon_torlance);
    
    if INFORMATION_FLAG
        fprintf('iteration:          %-3d    NFE:    %-3d\n',iteration,NFE);
        if infor_search_flag == 0
            fprintf('global x:          %s\n',num2str(x_global_infill));
            fprintf('global value:      %f\n',fval_global_infill);
            fprintf('global violation:  %s  %s\n',num2str(con_global_infill),num2str(coneq_global_infill));
        else
            fprintf('local  x:          %s\n',num2str(x_local_infill));
            fprintf('local  value:      %f\n',fval_local_infill);
            fprintf('local  violation:  %s  %s\n',num2str(con_local_infill),num2str(coneq_local_infill));
        end
        fprintf('\n');
    end
    
    result_x_best(iteration,:)=x_best';
    result_fval_best(iteration,:)=fval_best;
    iteration=iteration+1;
    
    % forced interrupt
    if iteration > iteration_max || NFE >= NFE_max
        done=1;
    end
    
    % convergence judgment
    if CONVERGENCE_JUDGMENT_FLAG
        if (iteration > 1 && ...
                abs((fval_best-fval_best_old)/fval_best_old) < torlance)
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
    fval_best_old=fval_best;
    
end

result_x_best=result_x_best(1:iteration-1,:);
result_fval_best=result_fval_best(1:iteration-1);

output.result_x_best=result_x_best;
output.result_fval_best=result_fval_best;

    function [x_global_infill,feasiable_index_list]=searchGlobal...
            (x_list,fval_nomlz_list,con_nomlz_list,coneq_nomlz_list,...
            variable_number,low_bou,up_bou,cheapcon_function,nonlcon_torlance,...
            population_number,scaling_factor,cross_rate,...
            expensive_nonlcon_flag,DRAW_FIGURE_FLAG)
        % find global infill point function
        %
        
        % step 4
        % updata kriging model and function
        [kriging_model_fval,kriging_model_con,kriging_model_coneq,output_kriging]=getKrigingModel...
            (x_list,fval_nomlz_list,con_nomlz_list,coneq_nomlz_list);
        object_function_surrogate=output_kriging.object_function_surrogate;
        object_function_variance=output_kriging.object_function_variance;
        nonlcon_function_surrogate=output_kriging.nonlcon_function_surrogate;
        nonlcon_function_variance=output_kriging.nonlcon_function_variance;
        
        % step 5
        % rank x_list data
        [x_rank_list,~,~,~]=rankData...
            (x_list,fval_nomlz_list,con_nomlz_list,coneq_nomlz_list,...
            cheapcon_function,nonlcon_torlance);
        
        % step 6
        % only the first population_number will be use
        x_best_popu_list=x_rank_list(1:population_number,:);
        
        % differ evolution mutations
        X_new_R1=differEvolutionRand...
            (low_bou,up_bou,x_best_popu_list,scaling_factor,population_number,1);
        X_new_R2=differEvolutionRand...
            (low_bou,up_bou,x_best_popu_list,scaling_factor,population_number,2);
        X_new_CR=differEvolutionCurrentRand...
            (low_bou,up_bou,x_best_popu_list,scaling_factor);
        X_new_CB=differEvolutionCurrentBest...
            (low_bou,up_bou,x_best_popu_list,scaling_factor,1);
        
        % differ evolution crossover
        X_new_R1=differEvolutionCrossover...
            (low_bou,up_bou,x_best_popu_list,X_new_R1,cross_rate);
        X_new_R2=differEvolutionCrossover...
            (low_bou,up_bou,x_best_popu_list,X_new_R2,cross_rate);
        X_new_CR=differEvolutionCrossover...
            (low_bou,up_bou,x_best_popu_list,X_new_CR,cross_rate);
        X_new_CB=differEvolutionCrossover...
            (low_bou,up_bou,x_best_popu_list,X_new_CB,cross_rate);
        
        % find global infill point base kriging model from offspring X
        x_DE_list=[X_new_R1;X_new_R2;X_new_CR;X_new_CB];
        fval_DE_list=zeros(4*population_number,1);
        fval_var_DE_list=zeros(4*population_number,1);
        if ~isempty(con_nomlz_list)
            con_DE_list=zeros(4*population_number,size(con_nomlz_list,2));
            con_var_DE_list=zeros(4*population_number,size(con_nomlz_list,2));
        end
        if ~isempty(coneq_nomlz_list)
            coneq_DE_list=zeros(4*population_number,size(coneq_nomlz_list,2));
            coneq_var_DE_list=zeros(4*population_number,size(coneq_nomlz_list,2));
        end
        
        % evaluate each x_offspring fval and constraints
        for x_index=1:4*population_number
            x=x_DE_list(x_index,:)';
            fval_DE_list(x_index,1)=object_function_surrogate(x);
            fval_var_DE_list(x_index,1)=object_function_variance(x);
        end
        feasiable_index_list=[];
        if expensive_nonlcon_flag
            if ~isempty(nonlcon_function_surrogate)
                [con_temp,coneq_temp]=nonlcon_function_surrogate(x);
                if ~isempty(con_temp)
                    con_DE_list(x_index,:)=con_temp';
                end
                if ~isempty(coneq_temp)
                    coneq_DE_list(x_index,:)=coneq_temp';
                end
            end
            if ~isempty(nonlcon_function_variance)
                [con_var,coneq_var]=nonlcon_function_variance(x);
                if ~isempty(con_var)
                    con_var_DE_list(x_index,:)=con_var';
                end
                if ~isempty(coneq_var)
                    coneq_var_DE_list(x_index,:)=coneq_var';
                end
            end
            if max([con_temp;abs(coneq_temp)]) < nonlcon_torlance
                feasiable_index_list=[feasiable_index_list,x_index];
            end
        else
            feasiable_index_list=1:4*population_number;
        end
        
        % if have feasiable_index_list, only use feasiable to choose
        if ~isempty(feasiable_index_list)
            x_DE_list=x_DE_list(feasiable_index_list,:);
            fval_DE_list=fval_DE_list(feasiable_index_list);
            fval_var_DE_list=fval_var_DE_list(feasiable_index_list);
            if ~isempty(con_nomlz_list)
                con_DE_list=con_DE_list(feasiable_index_list,:);
                con_var_DE_list=con_var_DE_list(feasiable_index_list,:);
            end
            if ~isempty(coneq_nomlz_list)
                coneq_DE_list=coneq_DE_list(feasiable_index_list,:);
                coneq_var_DE_list=coneq_var_DE_list(feasiable_index_list,:);
            end
        end
        
        % base on constaints improve select global infill
        % lack process of equal constraints
        if isempty(feasiable_index_list)
            con_DE_base=max(min(con_DE_list,[],1),0);
            con_impove_probability_list=sum(...
                normpdf((con_DE_base-con_DE_list)./con_var_DE_list),2);
            [~,con_best_index]=max(con_impove_probability_list);
            con_best_index=con_best_index(1);
            x_global_infill=x_DE_list(con_best_index,:)';
        end
        
        % base on best fitness select global infill
        if ~isempty(feasiable_index_list)
            fval_DE_min=min(fval_DE_list,[],1);
            fval_DE_max=max(fval_DE_list,[],1);
            fval_var_DE_min=min(fval_var_DE_list,[],1);
            fval_var_DE_max=max(fval_var_DE_list,[],1);
            DE_fitness_list=-(fval_DE_list-fval_DE_min)/(fval_DE_max-fval_DE_min)+...
                0.5*(fval_var_DE_list-fval_var_DE_min)/(fval_var_DE_max-fval_var_DE_min);
            [~,fitness_best_index]=max(DE_fitness_list);
            fitness_best_index=fitness_best_index(1);
            x_global_infill=x_DE_list(fitness_best_index,:)';
        end
        
        if DRAW_FIGURE_FLAG && variable_number < 3
            line(x_global_infill(1),x_global_infill(2),'Marker','o','color','r')
            interpolationVisualize(kriging_model_fval,low_bou,up_bou);
        end
    end
    function x_local_infill=searchLocal...
            (x_list,fval_nomlz_list,con_nomlz_list,coneq_nomlz_list,...
            variable_number,low_bou,up_bou,cheapcon_function,nonlcon_torlance,...
            population_number,RBF_number,...
            DRAW_FIGURE_FLAG)
        % find local infill point function
        %
        
        % step 8
        % rand select initial local point from x_list
        x_index=randi(population_number);
        x_initial=x_list(x_index,:)';
        
        % rank x_list data
        %         [x_list_rank,~,~,~]=rankData...
        %             (x_list,fval_nomlz_list,con_nomlz_list,coneq_nomlz_list,...
        %             cheapcon_function,nonlcon_torlance);
        %         x_initial=x_list_rank(1,:)';
        
        % select nearest point to construct RBF
        RBF_number=min(RBF_number,size(x_list,1));
        distance=sum(((x_initial'-x_list)./(up_bou'-low_bou')).^2,2);
        [~,index_list]=sort(distance);
        index_list=index_list(1:RBF_number);
        x_RBF_list=x_list(index_list,:);
        fval_RBF_nomlz_list=fval_nomlz_list(index_list,:);
        if ~isempty(con_nomlz_list)
            con_RBF_nomlz_list=con_nomlz_list(index_list,:);
        else
            con_RBF_nomlz_list=[];
        end
        if ~isempty(coneq_nomlz_list)
            coneq_RBF_nomlz_list=coneq_nomlz_list(index_list,:);
        else
            coneq_RBF_nomlz_list=[];
        end
        
        % get RBF model and function
        [radialbasis_model_fval,radialbasis_model_con,radialbasis_model_coneq,output_radialbasis]=getRadialBasisModel...
            (x_RBF_list,fval_RBF_nomlz_list,con_RBF_nomlz_list,coneq_RBF_nomlz_list);
        object_function_surrogate=output_radialbasis.object_function_surrogate;
        nonlcon_function_surrogate=output_radialbasis.nonlcon_function_surrogate;
        
        % get local infill point
        % obtian total constraint function
        if ~isempty(nonlcon_function_surrogate) || ~isempty(cheapcon_function)
            constraint_function=@(x) totalconFunction...
                (x,nonlcon_function_surrogate,cheapcon_function,nonlcon_torlance);
        else
            constraint_function=[];
        end
        fmincon_options=optimoptions('fmincon','Display','none','Algorithm','sqp');
        x_local_infill=fmincon(object_function_surrogate,x_initial,[],[],[],[],...
            low_bou,up_bou,constraint_function,fmincon_options);
        
        if DRAW_FIGURE_FLAG && variable_number < 3
            line(x_local_infill(1),x_local_infill(2),'Marker','o','color','r')
            interpolationVisualize(radialbasis_model_fval,low_bou,up_bou);
        end
    end
    function [fval,con,coneq]=modelFunction(x,object_function,nonlcon_function)
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
    function [con,coneq]=totalconFunction...
            (x,nonlcon_function,cheapcon_function,nonlcon_torlance)
        con=[];
        coneq=[];
        if ~isempty(nonlcon_function)
            [expencon,expenconeq]=nonlcon_function(x);
            con=[con;expencon-nonlcon_torlance];
            coneq=[coneq;expenconeq-nonlcon_torlance];
        end
        if ~isempty(cheapcon_function)
            [expencon,expenconeq]=cheapcon_function(x);
            con=[con;expencon];
            coneq=[coneq;expenconeq];
        end
    end
    function X_new=differEvolutionRand(low_bou,up_bou,X,F,x_number,rand_number)
        if nargin < 4
            rand_number=1;
            if nargin < 3
                x_number=1;
                if nargin < 2
                    error('differEvolutionRand: lack scaling factor F');
                end
            end
        end
        [x_number__,variable_number__]=size(X);
        X_new=zeros(x_number,variable_number__);
        for x_index__=1:x_number
            index__=randi(x_number__,2*rand_number+1,1);
            X_new(x_index__,:)=X(index__(1),:);
            for rand_index__=1:rand_number
                X_new(x_index__,:)=X_new(x_index__,:)+...
                    F*(X(index__(2*rand_index__),:)-X(index__(2*rand_index__+1),:));
                X_new(x_index__,:)=max(X_new(x_index__,:),low_bou');
                X_new(x_index__,:)=min(X_new(x_index__,:),up_bou');
            end
        end
    end
    function X_new=differEvolutionCurrentRand(low_bou,up_bou,X,F)
        [x_number__,variable_number__]=size(X);
        X_new=zeros(x_number__,variable_number__);
        for x_index__=1:x_number__
            index__=randi(x_number__,3,1);
            X_new(x_index__,:)=X(x_index__,:)+...
                F*(X(index__(1),:)-X(x_index__,:)+...
                X(index__(2),:)-X(index__(3),:));
            X_new(x_index__,:)=max(X_new(x_index__,:),low_bou');
            X_new(x_index__,:)=min(X_new(x_index__,:),up_bou');
        end
    end
    function X_new=differEvolutionCurrentBest(low_bou,up_bou,X,F,x_best_index)
        [x_number__,variable_number__]=size(X);
        X_new=zeros(x_number__,variable_number__);
        for x_index__=1:x_number__
            index__=randi(x_number__,2,1);
            X_new(x_index__,:)=X(x_index__,:)+...
                F*(X(x_best_index,:)-X(x_index__,:)+...
                X(index__(1),:)-X(index__(2),:));
            X_new(x_index__,:)=max(X_new(x_index__,:),low_bou');
            X_new(x_index__,:)=min(X_new(x_index__,:),up_bou');
        end
    end
    function X_new=differEvolutionCrossover(low_bou,up_bou,X,V,C_R)
        if size(X,1)~=size(V,1)
            error('differEvolutionOffspring: size incorrect');
        end
        [x_number__,variable_number__]=size(X);
        X_new=X;
        rand_number=rand(x_number__,variable_number__);
        index__=find(rand_number < C_R);
        X_new(index__)=V(index__);
        for x_index__=1:x_number__
            X_new(x_index__,:)=max(X_new(x_index__,:),low_bou');
            X_new(x_index__,:)=min(X_new(x_index__,:),up_bou');
        end
    end
    function [x_updata_list,fval_updata_list,con_updata_list,coneq_updata_list,NFE]=dataLibraryUpdataProtect...
            (data_library_name,model_function,x_add_list,...
            x_list,fval_list,con_list,coneq_list,...
            low_bou,up_bou,protect_range)
        % function updata data with same_point_avoid protect
        % return fval
        % all list is x_number x variable_number matrix
        %
        NFE=0;
        x_updata_list=[];fval_updata_list=[];con_updata_list=[];coneq_updata_list=[];
        for x_index__=size(x_add_list,1)
            x_updata__=x_add_list(x_index__,:);
            
            % check x_potential if exist in data library
            % if not, updata data libraray
            distance__=sum(((x_updata__-x_list)./(low_bou'-up_bou')).^2,2);
            [~,min_index__]=min(distance__);
            if distance__(min_index__) < protect_range^2
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
                    (data_library_name,model_function,x_updata__);NFE=NFE+1;
            end
            x_updata_list=[x_updata_list;x_updata__];
            fval_updata_list=[fval_updata_list;fval_updata__];
            con_updata_list=[con_updata_list;con_updata__];
            coneq_updata_list=[coneq_updata_list;coneq_updata__];
        end
    end
end

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

function [kriging_model_fval,kriging_model_con,kriging_model_coneq,output]=getKrigingModel...
    (x_list,fval_list,con_list,coneq_list)
% base on library_data to create kriging model and function
% if input model, function will updata model
% object_function is single fval output
% nonlcon_function is normal nonlcon_function which include con, coneq
% con is colume vector, coneq is colume vector
% var_function is same
%

kriging_model_fval=interpolationKrigingPreModel...
    (x_list,fval_list);

if ~isempty(con_list)
    kriging_model_con=struct('X',[],'Y',[],'X_normalize',[],'Y_normalize',[],...
        'base_function_list',[],'fval_regression',[],'covariance',[],'inv_convariance',[],...
        'theta',[],'beta',[],'gama',[],'sigma_sq',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],...
        'predict_function',[]);
    kriging_model_con=repmat(kriging_model_con,[size(con_list,2),1]);
    for con_index=1:size(con_list,2)
        kriging_model_con(con_index)=interpolationKrigingPreModel...
            (x_list,con_list(:,con_index));
    end
else
    kriging_model_con=[];
end

if ~isempty(coneq_list)
    kriging_model_coneq=struct('X',[],'Y',[],'X_normalize',[],'Y_normalize',[],...
        'base_function_list',[],'fval_regression',[],'covariance',[],'inv_convariance',[],...
        'theta',[],'beta',[],'gama',[],'sigma_sq',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],...
        'predict_function',[]);
    kriging_model_coneq=repmat(kriging_model_coneq,[size(coneq_list,2),1]);
    for coneq_index=1:size(coneq_list,2)
        kriging_model_coneq(coneq_index)=interpolationKrigingPreModel...
            (x_list,coneq_list(:,coneq_index),theta_coneq,base_function_list);
    end
else
    kriging_model_coneq=[];
end

object_function_surrogate=@(predict_x) objectFunctionSurrogate(predict_x,kriging_model_fval);
object_function_variance=@(predict_x) objectFunctionVariance(predict_x,kriging_model_fval);
if isempty(kriging_model_con) && isempty(kriging_model_coneq)
    nonlcon_function_surrogate=[];
    nonlcon_function_variance=[];
else
    nonlcon_function_surrogate=@(predict_x) nonlconFunctionSurrogate(predict_x,kriging_model_con,kriging_model_coneq);
    nonlcon_function_variance=@(predict_x) nonlconFunctionVariance(predict_x,kriging_model_con,kriging_model_coneq);
end

output.object_function_surrogate=object_function_surrogate;
output.object_function_variance=object_function_variance;
output.nonlcon_function_surrogate=nonlcon_function_surrogate;
output.nonlcon_function_variance=nonlcon_function_variance;
output.x_list=x_list;
output.fval_list=fval_list;
output.con_list=con_list;
output.coneq_list=coneq_list;


    function fval=objectFunctionSurrogate...
            (predict_x,kriging_model_fval)
        [fval,~]=kriging_model_fval.predict_function(predict_x);
    end
    function fval_var=objectFunctionVariance...
            (predict_x,kriging_model_fval)
        [~,fval_var]=kriging_model_fval.predict_function(predict_x);
    end
    function [con,coneq]=nonlconFunctionSurrogate...
            (predict_x,kriging_model_con,kriging_model_coneq)
        if isempty(kriging_model_con)
            con=[];
        else
            con=zeros(length(kriging_model_con),1);
            for con_index__=1:length(kriging_model_con)
                [con(con_index__),~]=kriging_model_con(con_index__).predict_function....
                    (predict_x);
            end
        end
        if isempty(kriging_model_coneq)
            coneq=[];
        else
            coneq=zeros(length(kriging_model_coneq),1);
            for coneq_index__=1:length(kriging_model_con)
                [coneq(coneq_index__),~]=kriging_model_coneq(coneq_index__).predict_function...
                    (predict_x);
            end
        end
    end
    function [con_var,coneq_var]=nonlconFunctionVariance...
            (predict_x,kriging_model_con,kriging_model_coneq)
        if isempty(kriging_model_con)
            con_var=[];
        else
            con_var=zeros(length(kriging_model_con),1);
            for con_index__=1:length(kriging_model_con)
                [~,con_var(con_index__)]=kriging_model_con(con_index__).predict_function....
                    (predict_x);
            end
        end
        if isempty(kriging_model_coneq)
            coneq_var=[];
        else
            coneq_var=zeros(length(kriging_model_coneq),1);
            for coneq_index__=1:length(kriging_model_con)
                [~,coneq_var(coneq_index__)]=kriging_model_coneq(coneq_index__).predict_function...
                    (predict_x);
            end
        end
    end
end
function kriging_model=interpolationKrigingPreModel...
    (X,Y,theta,base_function_list)
% prepare model, optimal theta, load result and calculation parameter
% input initial data X, Y, which are real data
% X, Y are x_number x variable_number matrix
% aver_X,stdD_X is 1 x x_number matrix
% output is a kriging model, include X, Y, base_function_list
% and predict_function
% theta beta gama sigma_sq is normalizede, so predict y is normalizede
%
% Copyright 2022 Adel
%
if nargin < 4
    base_functon=@(x) 1;
    base_function_list={base_functon};
    if nargin < 3
        theta=[];
        if nargin < 2
            error('interpolationKrigingPreModel: lack input');
        end
    end
end
[x_number,variable_number]=size(X);
if isempty(theta)
    theta=ones(variable_number,1);
end

% normalize data
aver_X = mean(X);
stdD_X = std(X);
aver_Y = mean(Y);
stdD_Y = std(Y);
index__ = find(stdD_X == 0);
if  ~isempty(index__),  stdD_X(index__) = 1; end
index__ = find(stdD_Y == 0);
if  ~isempty(index__),  stdD_Y(index__) = 1; end
X_nomlz = (X - repmat(aver_X,x_number,1)) ./ repmat(stdD_X,x_number,1);
Y_nomlz = (Y - repmat(aver_Y,x_number,1)) ./ repmat(stdD_Y,x_number,1);

% initial X_dis_sq
X_dis_sq=zeros(x_number,x_number,variable_number);
for rank=1:x_number
    for colume=1:rank-1
        X_dis_sq(rank,colume,:)=X_dis_sq(colume,rank,:);
    end
    for colume=rank:x_number
        X_dis_sq(rank,colume,:)=(X_nomlz(rank,:)-X_nomlz(colume,:)).^2;
    end
end

% optimal to get hyperparameter
options = optimoptions(@fmincon,'Display','off');
low_bou_kriging=1e-1*ones(variable_number,1);
up_bou_kriging=20*ones(variable_number,1);
object_function_kriging=@(theta) objectFunctionKriging...
    (X_dis_sq,X_nomlz,Y_nomlz,x_number,variable_number,theta,base_function_list);
theta=fmincon...
    (object_function_kriging,theta,[],[],[],[],low_bou_kriging,up_bou_kriging,[],options);

% get parameter
[covariance,inv_convariance,fval_reg,beta,sigma_sq]=interpolationKriging...
    (X_dis_sq,X_nomlz,Y_nomlz,x_number,variable_number,theta,base_function_list);
gama=inv_convariance*(Y_nomlz-fval_reg*beta);

% initialization predict function
predict_function=@(predict_x) interpolationKrigingPredictor...
    (X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    theta,beta,gama,sigma_sq,...
    inv_convariance,fval_reg,base_function_list,predict_x);

kriging_model.X=X;
kriging_model.Y=Y;
kriging_model.X_normalize=X_nomlz;
kriging_model.Y_normalize=Y_nomlz;
kriging_model.base_function_list=base_function_list;
kriging_model.fval_regression=fval_reg;
kriging_model.covariance=covariance;
kriging_model.inv_convariance=inv_convariance;

kriging_model.theta=theta;
kriging_model.beta=beta;
kriging_model.gama=gama;
kriging_model.sigma_sq=sigma_sq;
kriging_model.aver_X=aver_X;
kriging_model.stdD_X=stdD_X;
kriging_model.aver_Y=aver_Y;
kriging_model.stdD_Y=stdD_Y;

kriging_model.predict_function=predict_function;

    function sigma_sq=objectFunctionKriging...
            (X_dis_sq,X,Y,x_number,variable_number,theta,base_function_list)
        % function to minimize sigma_sq
        %
        [~,~,~,~,sigma_sq]=interpolationKriging...
            (X_dis_sq,X,Y,x_number,variable_number,theta,base_function_list);
    end
    function [covariance,inv_convariance,fval_reg,beta,sigma_sq]=interpolationKriging...
            (X_dis_sq,X,Y,x_number,variable_number,theta,base_function_list)
        % total riging interpolation function
        % input X, Y as initial data, theta and base function
        % output beta and gama, xita
        %
        % Copyright 2022 Adel
        %
        % get covariance matrix
        covariance=zeros(x_number,x_number);
        for rank_index__=1:x_number
            for colume_index__=1:rank_index__-1
                covariance(rank_index__,colume_index__)=...
                    covariance(colume_index__,rank_index__);
            end
            covariance(rank_index__,rank_index__)=1;
            for colume_index__=rank_index__+1:x_number
                temp__=0;
                for variable_index=1:variable_number
                    temp__=temp__+...
                        X_dis_sq(rank_index__,colume_index__,variable_index)*...
                        theta(variable_index);
                end
                covariance(rank_index__,colume_index__)=...
                    exp(-temp__);
            end
        end
        base_function_number__=size(base_function_list,1);
        
        % stabilize
        covariance=covariance+eye(x_number)*1e-6;
        
        % F is base funcion fval of data_point_x
        fval_reg=zeros(x_number,base_function_number__);
        for base_function_index__=1:base_function_number__
            base_function=base_function_list{base_function_index__};
            for x_index__=1:x_number
                fval_reg(x_index__,base_function_index__)=...
                    base_function(X(x_index__,:));
            end
        end
        
        % coefficient calculation
        inv_convariance=inv(covariance);
        beta=(fval_reg'*inv_convariance*fval_reg)\fval_reg'*inv_convariance*Y;
        sigma_sq=(Y-fval_reg*beta)'*inv_convariance*(Y-fval_reg*beta)/x_number;
    end
    function [predict_fval,predict_variance]=interpolationKrigingPredictor...
            (X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            theta,beta,gama,sigma_sq,...
            inv_convariance,fval_reg,base_function_list,predict_x)
        % kriging interpolation predict function
        % input predict_x and kriging model
        % predict_x is row vector
        % output the predict value
        %
        % Copyright 2022 Adel
        %
        if size(predict_x,1) > 1
            predict_x=predict_x';
        end
        
        % normalize data
        predict_x=(predict_x-aver_X)./stdD_X;
        
        % predict value
        base_function_number__=size(base_function_list,1);
        predict_cov__=exp(-(predict_x-X_nomlz).^2*theta);
        predict_fval_reg__=zeros(1,base_function_number__);
        for base_function_index=1:base_function_number__
            base_function=base_function_list{base_function_index};
            predict_fval_reg__(1,base_function_index)=base_function(predict_x');
        end
        predict_fval=predict_fval_reg__'*beta+predict_cov__'*gama;
        
        % predict variance
        u__=fval_reg'*inv_convariance*predict_cov__-predict_fval_reg__;
        predict_variance=sigma_sq*...
            (1+u__'/(fval_reg'*inv_convariance*fval_reg)*u__+...
            -predict_cov__'*inv_convariance*predict_cov__);
        
        % normalize data
        predict_fval=predict_fval*stdD_Y+aver_Y;
        predict_variance=predict_variance*stdD_Y*stdD_Y;
    end
end

function [radialbasis_model_fval,radialbasis_model_con,radialbasis_model_coneq,output]=getRadialBasisModel...
    (x_list,fval_list,con_list,coneq_list)
% base on library_data to create radialbasis model and function
% if input model, function will updata model
% object_function is single fval output
% nonlcon_function is normal nonlcon_function which include con, coneq
% con is colume vector, coneq is colume vector
% var_function is same
%
radialbasis_model_fval=interpolationRadialBasisPreModel...
    (x_list,fval_list);

if ~isempty(con_list)
    radialbasis_model_con=struct('X',[],'Y',[],'X_normalize',[],'Y_normalize',[],...
        'radialbasis_matrix',[],'inv_radialbasis_matrix',[],'beta',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],'basis_function',[],...
        'predict_function',[]);
    radialbasis_model_con=repmat(radialbasis_model_con,[size(con_list,2),1]);
    for con_index=1:size(con_list,2)
        radialbasis_model_con(con_index)=interpolationRadialBasisPreModel...
            (x_list,con_list(:,con_index));
    end
else
    radialbasis_model_con=[];
end

if ~isempty(coneq_list)
    radialbasis_model_coneq=struct('X',[],'Y',[],'X_normalize',[],'Y_normalize',[],...
        'radialbasis_matrix',[],'inv_radialbasis_matrix',[],'beta',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],'basis_function',[],...
        'predict_function',[]);
    radialbasis_model_coneq=repmat(radialbasis_model_coneq,[size(coneq_list,2),1]);
    for coneq_index=1:size(coneq_list,2)
        radialbasis_model_coneq(coneq_index)=interpolationRadialBasisPreModel...
            (x_list,coneq_list(:,coneq_index));
    end
else
    radialbasis_model_coneq=[];
end

object_function_surrogate=@(predict_x) objectFunctionSurrogate(predict_x,radialbasis_model_fval);
if isempty(radialbasis_model_con) && isempty(radialbasis_model_coneq)
    nonlcon_function_surrogate=[];
else
    nonlcon_function_surrogate=@(predict_x) nonlconFunctionSurrogate(predict_x,radialbasis_model_con,radialbasis_model_coneq);
end

output.object_function_surrogate=object_function_surrogate;
output.nonlcon_function_surrogate=nonlcon_function_surrogate;
output.x_list=x_list;
output.fval_list=fval_list;
output.con_list=con_list;
output.coneq_list=coneq_list;

    function fval=objectFunctionSurrogate...
            (predict_x,ERBF_model_fval)
        fval=ERBF_model_fval.predict_function(predict_x);
    end
    function [con,coneq]=nonlconFunctionSurrogate...
            (predict_x,ERBF_model_con,ERBF_model_coneq)
        if isempty(ERBF_model_con)
            con=[];
        else
            con=zeros(length(ERBF_model_con),1);
            for con_index__=1:length(ERBF_model_con)
                con(con_index__)=ERBF_model_con...
                    (con_index__).predict_function(predict_x);
            end
        end
        if isempty(ERBF_model_coneq)
            coneq=[];
        else
            coneq=zeros(length(ERBF_model_coneq),1);
            for coneq_index__=1:length(ERBF_model_con)
                coneq(coneq_index__)=ERBF_model_coneq...
                    (coneq_index__).predict_function(predict_x);
            end
        end
    end
end
function radialbasis_model=interpolationRadialBasisPreModel...
    (X,Y,basis_function)
% radial basis function interpolation pre model function version 1
% input initial data X, Y, which are real data
% X, Y are x_number x variable_number matrix
% aver_X,stdD_X is 1 x x_number matrix
% output is a radial basis model, include X, Y, base_function
% and predict_function
% beta is normalizede, so predict y is normalizede
%
% Copyright 2022 Adel
%
if nargin < 9
    X_nomlz=[];
    if nargin < 3
        basis_function=[];
    end
end

[x_number,variable_number]=size(X);

% normalize data
aver_X = mean(X);
stdD_X = std(X);
aver_Y = mean(Y);
stdD_Y = std(Y);
index__ = find(stdD_X == 0);
if ~isempty(index__),stdD_X(index__)=1;end
index__ = find(stdD_Y == 0);
if ~isempty(index__),stdD_Y(index__)=1;end
X_nomlz = (X - repmat(aver_X,x_number,1)) ./ repmat(stdD_X,x_number,1);
Y_nomlz = (Y - repmat(aver_Y,x_number,1)) ./ repmat(stdD_Y,x_number,1);

if isempty(basis_function)
    c=(prod(max(X_nomlz)-min(Y_nomlz))/x_number)^(1/variable_number);
    basis_function=@(r) exp(-(r'*r)/c);
    %     basis_function=@(r) sqrt(r'*r+c*c);
end

[beta,rdibas_matrix,inv_rdibas_matrix]=interpolationRadialBasis...
    (X_nomlz,Y_nomlz,basis_function,x_number);

% initialization predict function
predict_function=@(predict_x) interpolationRadialBasisPredictor...
    (X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    beta,basis_function,predict_x);

radialbasis_model.X=X;
radialbasis_model.Y=Y;
radialbasis_model.X_normalize=X_nomlz;
radialbasis_model.Y_normalize=Y_nomlz;
radialbasis_model.radialbasis_matrix=rdibas_matrix;
radialbasis_model.inv_radialbasis_matrix=inv_rdibas_matrix;

radialbasis_model.beta=beta;
radialbasis_model.aver_X=aver_X;
radialbasis_model.stdD_X=stdD_X;
radialbasis_model.aver_Y=aver_Y;
radialbasis_model.stdD_Y=stdD_Y;
radialbasis_model.basis_function=basis_function;

radialbasis_model.predict_function=predict_function;

    function [beta,rdibas_matrix,inv_rdibas_matrix]=interpolationRadialBasis...
            (X,Y,basis_function,x_number)
        % interpolation polynomial responed surface core function
        % calculation beta
        %
        % Copyright 2022 Adel
        %
        rdibas_matrix=zeros(x_number);
        for rank_index__=1:x_number
            for colume_index__=1:rank_index__-1
                rdibas_matrix(rank_index__,colume_index__)=...
                    rdibas_matrix(colume_index__,rank_index__);
            end
            for colume_index__=rank_index__:x_number
                rdibas_matrix(rank_index__,colume_index__)=...
                    basis_function(X(rank_index__,:)'-X(colume_index__,:)');
            end
        end
        
        % stabilize matrix
        rdibas_matrix=rdibas_matrix+eye(x_number)*1e-6;
        
        inv_rdibas_matrix=inv(rdibas_matrix);
        beta=inv_rdibas_matrix*Y;
    end
    function [predict_y]=interpolationRadialBasisPredictor...
            (X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            beta,basis_function,predict_x)
        % radial basis function interpolation predict function
        % input predict_x and radialbasis_model model
        % predict_x is row vector
        % output the predict value
        %
        % Copyright 2022 Adel
        %
        if size(predict_x,2) > 1
            predict_x=predict_x';
        end
        
        [x_number__,~]=size(X_nomlz);
        
        % normalize data
        predict_x=(predict_x-aver_X')./stdD_X';
        
        % predict value
        X_inner_product__=zeros(x_number__,1);
        for index_i=1:x_number__
            X_inner_product__(index_i,1)=...
                basis_function(X_nomlz(index_i,:)'-predict_x);
        end
        
        % predict variance
        predict_y=beta'*X_inner_product__;
        
        % normalize data
        predict_y=predict_y*stdD_Y+aver_Y;
    end
end

function interpolationVisualize...
    (model,low_bou,up_bou,figure_handle)
% visualization polynamial respond surface model
% figrue is 100
%
% Copyright 2022 Adel
%
if nargin < 4
    figure_handle=figure(101);
    if nargin < 3
        up_bou=[];
        if nargin < 2
            low_bou=[];
        end
    end
end
if size(low_bou,1) ~= size(low_bou,1)
    error('interpolationRadialBasisVisualize: boundary incorrect');
end
if size(low_bou,1) > 2
    error('interpolationRadialBasisVisualize: dimension large than two');
end

delete(figure_handle.Children);
axes_handle=axes(figure_handle);

x_list=model.X;
y_list=model.Y;
predict_function=model.predict_function;

% get boundary
if isempty(low_bou)
    low_bou=min(x_list,[],1)';
end
if isempty(up_bou)
    up_bou=max(x_list,[],1)';
end

grid_number=100;
d_bou=(up_bou-low_bou)/grid_number;

if size(x_list,2) == 1
    predict_result=zeros(grid_number+1,1);
    X_draw=low_bou:d_bou:(low_bou+grid_number*d_bou);
    for x_index=1:grid_number+1
        predict_x=(x_index-1).*d_bou+low_bou;
        predict_result(x_index)=predict_function(predict_x);
    end
    line(axes_handle,X_draw,predict_result);
    line(axes_handle,x_list,y_list,'Marker','o','LineStyle','none');
    xlabel('X');
    ylabel('Y');
elseif size(x_list,2) == 2
    predict_result=zeros(grid_number+1);
    [X_draw,Y_draw]=meshgrid(low_bou(1):d_bou(1):(low_bou(1)+grid_number*d_bou(1)),...
        low_bou(2):d_bou(2):(low_bou(2)+grid_number*d_bou(2)));
    for x_index=1:grid_number+1
        for y_index=1:grid_number+1
            predict_x=([x_index,y_index]-1).*d_bou'+low_bou';
            predict_result(y_index,x_index)=predict_function(predict_x);
        end
    end
    surf(axes_handle,X_draw,Y_draw,predict_result,'FaceAlpha',0.5,'EdgeColor','none');
    line(axes_handle,x_list(:,1),x_list(:,2),y_list,'Marker','o','LineStyle','none');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
end
end

function [fval_list,con_list,coneq_list]=dataLibraryUpdata...
    (data_library_name,model_function,x_list)
% updata data library
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
file_optimalSurrogate_output = fopen(data_library_name,'a');
file_result = fopen('result_total.txt','a');

% updata
for x_index=1:x_number
    x=x_list(x_index,:)';
    [fval,con,coneq]=model_function(x);
    fval_list=[fval_list;fval];
    con_list=[con_list;con'];
    coneq_list=[coneq_list;coneq'];
    
    % write data to txt_optimalSurrogateSADEKTS
    fprintf(file_optimalSurrogate_output,'%d ',variable_number);
    fprintf(file_optimalSurrogate_output,'%d ',length(con));
    fprintf(file_optimalSurrogate_output,'%d ',length(coneq));
    fprintf(file_optimalSurrogate_output,x_format,x);
    fprintf(file_optimalSurrogate_output,fval_format_base,fval);
    fval_format=repmat(fval_format_base,1,length(con));
    fprintf(file_optimalSurrogate_output,fval_format,con);
    fval_format=repmat(fval_format_base,1,length(coneq));
    fprintf(file_optimalSurrogate_output,fval_format,coneq);
    fprintf(file_optimalSurrogate_output,'\n');
    
    % write data to txt_result
    fprintf(file_result,'%d ',variable_number);
    fprintf(file_result,'%d ',length(con));
    fprintf(file_result,'%d ',length(coneq));
    fprintf(file_result,x_format,x);
    fprintf(file_result,fval_format_base,fval);
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
            con_number=data(2);
            coneq_number=data(3);
            
            x=data(4:3+variable_number);
            judge=sum(x < low_bou')+sum(x > up_bou');
            if ~judge
                x_list=[x_list;x];
                fval_list=[fval_list;data(4+variable_number)];
                con=data(5+variable_number:4+variable_number+con_number);
                if ~isempty(con)
                    con_list=[con_list;con];
                end
                coneq=data(6+variable_number+con_number:5+variable_number+con_number+coneq_number);
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

function [X,X_new,distance_min_nomlz]=getLatinHypercube...
    (sample_number,variable_number,X_exist,...
    low_bou,up_bou,cheapcon_function)
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
    cheapcon_function=[];
    if nargin < 5
        if nargin < 3
            X_exist=[];
            if nargin < 2
                error('getLatinHypercube: lack variable_number');
            end
        end
        low_bou=zeros(variable_number,1);
        up_bou=ones(variable_number,1);
    end
end
iteration_max=20*variable_number;

% check x_exist_list if meet boundary
if ~isempty(X_exist)
    index=find(X_exist < low_bou');
    index=[index,find(X_exist > up_bou')];
    if ~isempty(index)
        error('getLatinHypercube: x_exist_list range error');
    end
    if size(X_exist,2) ~= variable_number
        error('getLatinHypercube: x_exist_list variable_number error');
    end
    X_exist_nomlz=(X_exist-low_bou')./(up_bou'-low_bou');
else
    X_exist_nomlz=[];
end

% check x_new_number
x_new_number=sample_number-size(X_exist,1);
if x_new_number < 0
    X=X_exist;
    X_new=[];
    distance_min_nomlz=getMinDistance(X_exist_nomlz);
    return;
end

low_bou_nomlz=zeros(variable_number,1);
up_bou_nomlz=ones(variable_number,1);

% get initial X_new_nomalize by lhsdesign
X_new_nomlz=lhsdesign(x_new_number,variable_number);
distance_min_nomlz=getMinDistance([X_new_nomlz;X_exist_nomlz]);

% x is nomalize, so constraint function should change
if ~isempty(cheapcon_function)
    cheapcon_function=@(x) ...
        max(max(sample_number*cheapcon_function(x.*(up_bou-low_bou)+low_bou)+1,0),[],1);
end

iteration=0;
while iteration < iteration_max
    %     scatter(X_new_nomlz(:,1),X_new_nomlz(:,2));
    %     bou=[low_bou_nomlz,up_bou_nomlz]';
    %     axis(bou(:));axis equal
    %     radius=1/6;
    %     hold on;
    %     rectangle('Position',[-radius+0.5,-radius+0.5,2*radius,2*radius],'Curvature',[1 1])
    %     hold off;
    
    % change each x place by newton methods
    fval_list=zeros(x_new_number,1);
    gradient_list=zeros(variable_number,x_new_number);
    for x_new_index=1:x_new_number
        % get gradient
        [fval_list(x_new_index,1),gradient_list(:,x_new_index),~]=objectFunctionXPlace...
            (X_new_nomlz(x_new_index,:)',[X_new_nomlz(1:x_new_index-1,:);X_new_nomlz(x_new_index+1:end,:);X_exist_nomlz],...
            variable_number,low_bou_nomlz,up_bou_nomlz,cheapcon_function);
    end
    
    % normalize fval
    fval_list=fval_list/max(fval_list);
    for x_new_index=1:x_new_number
        x=X_new_nomlz(x_new_index,:)'-...
            fval_list(x_new_index,1)*gradient_list(:,x_new_index)/norm(gradient_list(:,x_new_index),2)*...
            distance_min_nomlz*(1-iteration/iteration_max);
        for variable_index=1:variable_number
            if x(variable_index,1) > 1
                x(variable_index,1)=1;
            end
            if x(variable_index,1) < 0
                x(variable_index,1)=0;
            end
        end
        X_new_nomlz(x_new_index,:)=x';
    end
    
    iteration=iteration+1;
end
distance_min_nomlz=getMinDistance([X_new_nomlz;X_exist_nomlz]);
X_new=X_new_nomlz.*(up_bou'-low_bou')+low_bou';
X=[X_new;X_exist];

    function [fval,gradient,hessian]=objectFunctionXPlace...
            (x,X_surplus,variable_number,low_bou,up_bou,cheapcon_function)
        % function describe distance between X and X_supply
        % x is colume vector and X_surplus is matrix which is num-1 x var
        % low_bou_limit__ and up_bou_limit__ is colume vector
        % variable in colume
        %
        [~,variable_number__]=size(X_surplus);
        X_surplus=X_surplus';
        
        sigma__=10;
        boundary__=0.1^variable_number__;
        
        sign__=((x>X_surplus)-0.5)*2;
        
        xi__=-sigma__*(x-X_surplus).*sign__;
        sum_xi__=sum(xi__,1);
        psi__=sigma__*(low_bou-x);
        zeta__=sigma__*(x-up_bou);
        
        exp_sum_xi__=exp(sum_xi__);
        exp_psi__=exp(psi__);
        exp_zeta__=exp(zeta__);
        
        xi_DF=-sigma__*sign__;
        % sum_xi_DF=sum(xi_DF,2);
        psi_DF=-sigma__*ones(variable_number__,1);
        zeta_DF=sigma__*ones(variable_number__,1);
        
        % get fval
        fval=sum(exp_sum_xi__,2)+...
            sum(boundary__*exp_psi__+...
            boundary__*exp_zeta__,1);
        
        % get gradient
        gradient=sum(exp_sum_xi__.*xi_DF,2)+...
            boundary__*exp_psi__.*psi_DF+...
            boundary__*exp_zeta__.*zeta_DF;
        
        % get hessian
        hessian=exp_sum_xi__.*xi_DF*xi_DF'+...
            diag(boundary__*exp_psi__.*psi_DF.*psi_DF+...
            boundary__*exp_zeta__.*zeta_DF.*zeta_DF);
        
        if ~isempty(cheapcon_function)
            fval_con=cheapcon_function(x);
            fval=fval+fval_con;
            [gradient_con,hessian_con]=differ...
                (cheapcon_function,x,fval_con,variable_number);
            gradient=gradient+gradient_con;
            hessian=hessian+hessian_con;
        end
        
        function [gradient,hessian]=differ(differ_function,x,fval,variable_number,step)
            % differ function to get gradient and hessian
            %
            if nargin < 5
                step=1e-6;
            end
            fval__=zeros(variable_number,2); % backward is 1, forward is 2
            gradient=zeros(variable_number,1);
            hessian=zeros(variable_number);
            
            % fval and gradient
            for variable_index__=1:variable_number
                x_forward__=x;
                x_backward__=x;
                x_backward__(variable_index__)=x_backward__(variable_index__)-step;
                fval__(variable_index__,1)=differ_function(x_backward__);
                
                x_forward__(variable_index__)=x_forward__(variable_index__)+step;
                fval__(variable_index__,2)=differ_function(x_forward__);
                
                gradient(variable_index__)=...
                    (fval__(variable_index__,2)-fval__(variable_index__,1))/2/step;
            end
            
            % hessian
            for variable_index__=1:variable_number
                hessian(variable_index__,variable_index__)=...
                    (fval__(variable_index__,2)-2*fval+fval__(variable_index__,1))/step/step;
                for variable_index_next__=variable_index__+1:variable_number
                    x_for_for=x;
                    x_for_for(variable_index__)=x_for_for(variable_index__)+step;
                    x_for_for(variable_index_next__)=x_for_for(variable_index_next__)+step;
                    
                    hessian(variable_index__,variable_index_next__)=(...
                        differ_function(x_for_for)-...
                        fval__(variable_index__,2)-fval__(variable_index_next__,2)+...
                        fval...
                        )/step/step;
                    hessian(variable_index_next__,variable_index__)=...
                        hessian(variable_index__,variable_index_next__);
                end
            end
        end
    end
    function distance_min__=getMinDistance(x_list__)
        % get distance min from x_list
        %
        
        % sort x_supply_list_initial to decrese distance calculate times
        x_list__=sortrows(x_list__,1);
        sample_number__=size(x_list__,1);
        variable_number__=size(x_list__,2);
        distance_min__=variable_number__;
        for x_index__=1:sample_number__
            x_curr__=x_list__(x_index__,:);
            x_next_index__=x_index__ + 1;
            % first dimension only search in min_distance
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

function drawFunction(object_function,low_bou,up_bou,...
    grid_number,Y_min,Y_max,figure_handle)
if nargin < 7
    figure_handle=figure(10);
    if nargin < 6
        Y_max=inf;
        if nargin < 5
            Y_min=-inf;
            if nargin < 4
                grid_number=100;
            end
        end
    end
end
axes_handle=figure_handle.CurrentAxes;
if isempty(axes_handle)
    axes_handle=axes(figure_handle);
end
axes_context=axes_handle.Children;
dimension=size(low_bou,1);

switch dimension
    case 1
        d_bou=(up_bou-low_bou)/grid_number;
        X__=low_bou:d_bou:up_bou;
        fval__=zeros(grid_number+1,1);
        for x_index__=1:(grid_number+1)
            fval__(x_index__)=object_function(X__(x_index__));
        end
        line(axes_handle,X__,fval__);
        xlabel('X');
        ylabel('value');
        
    case 2
        d_bou=(up_bou-low_bou)/grid_number;
        [X__,Y]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
        fval__=zeros(grid_number+1);
        for x_index__=1:grid_number+1
            for y_index__=1:grid_number+1
                predict_x=([x_index__;y_index__]-1).*d_bou+low_bou;
                fval__(x_index__,y_index__)=object_function(predict_x);
            end
        end
        fval__(find(fval__ > Y_max))=Y_max;
        fval__(find(fval__ < Y_min))=Y_min;
        axes_context=[axes_context;surface(X__',Y',fval__,'FaceAlpha',0.5,'EdgeColor','none')];
        axes_handle.set('Children',axes_context);
        xlabel('X');
        ylabel('Y');
        zlabel('value');
end
end