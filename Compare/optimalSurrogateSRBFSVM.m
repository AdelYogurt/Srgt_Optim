function [x_best,fval_best,NFE,output]=optimalSurrogateSRBFSVM...
    (object_function,variable_number,low_bou,up_bou,nonlcon_function,...
    cheapcon_function,model_function,NFE_max)
% surrogate base optimal use radias base function method version 1
% matlab internal svm was used
% use SVM to get interest point
% FS_FCM to get interest point center point
% and updata interest space
% all function exchange x should be colume vector
% x_list is x_number x variable_number matrix
% both nonlcon_function and cheapcon_function format is [con,coneq]
% model_function should output fval, format is [fval,con,coneq]
% con or coneq can be colume vector if there was more than one constrain
%
% Copyright 2022 Adel
%
if nargin < 8
    NFE_max=[];
    if nargin < 7
        cheapcon_function=[];
        if nargin < 6
            nonlcon_function=[];
        end
    end
end
INFORMATION_FLAG=0; % whether draw data
CONVERGENCE_JUDGMENT_FLAG=1; % whether judgment convergence

sample_number_initial=min((variable_number+1)*(variable_number+2)/2,5*variable_number);
sample_number_iteration=variable_number;
if variable_number < 10
    sample_number_data=100*sample_number_initial;
else
    sample_number_data=10*sample_number_initial;
end
eta=1/variable_number; % space decrease coefficient

kernel_function_SVM=@(x1,x2) exp(-((x1-x2)'*(x1-x2)));
C=10;
kernal_function_FS_FCM=@(sq) exp(-sq/2/0.1^2);
m=2; % clustering parameter

if isempty(NFE_max)
    NFE_max=40*variable_number;
end
iteration_max=100;
torlance=1e-3;

data_library_name='optimalSurrogate_SRBF_SVM_result.txt';
file_result = fopen('result_total.txt','a');
fprintf(file_result,'%s\n',datetime);
fclose(file_result);
clear('file_result');

done=0;NFE=0;iteration=0;
result_x_best=zeros(iteration_max,variable_number);
result_fval_best=zeros(iteration_max,1);

% if do not input model_function, generate model_function
if nargin < 11 || isempty(model_function)
    model_function=@(x) modelFunction(x,object_function,nonlcon_function);
end

iteration=iteration+1;

% step 2
% use latin hypercube method to get initial sample x_list
[~,x_list_updata,~]=getLatinHypercube...
    (sample_number_initial,variable_number,[],low_bou,up_bou,cheapcon_function);

% detech expensive constraints
if ~isempty(x_list_updata)
    [~,con,coneq]=dataLibraryUpdata...
        (data_library_name,model_function,x_list_updata(1,:));NFE=NFE+1;
    x_list_updata=x_list_updata(2:end,:);
else
    [~,con,coneq]=dataLibraryLoad(data_library_name,low_bou,up_bou);
end
if ~isempty(con) || ~isempty(coneq)
    expensive_nonlcon_flag=1;
else
    expensive_nonlcon_flag=0;
end

% import data from data library
[x_list,fval_list,con_list,coneq_list]=dataLibraryLoad...
    (data_library_name,low_bou,up_bou);

while ~done
    % step 3
    % updata data library by x_list
    [fval_list_updata,con_list_updata,coneq_list_updata]=dataLibraryUpdata...
        (data_library_name,model_function,x_list_updata);NFE=NFE+size(x_list_updata,1);
    x_list=[x_list;x_list_updata];
    fval_list=[fval_list;fval_list_updata];
    con_list=[con_list;con_list_updata];
    coneq_list=[coneq_list;coneq_list_updata];
    
    % nomalization con
    fval_max=max(abs(fval_list));
    fval_list_nomlz=fval_list./fval_max;
    if ~isempty(con_list)
        con_max_list=max(abs(con_list),[],1);
        con_list_nomlz=con_list./con_max_list;
    else
        con_list_nomlz=[];
    end
    if ~isempty(coneq_list)
        coneq_max_list=max(abs(coneq_list),[],1);
        coneq_list_nomlz=coneq_list./coneq_max_list;
    else
        coneq_list_nomlz=[];
    end
    
    % step 4
    % generate ERBF_QP model
    [ERBF_model_fval,ERBF_model_con,ERBF_model_coneq,output_ERBF]=getEnsemleRadialBasisModel...
        (x_list,fval_list,con_list,coneq_list);
    object_function=output_ERBF.object_function;
    nonlcon_function=output_ERBF.nonlcon_function;
    
    % step 5
    % MSP guideline to obtain x_adapt
    [x_potential,~]=findMinMSP...
        (object_function,variable_number,low_bou,up_bou,nonlcon_function,...
        cheapcon_function);
    
    % check x_potential if exist in data library
    % if not, updata data libraray
    distance=sum(((x_potential'-x_list)./(low_bou'-up_bou')).^2,2);
    [~,min_index]=min(distance);
    if distance(min_index)<1e-3
        x_potential=x_list(min_index,:)';
        fval_potential=fval_list(min_index,:);
        con_potential=[];
        coneq_potential=[];
        if ~isempty(con_list)
            con_potential=(con_list(min_index,:)/fval_max.*con_max_list)';
        end
        if ~isempty(coneq_list)
            coneq_potential=(coneq_list(min_index,:)/fval_max.*coneq_max_list)';
        end
    else
        [fval_potential,con_list_potential,coneq_list_potential]=dataLibraryUpdata...
            (data_library_name,model_function,x_potential');NFE=NFE+1;
        x_list=[x_list;x_potential'];
        fval_list=[fval_list;fval_potential];
        con_list=[con_list;con_list_potential];
        coneq_list=[coneq_list;coneq_list_potential];
    end
    
    if INFORMATION_FLAG && variable_number < 3
        interpolationVisualize(ERBF_model_fval,low_bou,up_bou);
        line(x_potential(1),x_potential(2),fval_potential,'Marker','o','color','r','LineStyle','none')
    end
    
    % step 6
    % forced interrupt
    if iteration >= iteration_max || NFE >= NFE_max
        done=1;
    end
    
    % convergence judgment
    if CONVERGENCE_JUDGMENT_FLAG
        if (iteration > 1 && ...
                abs((fval_potential-fval_potential_old)/fval_potential_old) < torlance)
            done=1;
            if ~isempty(con_best)
                if sum(abs(coneq_best) > torlance)
                    done=0;
                end
            end
            if ~isempty(coneq_best)
                if sum(abs(coneq_best) > torlance)
                    done=0;
                end
            end
        end
    end
    
    % step 7
    % interset sampling
    % step 7-1
    % classify exist data
    fval_thresh=min(fval_list)+eta*(max(fval_list)-min(fval_list));
    fval_label=zeros(size(x_list,1),1);
    for x_index=1:size(x_list,1)
       if fval_list(x_index) <= fval_thresh
           fval_label(x_index)=1;
       else
           fval_label(x_index)=0;
       end
    end
    
    % step 7-2
    % get a large number of x point, use SVM to predict x point
    x_data_list=lhsdesign(sample_number_data,variable_number).*...
        (up_bou'-low_bou')+low_bou';
    x_sup_list=[];
    SVM_model = fitcsvm(x_list,fval_label,'ClassNames',[0 1],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
    for x_index=1:sample_number_data
        if  predict(SVM_model,x_data_list(x_index,:)) == 1
            x_sup_list=[x_sup_list;x_data_list(x_index,:)];
        end
    end
    
    while size(x_sup_list,1) < 5*variable_number
        fval_thresh=fval_thresh+0.1*(max(fval_list)-min(fval_list));
        fval_label=zeros(size(x_list,1),1);
        for x_index=1:size(x_list,1)
            if fval_list(x_index) <= fval_thresh
                fval_label(x_index)=1;
            else
                fval_label(x_index)=-1;
            end
        end
        
        if sum(fval_label) == size(x_list,1)
           disp('error') 
        end
        
        SVM_model = fitcsvm(x_list,fval_label,'ClassNames',[0 1],'Standardize',true,...
            'KernelFunction','rbf','BoxConstraint',1);
        for x_index=1:sample_number_data
            if  predict(SVM_model,x_data_list(x_index,:)) == 1
                x_sup_list=[x_sup_list;x_data_list(x_index,:)];
            end
        end
    end
    
    % step 7-3
    % calculate clustering center
    FC_model=classifyFuzzyClusteringFeatureSpace...
        (x_sup_list,1,low_bou,up_bou,m);
    x_center=FC_model.center_list';
    x_potential_nomlz=(x_potential-low_bou)./(up_bou-low_bou);
    x_center_nomlz=(x_center-low_bou)./(up_bou-low_bou);
    
    % updata ISR
    bou_range=eta*norm(x_potential_nomlz-x_center_nomlz,2);
    if bou_range < 0.001
        bou_range=0.001;
    end
    bou_range=bou_range.*(up_bou-low_bou);
    low_bou_ISR=x_potential-bou_range;
    index=find(low_bou_ISR<low_bou);
    low_bou_ISR(index)=low_bou(index);
    up_bou_ISR=x_potential+bou_range;
    index=find(up_bou_ISR>up_bou);
    up_bou_ISR(index)=up_bou(index);
    
    if INFORMATION_FLAG && variable_number < 3
        bou_line=[low_bou_ISR,[low_bou_ISR(1);up_bou_ISR(2)],up_bou_ISR,[up_bou_ISR(1);low_bou_ISR(2)],low_bou_ISR];
        line(bou_line(1,:),bou_line(2,:));
    end
    
    % sampling in ISR
    [x_list_exist,~,~,~]=dataLibraryLoad...
        (data_library_name,low_bou_ISR,up_bou_ISR);
    [~,x_list_updata,~]=getLatinHypercube...
        (sample_number_iteration+size(x_list_exist,1),variable_number,x_list_exist,...
        low_bou_ISR,up_bou_ISR,cheapcon_function);
    
    % find best result to record
    [x_best,fval_best,con_best,coneq_best]=findMinRaw...
        (x_list,fval_list,con_list,coneq_list,...
        cheapcon_function);
    result_x_best(iteration,:)=x_best';
    result_fval_best(iteration,:)=fval_best;
    iteration=iteration+1;
    
    x_potential_old=x_potential;
    fval_potential_old=fval_potential;
    fval_best_old=fval_best;
end
result_x_best=result_x_best(1:iteration-1,:);
result_fval_best=result_fval_best(1:iteration-1);

output.result_x_best=result_x_best;
output.result_fval_best=result_fval_best;

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
end

function [x_best,fval_best]=findMinMSP...
    (object_function,variable_number,low_bou,up_bou,nonlcon_function,...
    cheapcon_function)
% find min fval use MSP guideline
% MSP: object_funtion is object_function (generate by surrogate model)
% nonlcon_function generate by surrogate model
% use ga as optimal method
%
if ~isempty(nonlcon_function) && ~isempty(cheapcon_function)
    totalcon_function=@(x) totalconFunction(x,nonlcon_function,cheapcon_function);
elseif isempty(nonlcon_function) && ~isempty(cheapcon_function)
    totalcon_function=cheapcon_function;
elseif ~isempty(nonlcon_function) && isempty(cheapcon_function)
    totalcon_function=nonlcon_function;
else
    totalcon_function=[];
end

% generate initial population for ga
population_matrix=zeros(2*variable_number,variable_number);
for population_index=1:2*variable_number
    x=rand(variable_number,1).*(up_bou-low_bou)+low_bou;
    if ~isempty(totalcon_function)
        [con,coneq]=totalcon_function(x);
        while sum(~(con < 0))
            x=rand(variable_number,1).*(up_bou-low_bou)+low_bou;
            [con,coneq]=totalcon_function(x);
        end
    end
    population_matrix(population_index,:)=x';
end

% optiaml
ga_option=optimoptions('ga','FunctionTolerance',1e-2,...
    'PopulationSize',max(10,2*variable_number),...
    'MaxGenerations',50,'InitialPopulationMatrix',population_matrix,...
    'display','none');
[x_best,fval_best,~,~]=ga...
    (object_function,variable_number,[],[],[],[],low_bou',up_bou',totalcon_function,ga_option);
x_best=x_best';
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
end
function [x_best,fval_best,con_best,coneq_best]=findMinRaw...
    (x_list,fval_list,con_list,coneq_list,...
    cheapcon_function)
% find min fval in raw data
% x_list, rank is variable
% con_list, rank is con
% coneq_list, rank is coneq
% function will find min fval in con==0
% if there was not feasible x, find min consum
%
con_sum_list=zeros(size(x_list,1),1);
% add expendsive con
con_best=[];
coneq_best=[];
if ~isempty(con_list)
    con_sum_list=con_sum_list+sum(max(con_list,0),2);
end
if ~isempty(coneq_list)
    con_sum_list=con_sum_list+sum(coneq_list.^2,2);
end

% add cheap con
for x_index=1:size(x_list,1)
    if ~isempty(cheapcon_function)
        [con,coneq]=cheapcon_function(x_list(x_index,:)');
        con_sum_list(x_index)=con_sum_list(x_index)+...
            sum(max(con,0))+sum(coneq.*coneq);
    end
end

index=find(con_sum_list<=0);
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
    [~,index_best]=min(con_sum_list);
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

function FC_model=classifyFuzzyClusteringFeatureSpace...
    (X,classify_number,low_bou,up_bou,m,kernal_function)
% get fuzzy cluster model with feature space
% kernal function recommend kernal_function=@(sq) exp(-sq/2*1000);
% X is x_number x variable_number matrix
% center_list is classify_number x variable_number matrix
%
if nargin < 6
    kernal_function=[];
    if nargin < 4
        up_bou=[];
        if nargin < 3
            low_bou=[];
        end
    end
end
iteration_max=100;
torlance=1e-3;

[x_number,variable_number]=size(X);

if isempty(kernal_function)
    kernal_function=@(sq) exp(-sq/2*1000);
end

% nomalization data
if isempty(low_bou)
    low_bou=min(X,[],1)';
end
if isempty(up_bou)
    up_bou=max(X,[],1)';
end
X_nomlz=(X-low_bou')./(up_bou'-low_bou');

% if x_number equal 1, clustering cannot done
if x_number==1
    FC_model.X=X;
    FC_model.X_normalize=X_nomlz;
    FC_model.center_list=X;
    FC_model.fval_loss_list=[];
    return;
end

U=zeros(classify_number,x_number);
center_list=rand(classify_number,variable_number)*0.5;
iteration=0;
done=0;
fval_loss_list=zeros(iteration_max,1);

% get X_center_dis_sq
X_center_dis_sq=zeros(classify_number,x_number);
for classify_index=1:classify_number
    for x_index=1:x_number
        X_center_dis_sq(classify_index,x_index)=...
            getSq((X_nomlz(x_index,:)-center_list(classify_index,:)));
    end
end

while ~done
    % updata classify matrix U
    for classify_index=1:classify_number
        for x_index=1:x_number
            U(classify_index,x_index)=...
                1/sum(((2-2*kernal_function(X_center_dis_sq(classify_index,x_index)))./...
                (2-2*kernal_function(X_center_dis_sq(:,x_index)))).^(1/(m-1)));
        end
    end
    
    % updata center_list
    center_list_old=center_list;
    for classify_index=1:classify_number
        center_list(classify_index,:)=...
            sum((U(classify_index,:)').^m.*X_nomlz,1)./...
            sum((U(classify_index,:)').^m,1);
    end
    
    % updata X_center_dis_sq
    X_center_dis_sq=zeros(classify_number,x_number);
    for classify_index=1:classify_number
        for x_index=1:x_number
            X_center_dis_sq(classify_index,x_index)=...
                getSq((X_nomlz(x_index,:)-center_list(classify_index,:)));
        end
    end
    
    % forced interrupt
    if iteration > iteration_max
        done=1;
    end
    
    % convergence judgment
    if sum(sum(center_list_old-center_list).^2)<torlance
        done=1;
    end
    
    iteration=iteration+1;
    fval_loss_list(iteration)=sum(sum(U.^m.*(2-2*kernal_function(X_center_dis_sq))));
end
fval_loss_list(iteration+1:end)=[];
center_list=center_list.*(up_bou'-low_bou')+low_bou';

FC_model.X=X;
FC_model.X_normalize=X_nomlz;
FC_model.center_list=center_list;
FC_model.fval_loss_list=fval_loss_list;

    function sq=getSq(dx)
        % dx is 1 x variable_number matrix
        %
        sq=dx*dx';
    end
end

function [SVM_predict_function,SVM_model]=classifySupportVectorMachine...
    (X,class,C,kernel_function,low_bou,up_bou)
% generate support vector machine model version 0
% version 0 use fmincon to get alpha
% only support binary classification, 0 and 1
% X, Y is x_number x variable_number matrix
% C is penalty factor, default is empty
% kernel_function default is gauss kernal function
% kernel_function should be @(x1,x2) ...
%
if nargin < 4
    kernel_function=[];
    if nargin < 3
        C=[];
    end
end

[x_number,~]=size(X);

% normalization data
if nargin < 6
    up_bou=max(X)';
    if nargin < 5
        low_bou=min(X)';
    end
end
X_nomlz=(X-low_bou')./(up_bou'-low_bou');

% transfer class into y
Y=class;
for x_index=1:x_number
    if Y(x_index) == 0
        Y(x_index)=-1;
    end
end

% default kernal function
if isempty(kernel_function)
    kernel_function=@(x1,x2) exp(-((x1-x2)'*(x1-x2))*10);
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

% min ||w||^2/2 use SMO algorithm
alpha_initial=C*rand(x_number,1);
alpha=optimalSMO(alpha_initial,X_cov,Y,C);

% obtain other paramter
w=sum(alpha.*Y.*X_nomlz);
index_list=find(alpha > 1e-6);

alpha_Y=alpha.*Y;
alpha_Y_cov=X_cov*alpha_Y;
b=sum(Y(index_list)-alpha_Y_cov(index_list))/length(index_list);

% generate predict function
SVM_predict_function=@(x) classifySupportVectorMachinePredictor...
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
SVM_model.SVM_predict_function=SVM_predict_function;

    function alpha=optimalSMO(alpha_initial,X_cov,Y,C)
        % SMO algorithm to optain alpha
        %
        iteration_max__=100;
        torlance__=1e-6;
        
        % initialization
        iteration__=0;
        done__=0;
        alpha=alpha_initial;
        x_number__=size(X_cov,1);
        alpha_Y__=alpha.*Y;
        alpha_Y_cov__=X_cov*alpha_Y__;
        b__=sum(Y-alpha_Y_cov__)/x_number__;
        Y_g_x__=Y.*(alpha_Y_cov__+b__);
        E__=alpha_Y_cov__+b__-Y;
        
        iteration__=iteration__+1;
        fval_old__=sum(alpha)-0.5*alpha_Y__'*X_cov*alpha_Y__;
        
        while ~done__
            % external loop choose max violation alpha
            index_1__=[];
            index_in_list__=[];
            index_out_list__=[];
            for x_index__=1:x_number__
                alpha_unit=alpha(x_index__);
                if (alpha_unit > torlance__) && (alpha_unit < C-torlance__)
                    index_in_list__=[index_in_list__;x_index__];
                else
                    index_out_list__=[index_out_list__;x_index__];
                end
            end
            if ~isempty(index_in_list__)
                [~,index_1__]=max(abs(Y_g_x__(index_in_list__)-1));
                index_1__=index_in_list__(index_1__(randi(length(index_1__))));
            end
            if isempty(index_1__)
               [~,index_1__]=max(abs(Y_g_x__(index_out_list__)-1));
                index_1__=index_out_list__(index_1__(randi(length(index_1__))));
            end
            if isempty(index_1__)
               index_1__=randi(x_number__);
            end
            
            % internal loop choose max change
            if E__(index_1__) < 0
                [~,index_2__]=max(E__);
            end
            if E__(index_1__) > 0
                [~,index_2__]=min(E__);
            end
            
            % initialization bou
            if Y(index_1__) ~= Y(index_2__)
                low_bou__=max(0,alpha(index_2__)-alpha(index_1__));
                up_bou__=min(C,C+alpha(index_2__)-alpha(index_1__));
            else
                low_bou__=max(0,alpha(index_2__)+alpha(index_1__)-C);
                up_bou__=min(C,alpha(index_2__)+alpha(index_1__));
            end
            
            % updata alpha_2_unc
            alpha_2_unc__=alpha(index_2__)+...
                Y(index_2__)*(E__(index_1__)-E__(index_2__))/...
                (X_cov(index_1__,index_1__)+X_cov(index_2__,index_2__)-X_cov(index_1__,index_2__));
            
            % updata alpha_2
            if alpha_2_unc__ > up_bou__
                alpha_2__=up_bou__;
            elseif alpha_2_unc__ < low_bou__
                alpha_2__=low_bou__;
            else
                alpha_2__=alpha_2_unc__;
            end
            
            % updata alpha_1
            alpha_1__=alpha(index_1__)+(alpha(index_2__)-alpha_2__)*Y(index_1__)*Y(index_2__);
            
            %             sum_alpha_Y_remain__=0;
            %             for x_index__=1:x_number__
            %                 if x_index__ ~= index_1__ && x_index__ ~= index_2__
            %                     sum_alpha_Y_remain__=sum_alpha_Y_remain__+alpha_Y__(x_index__);
            %                 end
            %             end
            %             alpha(index_1__)=Y(index_1__)*(-sum_alpha_Y_remain__-alpha_Y__(index_2__));
            
            b_1__=-E__(index_1__)-Y(index_1__)*X_cov(index_1__,index_1__)*(alpha_1__-alpha(index_1__))-...
                Y(index_2__)*X_cov(index_2__,index_1__)*(alpha_2__-alpha(index_2__))+b__;
            b_2__=-E__(index_2__)-Y(index_1__)*X_cov(index_1__,index_2__)*(alpha_1__-alpha(index_1__))-...
                Y(index_2__)*X_cov(index_2__,index_2__)*(alpha_2__-alpha(index_2__))+b__;
            alpha(index_1__)=alpha_1__;
            alpha(index_2__)=alpha_2__;
            
            alpha_Y__=alpha.*Y;
            alpha_Y_cov__=X_cov*alpha_Y__;
            b__=(b_1__+b_2__)/2;
            %             b__=sum(Y-alpha_Y_cov__)/x_number__;
            Y_g_x__=Y.*(alpha_Y_cov__+b__);
            E__=alpha_Y_cov__+b__-Y;
            
            fval__=sum(alpha)-0.5*alpha_Y__'*X_cov*alpha_Y__;
            
            if abs((fval__-fval_old__)/fval_old__) < torlance__
                done__=1;
            end
            
            if iteration__ >= iteration_max__
                done__=1;
            end
            
            fval_old__=fval__;
            iteration__=iteration__+1;
        end
    end
    function [predict_class,predict_fval]=classifySupportVectorMachinePredictor...
            (x,X_nomlz,Y,alpha,b,low_bou,up_bou,kernel_function)
        % predict value of x is 1 or -1
        % x input is colume vector
        %
        x_number__=size(X_nomlz,1);
        x_nomlz=(x-low_bou)./(up_bou-low_bou);
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
function classifySupportVectorMachineVisualization...
    (SVM_model,low_bou,up_bou)
% Visualization SVM_model
%
if nargin < 1
    error('classifySupportVectorMachineVisualization: not enough input');
end
X=SVM_model.X;
Y=SVM_model.Y;
% nomlz_X=SVM_model.nomlz_X;
% alpha=SVM_model.alpha;
% w=SVM_model.w;
% b=SVM_model.b;
% kernel_function=SVM_model.kernel_function;
SVM_predict_function=SVM_model.SVM_predict_function;

if nargin < 3
    up_bou=max(X)';
    if nargin < 2
        low_bou=min(X)';
    end
end

% check dimension
[~,variable_number]=size(X);
if variable_number ~= 2
    error('classifySupportVectorMachineVisualization: dimension do not equal 2');
end

X_positive=X(find(Y>0),:);
X_negative=X(find(Y<0),:);

% draw zero value line
grid_number=100;
d_bou=(up_bou-low_bou)/grid_number;
[X_draw,Y_draw]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
predict_class=zeros(grid_number+1);
predict_fval=zeros(grid_number+1);
for x_index=1:grid_number+1
    for y_index=1:grid_number+1
        predict_x=([x_index;y_index]-1).*d_bou+low_bou;
        [predict_class(y_index,x_index),predict_fval(y_index,x_index)]=...
            SVM_predict_function(predict_x);
    end
end
contour(X_draw,Y_draw,predict_class);
% contour(X_draw,Y_draw,predict_fval,[0.5,0.5]);

% draw point
line(X_positive(:,1),X_positive(:,2),'LineStyle','none','Marker','o','Color','b');
line(X_negative(:,1),X_negative(:,2),'LineStyle','none','Marker','o','Color','r');

% figure(2)
% surf(X,Y,fval_sum);
end

function [ERBF_model_fval,ERBF_model_con,ERBF_coneq,output]=getEnsemleRadialBasisModel...
    (x_list,fval_list,con_list,coneq_list)
% base on library_data to create kriging model and function
% object_function is single fval output
% nonlcon_function is normal nonlcon_function which include con, coneq
% con is colume vector, coneq is colume vector
%
if nargin < 7
    ERBF_coneq=[];
    if nargin < 6
        ERBF_model_con=[];
        if nargin < 5
            ERBF_model_fval=[];
        end
    end
end
% generate surrogate model
ERBF_model_fval=interpolationEnsemleRadialBasisPreModel...
    (x_list,fval_list);

if ~isempty(con_list)
    ERBF_model_con=struct('X',[],'Y',[],'X_normalize',[],'Y_normalize',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],...
        'predict_function',[]);
    ERBF_model_con=repmat(ERBF_model_con,[size(con_list,2),1]);
    for con_index=1:size(con_list,2)
        ERBF_model_con(con_index)=interpolationEnsemleRadialBasisPreModel...
            (x_list,con_list(:,con_index));
    end
end

if ~isempty(coneq_list)
    ERBF_coneq=struct('X',[],'Y',[],'X_normalize',[],'Y_normalize',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],...
        'predict_function',[]);
    ERBF_coneq=repmat(ERBF_coneq,[size(coneq_list,2),1]);
    for coneq_index=1:size(coneq_list,2)
        ERBF_coneq(coneq_index)=interpolationEnsemleRadialBasisPreModel...
            (x_list,coneq_list(:,coneq_index));
    end
end

object_function=@(predict_x) objectFunctionSurrogate(predict_x,ERBF_model_fval);
if isempty(ERBF_model_con) && isempty(ERBF_coneq)
    nonlcon_function=[];
else
    nonlcon_function=@(predict_x) nonlconFunctionSurrogate(predict_x,ERBF_model_con,ERBF_coneq);
end

output.object_function=object_function;
output.nonlcon_function=nonlcon_function;
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
function ensemleradialbasis_model=interpolationEnsemleRadialBasisPreModel...
    (X,Y)
% get ensemle radial basis function interpolation model function
% input initial data X, Y, which are real data
% X, Y are x_number x variable_number matrix
% aver_X,stdD_X is 1 x x_number matrix
% output is a radial basis model, include X, Y, base_function
% and predict_function
% beta is normalizede, so predict y is normalizede
%
% Copyright 2022 Adel
%
[x_number,variable_number]=size(X);
c_initial=(2/x_number)^(1/variable_number);

% normalize data
aver_X = mean(X);
stdD_X = std(X);
aver_Y = mean(Y);
stdD_Y = std(Y);
index = find(stdD_X == 0);
if ~isempty(index),stdD_X(index)=1;end
index = find(stdD_Y == 0);
if ~isempty(index),stdD_Y(index)=1;end
X_nomlz = (X - repmat(aver_X,x_number,1)) ./ repmat(stdD_X,x_number,1);
Y_nomlz = (Y - repmat(aver_Y,x_number,1)) ./ repmat(stdD_Y,x_number,1);

% initial sq of x and x
x_sq_matrix=zeros(x_number);
for rank_index=1:x_number
    for colume_index=1:rank_index-1
        x_sq_matrix(rank_index,colume_index)=...
            x_sq_matrix(colume_index,rank_index);
    end
    for colume_index=rank_index:x_number
        x_sq_matrix(rank_index,colume_index)=...
            sum((X_nomlz(rank_index,:)'-X_nomlz(colume_index,:)').^2);
    end
end

option_fmincon=optimoptions('fmincon','display','none','TolFun',1e-6,...
    'SpecifyObjectiveGradient',true);

% linear kernal function
basis_function_linear=@(x_sq,c) sqrt(x_sq)+c;
rdibas_matrix_gradient_function=@(x_number,x_sq_matrix,rdibas_matrix) ones(x_number);
object_function=@(c) objectFunctionRadiabasis....
    (x_sq_matrix,Y_nomlz,x_number,basis_function_linear,c,rdibas_matrix_gradient_function);
[linear_c_backward,fval_backward]=fmincon(object_function,-1e2,[],[],[],[],-10,10,[],option_fmincon);
[linear_c_forward,fval_forward]=fmincon(object_function,1e2,[],[],[],[],-10,10,[],option_fmincon);
if fval_forward < fval_backward
    c_linear=linear_c_forward;
else
    c_linear=linear_c_backward;
end

% gauss kernal function
basis_function_gauss=@(x_sq,c) exp(-c*x_sq);
rdibas_matrix_gradient_function=@(x_number,x_sq_matrix,rdibas_matrix) -x_sq_matrix.*rdibas_matrix;
object_function=@(c) objectFunctionRadiabasis....
    (x_sq_matrix,Y_nomlz,x_number,basis_function_gauss,c,rdibas_matrix_gradient_function);
c_gauss=fmincon(object_function,c_initial,[],[],[],[],0.1,10,[],option_fmincon);

% multiquadric kernal function
basis_function_multiquadric=@(x_sq,c) sqrt(x_sq+c);
rdibas_matrix_gradient_function=@(x_number,x_sq_matrix,rdibas_matrix) 0.5./rdibas_matrix;
object_function=@(c) objectFunctionRadiabasis....
    (x_sq_matrix,Y_nomlz,x_number,basis_function_multiquadric,c,rdibas_matrix_gradient_function);
c_binomial=fmincon(object_function,c_initial,[],[],[],[],1e-2,1e2,[],option_fmincon);

% inverse multiquadric kernal function
basis_function_inverse_multiquadric=@(x_sq,c) 1/sqrt(x_sq+c);
rdibas_matrix_gradient_function=@(x_number,x_sq_matrix,rdibas_matrix) -0.5*rdibas_matrix.^3;
object_function=@(c) objectFunctionRadiabasis....
    (x_sq_matrix,Y_nomlz,x_number,basis_function_inverse_multiquadric,c,rdibas_matrix_gradient_function);
% c_initial=1;
% [fval,gradient]=object_function(c_initial)
% [fval_diff,gradient_diff]=differ(object_function,c_initial)
% drawFunction(object_function,1e-1,10);
c_inverse_binomial=fmincon(object_function,c_initial,[],[],[],[],1e-2,1e2,[],option_fmincon);

% generate total model
basis_function_list={basis_function_linear;
    basis_function_gauss;
    basis_function_multiquadric;
    basis_function_inverse_multiquadric;};
c_list=[c_linear;c_gauss;c_binomial;c_inverse_binomial];

model_number=size(basis_function_list,1);
beta_list=zeros(x_number,1,model_number);
rdibas_matrix_list=zeros(x_number,x_number,model_number);
inv_rdibas_matrix_list=zeros(x_number,x_number,model_number);

% calculate model matrix and error
model_error_list=zeros(model_number,x_number);
for model_index=1:model_number
    basis_function=basis_function_list{model_index};
    c=c_list(model_index);
    [beta,rdibas_matrix,inv_rdibas_matrix]=interpolationRadialBasis...
        (x_sq_matrix,Y_nomlz,x_number,basis_function,c);
    beta_list(:,:,model_index)=beta;
    rdibas_matrix_list(:,:,model_index)=rdibas_matrix;
    inv_rdibas_matrix_list(:,:,model_index)=inv_rdibas_matrix;
    
    model_error_list(model_index,:)=(beta./...
        diag(inv_rdibas_matrix))';
end

% calculate weight of each model
C=model_error_list*model_error_list';
eta=trace(C)/x_number;
w=(C+eta*eye(model_number))\ones(model_number,1)/...
    (ones(1,model_number)/(C+eta*eye(model_number))*ones(model_number,1));
while min(w) < -0.1
    eta=eta*10;
    w=(C+eta*eye(model_number))\ones(model_number,1)/...
        (ones(1,model_number)/(C+eta*eye(model_number))*ones(model_number,1));
end

% initialization predict function
predict_function=@(predict_x) interpolationEnsemleRadialBasisPredictor...
    (X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    model_number,beta_list,basis_function_list,c_list,...
    w,predict_x);

ensemleradialbasis_model.X=X;
ensemleradialbasis_model.Y=Y;
ensemleradialbasis_model.X_normalize=X_nomlz;
ensemleradialbasis_model.Y_normalize=Y_nomlz;
ensemleradialbasis_model.aver_X=aver_X;
ensemleradialbasis_model.stdD_X=stdD_X;
ensemleradialbasis_model.aver_Y=aver_Y;
ensemleradialbasis_model.stdD_Y=stdD_Y;

ensemleradialbasis_model.basis_function_list=basis_function_list;
ensemleradialbasis_model.c_list=c_list;
ensemleradialbasis_model.beta_list=beta_list;
ensemleradialbasis_model.rdibas_matrix_list=rdibas_matrix_list;
ensemleradialbasis_model.inv_rdibas_matrix_list=inv_rdibas_matrix_list;
ensemleradialbasis_model.model_error_list=model_error_list;
ensemleradialbasis_model.w=w;

ensemleradialbasis_model.predict_function=predict_function;

    function [fval,gradient]=objectFunctionRadiabasis....
            (x_sq_matrix,Y,x_number,basis_function,c,rdibas_matrix_gradient_function)
        % MSE_CV function, simple approximation to RMS
        % basis_function input is c and x_sq
        %
        [beta__,rdibas_matrix__,inv_rdibas_matrix__]=interpolationRadialBasis...
            (x_sq_matrix,Y,x_number,basis_function,c);
        fval=0;
        U=zeros(x_number,1);
        for x_index=1:x_number
            U(x_index)=...
                beta__(x_index)/inv_rdibas_matrix__(x_index,x_index);
            fval=fval+(U(x_index))^2;
        end
        
        % calculate gradient
        if nargout > 1
            inv_rdibas_matrix_gradient=-inv_rdibas_matrix__*...
                rdibas_matrix_gradient_function...
                (x_number,x_sq_matrix,rdibas_matrix__)*inv_rdibas_matrix__;
            U_gradient=zeros(x_number,1);
            I=eye(x_number);
            for x_index=1:x_number
                U_gradient(x_index)=(I(x_index,:)*inv_rdibas_matrix_gradient*Y)/...
                    inv_rdibas_matrix__(x_index,x_index)-...
                    beta__(x_index)*(I(x_index,:)*inv_rdibas_matrix_gradient*I(:,x_index))/...
                    inv_rdibas_matrix__(x_index,x_index)^2;
            end
            
            gradient=2*sum(U.*U_gradient);
        end
    end
    function [beta,rdibas_matrix,inv_rdibas_matrix]=interpolationRadialBasis...
            (x_sq_matrix,Y,x_number,basis_function,c)
        % interpolation polynomial responed surface core function
        % calculation beta       
        rdibas_matrix=zeros(x_number);
        for rank_index__=1:x_number
            for colume_index__=1:rank_index__-1
                rdibas_matrix(rank_index__,colume_index__)=...
                    rdibas_matrix(colume_index__,rank_index__);
            end
            for colume_index__=rank_index__:x_number
                rdibas_matrix(rank_index__,colume_index__)=...
                    basis_function(x_sq_matrix(rank_index__,colume_index__),c);
            end
        end
        rdibas_matrix=rdibas_matrix+eye(x_number)*1e-3;
        
        if rcond(rdibas_matrix) < eps || isnan(rdibas_matrix(1))
            disp('error');
        end
        
        inv_rdibas_matrix=inv(rdibas_matrix);
        beta=inv_rdibas_matrix*Y;
    end
    function predict_y=interpolationEnsemleRadialBasisPredictor...
            (X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            model_number,beta_list,basis_function_list,c_list,...
            w,predict_x)
        % ensemle radial basis function interpolation predict function
        % input predict_x and radialbasis_model model
        if size(predict_x,2) > 1
            predict_x=predict_x';
        end
        
        % normalize data
        predict_x_nomlz=(predict_x-aver_X')./stdD_X';
        
        % calculate each sub model predict fval and get predict_y
        predict_y_nomlz=0;
        for model_index__=1:model_number
            basis_function__=basis_function_list{model_index__};
            c__=c_list(model_index__);
            beta__=beta_list(:,:,model_index__);
            predict_y_nomlz_sub=interpolationRadialBasisPredictor...
                (X_nomlz,beta__,basis_function__,c__,predict_x_nomlz);
            predict_y_nomlz=predict_y_nomlz+w(model_index__)*predict_y_nomlz_sub;
        end
        
        % normalize data
        predict_y=predict_y_nomlz*stdD_Y+aver_Y;
        
        function predict_y_nomlz=interpolationRadialBasisPredictor...
                (X_nomlz,beta,basis_function,c,predict_x_nomlz)
            % radial basis function interpolation predict function
            % input predict_x and radialbasis_model model
            % predict_x is row vector
            % output the predict value
            [x_number__,~]=size(X_nomlz);
            
            % predict value
            X_inner_product__=zeros(x_number__,1);
            for x_index__=1:x_number__
                X_inner_product__(x_index__)=...
                    basis_function(sum((X_nomlz(x_index__,:)'-predict_x_nomlz).^2),c);
            end
            
            % predict variance
            predict_y_nomlz=beta'*X_inner_product__;
        end
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

axes_handle=figure_handle.CurrentAxes;
if isempty(axes_handle)
    axes_handle=axes(figure_handle);
end

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
                con_list=[con_list;...
                    data(5+variable_number:4+variable_number+con_number)];
                coneq_list=[coneq_list;...
                    data(6+variable_number+con_number:5+variable_number+con_number+coneq_number)];
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
iteration_max=10*variable_number;

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

% initialize X_new by sample with constraint function
% generate sample sequence latin hypercube
% election sequential method is used
% sample number is total point in area
% default low_bou is 0, up_bou is 1, cheapcon_function is []
% low_bou and up_bou is colume vector
% x in x_exist_list, x_list, supply_x_list is row vector
% x_exist_list should meet bou
% get quasi-feasible point
x_initial_number=100*x_new_number;
if ~isempty(cheapcon_function)
    X_supply_quasi_nomlz=[];
    % check if have enough X_supply_nomlz
    while size(X_supply_quasi_nomlz,1) < 100*x_new_number
        X_supply_initial_nomlz=rand(x_initial_number,variable_number);
        x_index=1;
        while x_index <= size(X_supply_initial_nomlz,1)
            x_supply=X_supply_initial_nomlz(x_index,:).*(up_bou'-low_bou')+low_bou';
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
distance_min_nomlz__=0;
X_new_nomlz=[];
while iteration <= iteration_max
    % random select x_new_number X to X_trial_nomlz
    x_select_index=randperm(x_supply_quasi_number,x_new_number);
    
    % get distance min itertion X_
    distance_min_iteration=getMinDistanceIter...
        (X_supply_quasi_nomlz(x_select_index,:),X_exist_nomlz);
    
    % if distance_min_iteration is large than last time
    if distance_min_iteration > distance_min_nomlz__
        distance_min_nomlz__=distance_min_iteration;
        X_new_nomlz=X_supply_quasi_nomlz(x_select_index,:);
    end
    
    iteration=iteration+1;
end
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
        surf(axes_handle,X__',Y',fval__,'FaceAlpha',0.5,'EdgeColor','none');
        xlabel('X');
        ylabel('Y');
        zlabel('value');
end

end