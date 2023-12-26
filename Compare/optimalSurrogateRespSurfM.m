function [x_best,fval_best,NFE,output]=optimalSurrogateRespSurfM...
    (object_function,variable_number,A,B,Aeq,Beq,low_bou,up_bou,nonlcon_function,...
    cheapcon_function,model_function,NFE_max)
% surrogate base optimal use polynomial responed surface method version 2
% use MSP guideline to decide which point was select to updata, spq optimal
% x_best is row vector by ga, so need to transpose into colume vector
% all function exchange x should be colume vector,x_list except
% both nonlcon_function and cheapcon_function format is [con,coneq]
% model_function should output fval, format is [fval,con,coneq]
% con or coneq can be colume vector if there was more than one constrain
%
% Copyright 2022 Adel
%
if nargin < 12
    if nargin <10
        cheapcon_function=[];
        if nargin < 9
            nonlcon_function=[];
        end
    end
    if isempty(A) && isempty(B) && isempty(Aeq) && isempty(Beq) &&...
            isempty(cheapcon_function) && isempty(nonlcon_function)
        NFE_max=20*variable_number;
    else
        NFE_max=50*variable_number;
    end
end
INFORMATION_FLAG=0; % whether draw data
CONVERGENCE_JUDGMENT_FLAG=0; % whether judgment convergence
range_min=0.01;
range_max=0.5;

iteration_max=100;
torlance=1e-2;

% augmented lagrange parameter
lambda_initial=1;
miu=1;
miu_max=1000;
gama=2;

data_library_name='optimalSurrogateRespSurf_result.txt';
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

sample_number=(variable_number+1)*(variable_number+2)/2; % Latin hypercube sample count

iteration=iteration+1;

% bourdary updata
low_bou_iter=low_bou;
up_bou_iter=up_bou;

respsurf_model_fval=[];
respsurf_model_con=[];
respsurf_model_coneq=[];

% Step 1
% generate latin hypercube sequence ESLHS
[~,x_list_updata,~]=getLatinHypercube...
    (sample_number,variable_number,[],...
    low_bou_iter,up_bou_iter,cheapcon_function);

% if only input model function, detect constraint
[~,con,coneq]=dataLibraryUpdata...
    (data_library_name,model_function,x_list_updata(1,:));NFE=NFE+1;
if ~isempty(con) || ~isempty(coneq)
    expensive_nonlcon_flag=1;
else
    expensive_nonlcon_flag=0;
end
x_list_updata=x_list_updata(2:end,:);

% loop
while ~done
    % Step 2
    % update x_list_updata into data library
    [~,~,~]=dataLibraryUpdata...
        (data_library_name,model_function,x_list_updata);NFE=NFE+size(x_list_updata,1);
    
    % Step 3
    % load data
    [x_list,fval_list,con_list,coneq_list]=dataLibraryLoad...
        (data_library_name,low_bou_iter,up_bou_iter);
    
    % nomalization con
    fval_max=max(fval_list);
    if ~isempty(con_list)
        con_max_list=max(abs(con_list),[],1);
        con_list=con_list./con_max_list.*fval_max;
    end
    if ~isempty(coneq_list)
        coneq_max_list=max(abs(coneq_list),[],1);
        coneq_list=coneq_list./coneq_max_list.*fval_max;
    end
    
    % get surrogate model and function
    % avoid too close point
    select_index_list=1:size(x_list,1);
    index=1;
    while index<length(select_index_list)
        select_index=select_index_list(index);
        distance=sum(((x_list(select_index,:)-[x_list(1:select_index-1,:);x_list(select_index+1:end,:)])./(up_bou_iter'-low_bou_iter')).^2,2);
        if min(distance)<1e-6
            select_index_list(index)=[];
        else
            index=index+1;
        end
    end
    x_list_S=x_list(select_index_list,:);
    fval_list_S=fval_list(select_index_list,:);
    con_list_S=con_list(select_index_list,:);
    coneq_list_S=coneq_list(select_index_list,:);
    
    % if point too less, add more point
    if size(x_list_S,1) < 6
        % generate latin hypercube sequence
        [~,x_list_updata,~]=getLatinHypercube...
            (size(x_list,1)+(6-size(x_list_S,1)),variable_number,x_list,...
            low_bou_iter,up_bou_iter,cheapcon_function);
        % update x_list_updata into data library
        [fval_list_updata,con_list_updata,coneq_list_updata]=dataLibraryUpdata...
            (data_library_name,model_function,x_list_updata);NFE=NFE+size(x_list_updata,1);
        x_list_S=[x_list_S;x_list_updata];x_list=[x_list;x_list_updata];
        fval_list_S=[fval_list_S;fval_list_updata];fval_list=[fval_list;fval_list_updata];
        if ~isempty(con_list_updata)
            con_list_S=[con_list_S;con_list_updata./con_max_list.*fval_max];
            con_list=[con_list;con_list_updata./con_max_list.*fval_max];
        end
        if ~isempty(coneq_list_updata)
            coneq_list_S=[coneq_list_S;coneq_list_updata./coneq_max_list.*fval_max];
            coneq_list=[coneq_list;coneq_list_updata./coneq_max_list.*fval_max];
        end
    end
    [respsurf_model_fval,~,~,output_S]=getRespSurfModel...
        (x_list_S,fval_list_S,con_list_S,coneq_list_S);
    object_function_S=output_S.object_function;
    nonlcon_function_S=output_S.nonlcon_function;
    
    % generate merit function
    if expensive_nonlcon_flag
        if iteration == 1
            con_number=size(con_list,2);
            coneq_number=size(coneq_list,2);
            lambda_con=lambda_initial*ones(con_number,1);
            lambda_coneq=lambda_initial*ones(coneq_number,1);
        end
        % generate merit function fval and add fval
        merit_function=@(x) meritFunction...
            (x,object_function_S,nonlcon_function_S,...
            lambda_con,lambda_coneq,miu);
    end
    
    if INFORMATION_FLAG
        interpolationRespSurfVisualize(respsurf_model_fval,low_bou_iter,up_bou_iter);
    end
    
    % Step 4
    % updata data, x_potential
    if expensive_nonlcon_flag
        [x_best,~,~,~]=findMinRaw...
            (x_list,fval_list,con_list,coneq_list,...
            A,B,Aeq,Beq,cheapcon_function);
        [x_potential,fval_potential_predict]=findMinMSPFmincon...
            (merit_function,x_best,A,B,Aeq,Beq,low_bou_iter,up_bou_iter,[],...
            cheapcon_function);
    else
        [x_best,~,~,~]=findMinRaw...
            (x_list,fval_list,con_list,coneq_list,...
            A,B,Aeq,Beq,cheapcon_function);
        [x_potential,fval_potential_predict]=findMinMSPFmincon...
            (object_function_S,x_best,A,B,Aeq,Beq,low_bou_iter,up_bou_iter,[],...
            cheapcon_function);
    end
    
    % check x_potential if exit in data library
    % if not, updata data libraray
    distance=sum(((x_potential'-x_list)./(low_bou_iter'-up_bou_iter')).^2,2);
    [~,min_index]=min(distance);
    if distance(min_index)<1e-12
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
        [fval_potential,con_potential,coneq_potential]=dataLibraryUpdata...
            (data_library_name,model_function,x_potential');NFE=NFE+1;
        con_potential=con_potential';
        coneq_potential=coneq_potential';
    end
    
    % updata penalty factor and lagrangian
    if expensive_nonlcon_flag
        lambda_con=lambda_con+2*miu*max(con_potential,-lambda_con/2/miu);
        lambda_coneq=lambda_coneq+2*miu*coneq_potential;
        if miu < miu_max
            miu=gama*miu;
        else
            miu=miu_max;
        end
    end
    
    % Step 5
    % find best result to record
    [x_list,fval_list,con_list,coneq_list]=dataLibraryLoad...
        (data_library_name,low_bou,up_bou);
    [x_best,fval_best,con_best,coneq_best]=findMinRaw...
        (x_list,fval_list,con_list,coneq_list,...
        A,B,Aeq,Beq,cheapcon_function);
    result_x_best(iteration,:)=x_best';
    result_fval_best(iteration,:)=fval_best;
    iteration=iteration+1;
    
    % forced interrupt
    if iteration > iteration_max || NFE > NFE_max
        done=1;
    end
    
    % convergence judgment
    if CONVERGENCE_JUDGMENT_FLAG
        if (iteration > iteration_max*0.2 && ...
                abs((fval_potential-fval_potential_old)/fval_potential_old) < torlance)
            if ~sum(con_best > torlance)
                done=1;
            end
        end
    end
    
    % Step 6
    % TR guideline first iteration
    if iteration == 2
        bou_range_nomlz=0.5;
    else
        r=(fval_potential_old-fval_potential)/(fval_potential_old-fval_potential_predict);
        
        scale=bou_range_nomlz*2/(1+exp(-(r-0.5)));
        bou_range_nomlz=(1/(1+exp(-10*(scale-0.25+rand()*0.5-0.25))))*(range_max-range_min)+range_min;
    end
    bou_range=bou_range_nomlz.*(up_bou-low_bou)+low_bou;
    
    % updata trust range
    low_bou_temp=x_best-bou_range;
    select_index=find(low_bou_temp < low_bou);
    low_bou_temp(select_index)=low_bou(select_index);
    low_bou_iter=low_bou_temp;
    up_bou_temp=x_best+bou_range;
    select_index=find(up_bou_temp > up_bou);
    up_bou_temp(select_index)=up_bou(select_index);
    up_bou_iter=up_bou_temp;
    
    % Step 7
    % check whether exist data
    [x_list_exist,~,~,~]=dataLibraryLoad...
        (data_library_name,low_bou_iter,up_bou_iter);
    
    % generate latin hypercube sequence
    [~,x_list_updata,~]=getLatinHypercube...
        (sample_number,variable_number,x_list_exist,...
        low_bou_iter,up_bou_iter,cheapcon_function);
    
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
    function fval=meritFunction...
            (x,object_function,nonlcon_function,lambda_con,lambda_coneq,miu)
        % penalty function
        % augmented lagrange multiplier method was used
        %
        fval=object_function(x);
        if ~isempty(nonlcon_function)
            [con__,coneq__]=nonlcon_function(x);
            if ~isempty(con__)
                psi=max(con__,-lambda_con/2/miu);
                fval=fval+sum(lambda_con.*psi+miu*psi.*psi);
            end
            if ~isempty(coneq__)
                fval=fval+sum(lambda_coneq.*coneq__+miu*coneq__.*coneq__);
            end
        end
    end
end

function [x_best,fval_best]=findMinMSPFmincon...
    (object_function,x_initial,A,B,Aeq,Beq,low_bou,up_bou,nonlcon_function,...
    cheapcon_function)
% find min fval use MSP guideline
% MSP: object_funtion is object_function (generate by surrogate model)
% nonlcon_function generate by surrogate model
% use ga as optimal method
%
if ~isempty(nonlcon_function) || ~isempty(cheapcon_function)
    totalcon_function=@(x) totalconFunction(x,nonlcon_function,cheapcon_function);
else
    totalcon_function=[];
end

% optiaml
fmincon_option=optimoptions('fmincon','display','none','algorithm','sqp');
[x_best,fval_best,~,~]=fmincon...
    (object_function,x_initial,A,B,Aeq,Beq,low_bou,up_bou,totalcon_function,fmincon_option);

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
    A,B,Aeq,Beq,cheapcon_function)
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
    if ~isempty(A)
        if ~isempty(B)
            lincon=A*x_list(x_index,:)'-B;
        else
            lincon=A*x_list(x_index,:)';
        end
        con_sum_list(x_index)=con_sum_list(x_index)+sum(max(lincon,0));
    end
    if ~isempty(Aeq)
        if ~isempty(Beq)
            linconeq=Aeq*x_list(x_index,:)'-Beq;
        else
            linconeq=Aeq*x_list(x_index,:)';
        end
        con_sum_list(x_index)=con_sum_list(x_index)+sum(coneq_list.*coneq_list);
    end
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

function [respsurf_model_fval,respsurf_model_con,respsurf_model_coneq,output]=getRespSurfModel...
    (x_list,fval_list,con_list,coneq_list,...
    respsurf_model_fval,respsurf_model_con,respsurf_model_coneq)
% base on library_data to create respsurf model and function
% if input model, function will updata model
% object_function is single fval output
% nonlcon_function is normal nonlcon_function which include con, coneq
% con is colume vector, coneq is colume vector
% var_function is same
%
if nargin < 7
    respsurf_model_coneq=[];
    if nargin < 6
        respsurf_model_con=[];
        if nargin < 5
            respsurf_model_fval=[];
        end
    end
end

% generate surrogate model
if isempty(respsurf_model_fval)
    respsurf_model_fval=interpolationRespSurfPreModel...
        (x_list,fval_list);
else
    respsurf_model_fval=interpolationRespSurfUpdata...
        (respsurf_model_fval,x_list,fval_list);
end

if ~isempty(con_list)
    if isempty(respsurf_model_con)
        respsurf_model_con=struct('X',[],'Y',[],'X_normalize',[],'Y_normalize',[],...
            'beta',[],...
            'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[]);
        respsurf_model_con=repmat(respsurf_model_con,[size(con_list,2),1]);
        for con_index=1:size(con_list,2)
            respsurf_model_con(con_index)=interpolationRespSurfPreModel...
                (x_list,con_list(:,con_index));
        end
    else
        for con_index=1:size(con_list,2)
            respsurf_model_con(con_index)=interpolationRespSurfUpdata...
                (respsurf_model_con(con_index),x_list,con_list(:,con_index));
        end
    end
end

if ~isempty(coneq_list)
    if isempty(respsurf_model_coneq)
        respsurf_model_coneq=struct('X',[],'Y',[],'X_normalize',[],'Y_normalize',[],...
            'beta',[],...
            'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[]);
        respsurf_model_coneq=repmat(respsurf_model_coneq,[size(coneq_list,2),1]);
        for coneq_index=1:size(coneq_list,2)
            respsurf_model_coneq(coneq_index)=interpolationRespSurfPreModel...
                (x_list,coneq_list(:,coneq_index));
        end
    else
        for coneq_index=1:size(coneq_list,2)
            respsurf_model_coneq(coneq_index)=interpolationRespSurfUpdata...
                (respsurf_model_coneq(coneq_index),x_new_list,coneq_list(:,coneq_index));
        end
    end
end

object_function=@(predict_x) objectFunctionSurrogate(predict_x,respsurf_model_fval);
if isempty(respsurf_model_con) && isempty(respsurf_model_coneq)
    nonlcon_function=[];
else
    nonlcon_function=@(predict_x) constraintFunctionSurrogate(predict_x,respsurf_model_con,respsurf_model_coneq);
end

output.object_function=object_function;
output.nonlcon_function=nonlcon_function;

    function fval=objectFunctionSurrogate...
            (predict_x,respsurf_model_fval)
        fval=interpolationRespSurfPredictor(respsurf_model_fval,predict_x);
    end
    function [con,coneq]=constraintFunctionSurrogate...
            (predict_x,respsurf_model_con,respsurf_model_coneq)
        if isempty(respsurf_model_con)
            con=[];
        else
            con=zeros(length(respsurf_model_con),1);
            for con_ind=1:length(respsurf_model_con)
                con(con_ind)=interpolationRespSurfPredictor...
                    (respsurf_model_con(con_ind),predict_x);
            end
        end
        if isempty(respsurf_model_coneq)
            coneq=[];
        else
            coneq=zeros(length(respsurf_model_coneq),1);
            for coneq_ind=1:length(respsurf_model_con)
                coneq(coneq_ind)=interpolationRespSurfPredictor...
                    (respsurf_model_coneq(coneq_ind),predict_x);
            end
        end
    end
end
function respsurf_model=interpolationRespSurfPreModel...
    (X,Y)
% polynomial response surface interpolation pre model function
% input initial data X and Y, x is row vector
% return model
% beta in respsurf_model is colume vector
%
% Copyright 2022 Adel
%
x_number=size(X,1);
% normalize data
aver_X = mean(X);
stdD_X = std(X);
aver_Y = mean(Y);
stdD_Y = std(Y);
index__ = find(stdD_X == 0);
if ~isempty(index__),stdD_X(index__)=1;end
index__ = find(stdD_Y == 0);
if ~isempty(index__),stdD_Y(index__)=1;end
X_normalize = (X - repmat(aver_X,x_number,1)) ./ repmat(stdD_X,x_number,1);
Y_normalize = (Y - repmat(aver_Y,x_number,1)) ./ repmat(stdD_Y,x_number,1);

beta=interpolationRespSurf(X_normalize,Y_normalize);

respsurf_model.X=X;
respsurf_model.Y=Y;
respsurf_model.X_normalize=X_normalize;
respsurf_model.Y_normalize=Y_normalize;
respsurf_model.beta=beta;
respsurf_model.aver_X=aver_X;
respsurf_model.stdD_X=stdD_X;
respsurf_model.aver_Y=aver_Y;
respsurf_model.stdD_Y=stdD_Y;
end
function [predict_y]=interpolationRespSurfPredictor...
    (respsurf_model,predict_x)
% polynomial response surface interpolation predict function
% input predict_x and respsurf_model model
% predict_x is row vector
% output the predict value
%
% Copyright 2022 Adel
%
if nargin < 2
    error('interpolationKrigingPredictor: lack predict_x')
end
if size(predict_x,1) > 1
    predict_x=predict_x';
end
variable_number=size(predict_x,2);

% giving value
beta=respsurf_model.beta;
aver_X=respsurf_model.aver_X;
stdD_X=respsurf_model.stdD_X;
aver_Y=respsurf_model.aver_Y;
stdD_Y=respsurf_model.stdD_Y;

% normalize data
predict_x=(predict_x-aver_X)./stdD_X;

% predict value
x_cross=zeros(1,(variable_number-1)*variable_number/2);
x_sqrt=predict_x.^2;
cross_index=1;
for i_index=1:variable_number
    for j_index=i_index+1:variable_number
        x_cross(cross_index)=predict_x(i_index)*predict_x(j_index);
        cross_index=cross_index+1;
    end
end
x_inter=[1,predict_x,x_sqrt,x_cross];

% predict variance
predict_y=x_inter*beta;

% normalize data
predict_y=predict_y*stdD_Y+aver_Y;
end
function respsurf_model=interpolationRespSurfUpdata...
    (respsurf_model,X_new,Y_new)
% add new point in exiting polynomial response surface model
% X_new is row vector, can be more than one
% Y_new is row vector, can be more than one
%

% giving value
X=respsurf_model.X;
Y=respsurf_model.Y;

% add new data
X=[X;X_new];
Y=[Y;Y_new];

x_number=size(X,1);

% normalize data
aver_X = mean(X);
stdD_X = std(X);
aver_Y = mean(Y);
stdD_Y = std(Y);
index__ = find(stdD_X == 0);
if  ~isempty(index__),  stdD_X(index__) = 1; end
index__ = find(stdD_Y == 0);
if  ~isempty(index__),  stdD_Y(index__) = 1; end
X_normalize = (X - repmat(aver_X,x_number,1)) ./ repmat(stdD_X,x_number,1);
Y_normalize = (Y - repmat(aver_Y,x_number,1)) ./ repmat(stdD_Y,x_number,1);

% updata model
beta=interpolationRespSurf(X_normalize,Y_normalize);

respsurf_model.X=X;
respsurf_model.Y=Y;
respsurf_model.X_normalize=X_normalize;
respsurf_model.Y_normalize=Y_normalize;
respsurf_model.beta=beta;
respsurf_model.aver_X=aver_X;
respsurf_model.stdD_X=stdD_X;
respsurf_model.aver_Y=aver_Y;
respsurf_model.stdD_Y=stdD_Y;
end
function interpolationRespSurfVisualize...
    (respsurf_model,low_bou,up_bou,figure_handle)
% visualization polynamial respond surface model
% figrue is 100
%
if nargin < 4
    figure_handle=figure(100);
end
if size(low_bou,1) ~= size(low_bou,1)
    error('interpolationRespSurfVisualize: boundary incorrect');
end
if size(low_bou,1) > 2
    error('interpolationRespSurfVisualize: dimension large than two');
end
axes_handle=figure_handle.CurrentAxes;
if isempty(axes_handle)
    axes_handle=axes(figure_handle);
end

x_list=respsurf_model.X;
y_list=respsurf_model.Y;

% calculate predict value
grid_number=100;
d_bou=(up_bou-low_bou)/grid_number;
predict_result=zeros(grid_number+1);
[X,Y]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
for x_index=1:grid_number+1
    for y_index=1:grid_number+1
        predict_x=([x_index,y_index]-1).*d_bou'+low_bou';
        [predict_result(y_index,x_index)]=interpolationRespSurfPredictor...
            (respsurf_model,predict_x);
    end
end
bou=[low_bou,up_bou]';
bou=bou(:);
axis(bou');
surf(axes_handle,X,Y,predict_result,'FaceAlpha',0.5,'EdgeColor','none');
line(axes_handle,x_list(:,1),x_list(:,2),y_list,'Marker','o','LineStyle','none');
xlabel('X');
ylabel('Y');
zlabel('Z');
end
function beta=interpolationRespSurf(X,Y)
% interpolation polynomial responed surface core function
% calculation beta
%
[x_number,variable_number]=size(X);

if size(X,1)<6
   disp('error'); 
end

X_inter=zeros(x_number,(variable_number+1)*(variable_number+2)/2);
x_cross=zeros(1,(variable_number-1)*variable_number/2);
for x_index=1:x_number
    x=X(x_index,:);
    x_sqrt=x.^2;
    cross_index=1;
    for i_index=1:variable_number
        for j_index=i_index+1:variable_number
            x_cross(cross_index)=x(i_index)*x(j_index);
            cross_index=cross_index+1;
        end
    end
    X_inter(x_index,:)=[1,x,x_sqrt,x_cross];
end

X_inter_X_inter=X_inter'*X_inter;
beta=X_inter_X_inter\X_inter'*Y;
if isnan(beta(1))
    disp('error');
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
        % search whether exit point
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

function [X,X_new,distance_min_normalize]=getLatinHypercube...
    (sample_number,variable_number,X_exit,...
    low_bou,up_bou,cheapcon_function)
% generate sample sequence latin hypercube
% iteration optimal method is used
% sample number is total point in area
% default low_bou is 0, up_bou is 1, cheapcon_function is []
% low_bou and up_bou is colume vector
% x in x_exit_list, x_list, supply_x_list is row vector
% x_exit_list should meet bou
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
iteration_max=20*variable_number;

% check x_exit_list if meet boundary
if ~isempty(X_exit)
    index=find(X_exit < low_bou');
    index=[index,find(X_exit > up_bou')];
    if ~isempty(index)
        error('getLatinHypercube: x_exit_list range error');
    end
    if size(X_exit,2) ~= variable_number
        error('getLatinHypercube: x_exit_list variable_number error');
    end
    X_exit_nomlz=(X_exit-low_bou')./(up_bou'-low_bou');
else
    X_exit_nomlz=[];
end

% check x_new_number
x_new_number=sample_number-size(X_exit,1);
if x_new_number < 0
    X=X_exit;
    X_new=[];
    distance_min_normalize=getMinDistance(X_exit_nomlz);
    return;
end

low_bou_nomlz=zeros(variable_number,1);
up_bou_nomlz=ones(variable_number,1);

% initialize X_new by sample with constraint function
[X_new,distance_min_normalize]=getLatinHypercubeInitial...
    (sample_number,variable_number,X_exit,...
    low_bou,up_bou,cheapcon_function);

% x is nomalize, so constraint function should change
if ~isempty(cheapcon_function)
    cheapcon_function=@(x) ...
        max(max(sample_number*cheapcon_function(x.*(up_bou-low_bou)+low_bou)+1,0),[],1);
end

if size(X_new,1)~=x_new_number || size(X_new,2)~=variable_number
    disp('error');
end

iteration=0;
while iteration < iteration_max
    %     scatter(X_new(:,1),X_new(:,2));
    %     bou=[low_bou_nomlz,up_bou_nomlz]';
    %     axis(bou(:));axis equal
    %     radius=1/6;
    %     hold on;
    %     rectangle('Position',[-radius+0.5,-radius+0.5,2*radius,2*radius],'Curvature',[1 1])
    %     hold off;
    
    % change each x place by newton methods
    fval_list=zeros(x_new_number,1);
    gradient_list=zeros(variable_number,x_new_number);
    for X_new_index=1:x_new_number
        % get gradient
        [fval_list(X_new_index,1),gradient_list(:,X_new_index),~]=objectFunctionXPlace...
            (X_new(X_new_index,:)',[X_new(1:X_new_index-1,:);X_new(X_new_index+1:end,:);X_exit_nomlz],...
            variable_number,low_bou_nomlz,up_bou_nomlz,cheapcon_function);
    end
    
    fval_list=fval_list/max(fval_list);
    for X_new_index=1:x_new_number
        x=X_new(X_new_index,:)'-...
            fval_list(X_new_index,1)*gradient_list(:,X_new_index)/norm(gradient_list(:,X_new_index),2)*...
            distance_min_normalize*(1-iteration/iteration_max);
        for variable_index=1:variable_number
            if x(variable_index,1) > 1
                x(variable_index,1)=1;
            end
            if x(variable_index,1) < 0
                x(variable_index,1)=0;
            end
        end
        X_new(X_new_index,:)=x';
    end
    
    iteration=iteration+1;
end
distance_min_normalize=getMinDistance([X_new;X_exit_nomlz]);
X_new=X_new.*(up_bou'-low_bou')+low_bou';
X=[X_new;X_exit];


    function [fval,gradient,hessian]=objectFunctionXPlace...
            (x,X_surplus,variable_number,low_bou,up_bou,cheapcon_function)
        % function describe distance between X and X_supply
        % X__ is colume vector and X_supply__ is matrix which is num-1 x var
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
            
            fval__(:,2)=differ_function(x);
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
    function [X_new_nomlz,distance_min_normalize__]=getLatinHypercubeInitial...
            (sample_number,variable_number,X_exit,...
            low_bou,up_bou,cheapcon_function)
        % generate sample sequence latin hypercube
        % election sequential method is used
        % sample number is total point in area
        % default low_bou is 0, up_bou is 1, cheapcon_function is []
        % low_bou and up_bou is colume vector
        % x in x_exit_list, x_list, supply_x_list is row vector
        % x_exit_list should meet bou
        %
        % Copyright 2022 Adel
        %
        
        iteration_max__=10*variable_number;
        x_new_number__=sample_number-size(X_exit,1);
        
        % get quasi-feasible point
        iteration__=0;
        x_initial_number=100*x_new_number__;
        X_supply_initial_normalize=rand(x_initial_number,variable_number);
        if ~isempty(cheapcon_function)
            x_index=1;
            while x_index <= size(X_supply_initial_normalize,1)
                x_supply=X_supply_initial_normalize(x_index,:).*(up_bou'-low_bou')+low_bou';
                if max(cheapcon_function(x_supply),[],1) > 0
                    X_supply_initial_normalize(x_index,:)=[];
                else
                    x_index=x_index+1;
                end
            end
            % check if have enough X_supply_normalize
            while size(X_supply_initial_normalize,1) < 10*x_new_number__ &&...
                    iteration__ < iteration_max__
                
                X_supply_initial_normalize_new=rand(x_initial_number,variable_number);
                x_index=1;
                while x_index <= size(X_supply_initial_normalize_new,1)
                    x_supply=X_supply_initial_normalize_new(x_index,:).*(up_bou'-low_bou')+low_bou';
                    if cheapcon_function(x_supply) > 0
                        X_supply_initial_normalize_new(x_index,:)=[];
                    else
                        x_index=x_index+1;
                    end
                end
                
                % add more point
                X_supply_initial_normalize=[X_supply_initial_normalize;...
                    X_supply_initial_normalize_new];
                iteration__=iteration__+1;
            end
        end
        X_supply_quasi_normalize=X_supply_initial_normalize;
        
        % iterate and get final x_supply_list
        iteration__=0;
        x_supply_quasi_number=size(X_supply_quasi_normalize,1);
        if ~isempty(X_exit)
            x_exit_list_normalize=(X_exit-low_bou')./(up_bou'-low_bou');
        else
            x_exit_list_normalize=[];
        end
        distance_min_normalize__=0;
        X_new_nomlz=[];
        while iteration__ <= iteration_max__
            % random select x_new_number X to X_trial_normalize
            x_select_index=randperm(x_supply_quasi_number,x_new_number__);
            X_trial_normalize=[X_supply_quasi_normalize(x_select_index,:);x_exit_list_normalize];
            
            % get distance min itertion X_
            distance_min_iteration=getMinDistance(X_trial_normalize);
            
            % if distance_min_iteration is large than last time
            if distance_min_iteration > distance_min_normalize__
                distance_min_normalize__=distance_min_iteration;
                X_new_nomlz=X_supply_quasi_normalize(x_select_index,:);
            end
            
            iteration__=iteration__+1;
        end
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
        for x_index=1:(grid_number+1)
            fval__(x_index)=object_function(X__(x_index));
        end
        line(axes_handle,X__,fval__);
        xlabel('X');
        ylabel('value');
        
    case 2
        d_bou=(up_bou-low_bou)/grid_number;
        [X__,Y]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
        fval__=zeros(grid_number+1);
        for x_index=1:grid_number+1
            for y_index=1:grid_number+1
                predict_x=([x_index;y_index]-1).*d_bou+low_bou;
                fval__(x_index,y_index)=object_function(predict_x);
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