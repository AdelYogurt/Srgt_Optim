clc;
clear;
close all hidden;

% % regression
% 
% mu_y=0;
% sigma_y=0.2; % noise gaussian sigma
% 
% w_real=[1;1;2.5];
% x_number=10;
% X=[ones(1,x_number);rand(2,x_number)*2-1];
% Y=X'*w_real+normrnd(mu_y,sigma_y,[x_number,1]);
% 
% scatter3(X(2,:),X(3,:),Y);
% 
% sigma_w=eye(length(w_real)); % priori disturbution of w
% 
% A=(X*X')/sigma_y^2+inv(sigma_w);
% w_predict=(1/sigma_y^2)*(A\(X*Y))
% inv(A)
% 
% object_function=@(w) -logPosteriorFunction(w,X,Y,sigma_y,sigma_w);
% w_best=fmincon(object_function,[1;1;1])
% 
% function possibility=logPosteriorFunction(w,X,Y,sigma_y,sigma_w)
% Y_XW=Y-X'*w;
% possibility=-0.5*(Y_XW'*Y_XW)/sigma_y^2-0.5*w'*(sigma_w\w);
% end

% classification

w_real=[0;1;2.5];
x_number=100;
X=[ones(1,x_number);rand(2,x_number)*2-1];
F=X'*w_real;
Y=sigmaFunction(F);
Y(Y>0.5)=1;
Y(Y<0.5)=-1;

sigma_w=eye(length(w_real)); % priori disturbution of w

object_function=@(w) -logPosteriorFunction(w,X,Y,sigma_w);
w_best=fmincon(object_function,[1;1;1])

log_posterior_function=@(w) logPosteriorFunction(w,X,Y,sigma_w);

sample_number=1000;
sample_adapt=500;
accept_probability=0.65;
[sample_list,~]=samplerNUTS...
    (log_posterior_function,[1;1;1],sample_number,sample_adapt,accept_probability);
hyperparameter=(sum(sample_list,1)/sample_number)';
disp(['hyperparameter: ']);disp(hyperparameter);

count_number=100;
[count_list,~,~,X_draw]=drawSample...
    (sample_list,count_number);

function [fval,gradient]=logPosteriorFunction(w,X,Y,sigma_w)
F=X'*w;
YF=Y.*F;
sigma_YF=sigmaFunction(YF);
fval=sum(log(sigma_YF))-0.5*w'*(sigma_w\w);

gradient=zeros(length(w),1);
for gradient_index=1:length(w)
    gradient(gradient_index)=sum((exp(-YF).*sigma_YF).*(Y.*X(gradient_index,:)'));
end
gradient=gradient-sigma_w\w;
end

function [gradient,hessian]=differ(differ_function,x,fval,step)
% differ function to get gradient and hessian
%
variable_number=length(x);
if nargin < 4
    step=1e-6;
end
if nargin < 3
    fval=differ_function(x);
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

function F=sigmaFunction(x)
F=1./(1+exp(-x));
end

function [sample_list,output]=samplerNUTS...
    (log_sample_function,sample_initial,sample_number,iteration_adapt,accept_probability)
% No-U-Turn Sampler
% sample function is log density function
% sample_list is sample_number x var_number matrix
%
sample_list=zeros(sample_number,length(sample_initial));
step_list=zeros(sample_number,1);
step_dual_list=zeros(sample_number,1);
H_dual_list=zeros(sample_number,1);
log_fval_list=zeros(sample_number,1);
depth=zeros(sample_number,1);

iteration_max=sample_number;

% normalize function to improve perform
% nomlz_sample=100;
% fval_list=zeros(nomlz_sample,1);
% X_sample_nomlz=(lhsdesign(nomlz_sample,length(sample_initial)))*1;
% for nomlz_index=1:nomlz_sample
%     x=X_sample_nomlz(nomlz_index,:)';
%     fval_list(nomlz_index,1)=(log_sample_function(x));
% end
% max_fval=sum(fval_list,1)/nomlz_sample;
% log_sample_function=@(x) normalizeFunction(x,log_sample_function,max_fval);

% sampling initial value
step=getReasonableStepSize...
    (log_sample_function,sample_initial);
mu=log(10*step);
gama=0.05;
t0=10;
kappa=0.75;
j_max=8;

% recode
sample_list(1,:)=sample_initial;
step_list(1)=step;
step_dual_list(1)=1;
log_fval_list(1)=log_sample_function(sample_initial);

% sampling
disp('samplerNUTS: start sampling');
for iteration=1:iteration_max
    theta=sample_list(iteration,:)';
    step=step_list(iteration);
    log_fval=log_fval_list(iteration);
    
    r=mvnrnd(zeros(length(theta),1),eye(length(theta)))';
    u=rand()*exp(log_sample_function(theta)-0.5*(r'*r));
    
    theta_negative=theta;
    theta_postive=theta;
    r_negative=r;
    r_postive=r;
    j=0;
    n=1;
    s=1; % judgement
    
    while s==1 && j < j_max
        % choose a direction
        dir=rand();
        if dir>=0.5
            dir=1;
        else
            dir=-1;
        end
        
        % negatiev direction or postive direction
        if dir==-1
            [theta_negative,r_negative,~,~,...
                theta_temp,log_fval_temp,n_temp,s_temp,...
                alpha,n_alpha]=buildTree...
                (log_sample_function,theta_negative,r_negative,u,dir,j,step,...
                theta,r,log_fval);
        else
            [~,~,theta_postive,r_postive,...
                theta_temp,log_fval_temp,n_temp,s_temp,...
                alpha,n_alpha]=buildTree...
                (log_sample_function,theta_postive,r_postive,u,dir,j,step,...
                theta,r,log_fval);
        end
        
        if s_temp == 1
            if rand() < min(1,n_temp/n)
                theta=theta_temp;
                log_fval=log_fval_temp;
            end
        end
        
        n=n+n_temp;
        s=s_temp*(((theta_postive-theta_negative)'*r_negative)>=0)*...
            (((theta_postive-theta_negative)'*r_postive)>=0);
        j=j+1;
    end
    
    % adapt step_size
    if iteration <= iteration_adapt
        accept_ratio=alpha/n_alpha;
        H_dual_list(iteration+1)=(1-1/(iteration+t0))*H_dual_list(iteration)+...
            1/(iteration+t0)*(accept_probability-accept_ratio);
        step_list(iteration+1)=exp(mu-sqrt(iteration)/gama*H_dual_list(iteration+1));
        step_dual_list(iteration+1)=exp(iteration^(-kappa)*log(step_list(iteration+1))+...
            (1-iteration^(-kappa))*log(step_dual_list(iteration)));
    else
        step_list(iteration+1)=step_dual_list(iteration_adapt+1);
    end
    
    % record
    sample_list(iteration+1,:)=theta';
    depth(iteration+1)=j;
    log_fval_list(iteration+1)=log_fval;
end
disp('samplerNUTS: sampling done');
sample_list(1,:)=[];

output.depth=depth;
    function [fval,gradient]=normalizeFunction(x,log_sample_function,max_fval)
        [fval,gradient]=log_sample_function(x);
        fval=fval*1e-3;
        gradient=gradient*1e-3;
    end
end
function step=getReasonableStepSize...
    (log_sample_function,theta)
% function to find a appropriate step size
%
step=1;

r=mvnrnd(zeros(length(theta),1),eye(length(theta)))';

log_fval=log_sample_function(theta);
[~,r_temp,log_fval_temp]=leapfrog...
    (log_sample_function,theta,r,step);

energy_delta=(log_fval_temp-0.5*(r_temp'*r_temp))-(log_fval-0.5*(r'*r));

ratio=exp(energy_delta);
a=2*(ratio > 0.5)-1;
while (ratio)^a > 2^(-a)
    step=step*2^a;
    [~,r_temp,log_fval_temp]=leapfrog...
        (log_sample_function,theta,r,step);
    
    energy_delta=(log_fval_temp-0.5*(r_temp'*r_temp))-(log_fval-0.5*(r'*r));
    ratio=exp(energy_delta);
end
end
function [theta_negative,r_negative,theta_postive,r_postive,...
    theta_temp,log_fval_temp,n_temp,s_temp,...
    alpha,n_alpha]=buildTree...
    (log_sample_function,theta,r,u,dir,j,step_size,...
    theta_initial,r_initial,log_fval_initial)
delta_max=1000;

if j == 0
    [theta_temp,r_temp,log_fval_temp]=leapfrog...
        (log_sample_function,theta,r,step_size*dir);
    
    n_temp=(u <= exp( log_fval_temp-0.5*(r_temp'*r_temp) ));
    s_temp=(u < exp( delta_max + log_fval_temp-0.5*(r_temp'*r_temp) ));
    
    % return
    theta_negative=theta_temp;
    r_negative=r_temp;
    theta_postive=theta_temp;
    r_postive=r_temp;
    
    energy_delta=(log_fval_temp-0.5*(r_temp'*r_temp)) -...
        (log_fval_initial-0.5*(r_initial'*r_initial));
    alpha=min(1,exp(energy_delta));
    n_alpha=1;
else
    [theta_negative,r_negative,theta_postive,r_postive,...
        theta_temp,log_fval_temp,n_temp,s_temp,...
        alpha_temp,n_alpha_temp]=buildTree...
        (log_sample_function,theta,r,u,dir,j-1,step_size,...
        theta_initial,r_initial,log_fval_initial);
    
    if s_temp == 1
        if dir == -1
            [theta_negative,r_negative,~,~,...
                theta_tempsq,log_fval_tempsq,n_tempsq,s_tempsq,...
                alpha_tempsq,n_alpha_tempsq]=buildTree...
                (log_sample_function,theta_negative,r_negative,u,dir,j-1,step_size,...
                theta_initial,r_initial,log_fval_initial);
        else
            [~,~,theta_postive,r_postive,...
                theta_tempsq,log_fval_tempsq,n_tempsq,s_tempsq,...
                alpha_tempsq,n_alpha_tempsq]=buildTree...
                (log_sample_function,theta_postive,r_postive,u,dir,j-1,step_size,...
                theta_initial,r_initial,log_fval_initial);
        end
        
        if rand() < n_tempsq/(n_temp+n_tempsq)
            theta_temp=theta_tempsq;
            log_fval_temp=log_fval_tempsq;
        end
        
        alpha_temp=alpha_temp+alpha_tempsq;
        n_alpha_temp=n_alpha_temp+n_alpha_tempsq;
        
        s_temp=s_tempsq*(((theta_postive-theta_negative)'*r_negative)>=0)*...
            (((theta_postive-theta_negative)'*r_postive)>=0);
        n_temp=n_temp+n_tempsq;
    end
    
    alpha=alpha_temp;
    n_alpha=n_alpha_temp;
end
end
function [theta_temp,r_temp,log_fval_temp]=leapfrog...
    (log_sample_function,theta,r,step_size)
% jump
%
[~,gradient]=log_sample_function(theta);
r_temp=r+step_size/2*gradient;
theta_temp=theta+step_size*r_temp;
[log_fval_temp,gradient_temp]=log_sample_function(theta_temp);
r_temp=r_temp+step_size/2*gradient_temp;
end

function [count_list,low_bou,up_bou,X_draw]=drawSample...
    (sample_list,count_number,low_bou,up_bou,figure_handle)
% sample_list is sample_number x variable_number matrix
% low_bou, up_bou is variable_number x 1 matrix
%
if nargin < 5
    figure_handle=figure(75);
    if nargin < 4
        up_bou=[];
        if nargin < 3
            low_bou=[];
        end
    end
end

% boundary
if isempty(up_bou)
    up_bou=max(sample_list)';
end
if isempty(low_bou)
    low_bou=min(sample_list)';
end

[sample_number,variable_number]=size(sample_list);
% every interval count
count_list=zeros(count_number,variable_number);
interval=(up_bou-low_bou)/count_number;

for sample_index=1:sample_number
    sample=sample_list(sample_index,:);
    
    for variable_index=1:variable_number
        count_index=1;
        while (count_index*interval(variable_index)+low_bou(variable_index)...
                < sample(variable_index)) &&...
                (count_index<count_number)
            count_index=count_index+1;
        end
        count_list(count_index,variable_index)=count_list(count_index,variable_index)+1;
    end
end

% draw sample
X_draw=zeros(count_number,variable_number);
for variable_index=1:variable_number
    X_draw(:,variable_index)=...
        (low_bou(variable_index)+interval(variable_index)/2):...
        (interval(variable_index)):...
        (up_bou(variable_index)-interval(variable_index)/2);
end

rank_number=ceil(sqrt(variable_number));
for variable_index=1:variable_number
    axes_handle=axes(figure_handle);
    line(axes_handle,X_draw(:,variable_index),count_list(:,variable_index));
    subplot(rank_number,rank_number,variable_index,axes_handle);
end
end

