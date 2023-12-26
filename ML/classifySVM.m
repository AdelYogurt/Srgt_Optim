function model_SVM=classifySVM(X,Y,model_option)
% generate support vector machine model
% use fmincon to get alpha
%
% input:
% X(x_num x vari_num matrix), Y(x_num x 1 matrix),...
% model_option(optional, include: box_con, kernel_fcn, optimize_option)
%
% output:
% model_SVM(a support vector machine model)
%
% abbreviation:
% num: number, pred: predict, vari: variable
% nomlz: normalization, var: variance, fcn: function
%
if nargin < 3,model_option=struct();end
if ~isfield(model_option,'box_con'), model_option.('box_con')=1;end
if ~isfield(model_option,'kernel_fcn'), model_option.('kernel_fcn')=[];end
if ~isfield(model_option,'optimize_option'), model_option.('optimize_option')=optimoptions...
        ('quadprog','Display','none');end

% normalization data
[x_num,vari_num]=size(X);
aver_X=mean(X);
stdD_X=std(X);stdD_X(stdD_X == 0)=1;
X_nomlz=(X-aver_X)./stdD_X;
Y(Y == 0)=-1;

% default kernal function
kernel_fcn=model_option.kernel_fcn;
if isempty(kernel_fcn)
    % notice after standard normal distribution normalize
    sigma=x_num/vari_num;
    kernel_fcn=@(U,V) kernelGaussian(U,V,sigma);
end

% initialization kernal function process X_cov
K=kernel_fcn(X_nomlz,X_nomlz);
box_con=model_option.box_con;
alpha=solveAlpha(K,Y,x_num,box_con,model_option);

% obtain other paramter
alpha_Y=alpha.*Y;
w=sum(alpha_Y.*X_nomlz);
bool_support=(alpha > 1e-6); % support vector
alpha_Y_cov=K*alpha_Y;
b=sum(Y(bool_support)-alpha_Y_cov(bool_support))/length(bool_support);

% generate predict function
pred_fcn=@(X_pred) predictSVM(X_pred,X_nomlz,alpha_Y,b,aver_X,stdD_X,kernel_fcn);

% output model
model_SVM.X=X;
model_SVM.Y=Y;
model_SVM.aver_X=aver_X;
model_SVM.stdD_X=stdD_X;

model_SVM.alpha=alpha;
model_SVM.bool_support=bool_support;
model_SVM.w=w;
model_SVM.b=b;

model_SVM.box_con=box_con;
model_SVM.kernel=kernel_fcn;
model_SVM.predict=pred_fcn;

    function [Y_pred,Prob_pred]=predictSVM...
            (X_pred,X_nomlz,alpha_Y,b,aver_X,stdD_X,kernel_fcn)
        % SVM predict function
        %
        [x_pred_num,~]=size(X_pred);
        X_pred_nomlz=(X_pred-aver_X)./stdD_X;

        K_pred=kernel_fcn(X_pred_nomlz,X_nomlz);
        Prob_pred=K_pred*alpha_Y+b;

        Prob_pred=1./(1+exp(-Prob_pred));
        Y_pred=ones(x_pred_num,1);
        Y_pred(Prob_pred < 0.5)=0;
    end

    function K=kernelGaussian(U,V,sigma)
        % gaussian kernal function
        %
        K=zeros(size(U,1),size(V,1));
        for vari_index=1:size(U,2)
            K=K+(U(:,vari_index)-V(:,vari_index)').^2;
        end
        K=exp(-K*sigma);
    end

    function alpha=solveAlpha(K,Y,x_num,box_con,model_option)
        % min SVM object function to get alpha
        %
        
        % solve alpha need to minimize followed equation
        % obj=sum(alpha)-(alpha.*Y)'*K*(alpha.*Y)/2;

        alpha=ones(x_num,1)*0.5;
        low_bou_A=0*ones(x_num,1);
        if isempty(box_con) || box_con==0
            up_bou_A=[];
        else
            up_bou_A=box_con*ones(x_num,1);
        end
        Aeq=Y';
        hess=-K.*(Y*Y');
        grad=ones(x_num,1);
        alpha=quadprog(hess,grad,[],[],Aeq,0,low_bou_A,up_bou_A,alpha,model_option.optimize_option);
    end

end
