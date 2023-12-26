function model_KRG=srgtKRG(X,Y,model_option)
% generate Kriging surrogate model
% input data will be normalize by average and standard deviation of data
%
% input:
% X (matrix): x_num x vari_num matrix
% Y (matrix): x_num x 1 matrix
% model_option (struct): optional input
%
% model_option include:
% optimize_hyp, simplify_hyp, optimize_option, hyp, reg_fcn
%
% output:
% model_KRG (struct): a Kriging model
%
% abbreviation:
% num: number, pred: predict, vari: variable, hyp: hyper parameter
% NLL: negative log likelihood
%
% Copyright 2023.2 Adel
%
if nargin < 3,model_option=struct();end

% Kriging option
if ~isfield(model_option,'optimize_hyp'), model_option.('optimize_hyp')=true;end
if ~isfield(model_option,'simplify_hyp'), model_option.('simplify_hyp')=true;end
if model_option.('simplify_hyp'),FLAG_GRAD=false;else,FLAG_GRAD=true;end
if ~isfield(model_option,'optimize_option')
    model_option.('optimize_option')=optimoptions...
        ('fminunc','Display','none',...
        'OptimalityTolerance',1e-6,...
        'FiniteDifferenceStepSize',1e-5,...,
        'MaxIterations',20,'SpecifyObjectiveGradient',FLAG_GRAD);
end

if ~isfield(model_option,'hyp'), model_option.('hyp')=[];end
if ~isfield(model_option,'reg_fcn'), model_option.('reg_fcn')=[];end

% normalize data
[x_num,vari_num]=size(X);
aver_X=mean(X);
stdD_X=std(X);stdD_X(stdD_X == 0)=1;
aver_Y=mean(Y);
stdD_Y=std(Y);stdD_Y(stdD_Y == 0)=1;
X_nomlz=(X-aver_X)./stdD_X;
Y_nomlz=(Y-aver_Y)./stdD_Y;

% initial X_dis_sq
X_dis_sq=zeros(x_num,x_num,vari_num);
for vari_idx=1:vari_num
    X_dis_sq(:,:,vari_idx)=...
        (X_nomlz(:,vari_idx)-X_nomlz(:,vari_idx)').^2;
end

% regression function define
reg_fcn=model_option.('reg_fcn');
if isempty(reg_fcn)
    % reg_fcn=@(X) ones(size(X,1),1); % zero
    reg_fcn=@(X) [ones(size(X,1),1),X-stdD_X]; % linear
end

% calculate reg
fval_reg=reg_fcn(X);
fval_reg_nomlz=(fval_reg-aver_Y)./stdD_Y;

hyp=model_option.('hyp');
% kernal function is exp(-X_sq/vari_num^2*exp(hyp))
if isempty(hyp), hyp=ones(1,vari_num);end
% if isempty(hyp), hyp=log(x_num^(1/vari_num)*vari_num)*ones(1,vari_num);end

% if optimize hyperparameter
if model_option.optimize_hyp
    simplify_hyp=model_option.('simplify_hyp');
    obj_fcn_hyp=@(hyp) probNLLKRG(X_dis_sq,Y_nomlz,x_num,vari_num,hyp,fval_reg_nomlz);

    if simplify_hyp
        hyp=mean(hyp);
        low_bou_hyp=-4;
        up_bou_hyp=4;
    else
        low_bou_hyp=-4*ones(1,vari_num);
        up_bou_hyp=4*ones(1,vari_num);
    end

    % [fval,gradient]=obj_fcn_hyp(hyp)
    % [~,gradient_differ]=differ(obj_fcn_hyp,hyp)
    % drawFcn(obj_fcn_hyp,low_bou_hyp,up_bou_hyp);

    [hyp,~,~,~]=fminunc(obj_fcn_hyp,hyp,model_option.('optimize_option'));
    hyp=min(hyp,up_bou_hyp);hyp=max(hyp,low_bou_hyp);

    if simplify_hyp, hyp=hyp*ones(1,vari_num);end
end

% get parameter
[~,L_cov,beta,sigma_sq,inv_L_F_reg,~,inv_L_U]=calCovKRG...
    (X_dis_sq,Y_nomlz,x_num,vari_num,exp(hyp),fval_reg_nomlz);
sigma_sq=sigma_sq*stdD_Y^2; % renormalize data
gamma=L_cov'\inv_L_U;
inv_FTcovF=(inv_L_F_reg'*inv_L_F_reg)\eye(size(fval_reg_nomlz,2));

% initialization predict function
pred_fcn=@(X_predict) predictKRG...
    (X_predict,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_num,vari_num,exp(hyp),reg_fcn,...
    L_cov,beta,sigma_sq,gamma,inv_L_F_reg,inv_FTcovF);

model_KRG=model_option;

model_KRG.X=X;
model_KRG.Y=Y;

model_KRG.hyp=hyp;
model_KRG.beta=beta;
model_KRG.gamma=gamma;

model_KRG.predict=pred_fcn;

    function [fval,gradient]=probNLLKRG(X_dis_sq,Y,x_num,vari_num,hyp,F_reg)
        % function to minimize sigma_sq
        %
        hyp=min(hyp,up_bou_hyp);hyp=max(hyp,low_bou_hyp); % Prevent excessive hyp
        theta=exp(hyp);
        if simplify_hyp, theta=theta*ones(1,vari_num);end % extend hyp
        [R,L,Beta,sigma2,inv_L_F]=calCovKRG...
            (X_dis_sq,Y,x_num,vari_num,theta,F_reg);

        if sigma2 == 0
            fval=0;gradient=zeros(vari_num,1);
            if simplify_hyp, gradient=0;end
            return;
        end

        % calculation negative log likelihood
        fval=x_num/2*log(sigma2)+sum(log(diag(L)));

        if nargout > 1
            % calculate gradient
            gradient=zeros(1,vari_num);
            inv_R=L'\(L\eye(x_num));
            inv_FTRF=(inv_L_F'*inv_L_F)\eye(size(F_reg,2));
            Y_Fmiu=Y-F_reg*Beta;

            for vari_i=1:vari_num
                dR_dtheta=-(X_dis_sq(:,:,vari_i).*R)*theta(vari_i)/vari_num^2;
                dinv_R_dtheta=...
                    -inv_R*dR_dtheta*inv_R;
                dinv_FTRF_dtheta=-inv_FTRF*...
                    (F_reg'*dinv_R_dtheta*F_reg)*...
                    inv_FTRF;
                dmiu_dtheta=dinv_FTRF_dtheta*(F_reg'*inv_R*Y)+...
                    inv_FTRF*(F_reg'*dinv_R_dtheta*Y);
                dU_dtheta=-F_reg*dmiu_dtheta;
                dsigma2_dtheta=(dU_dtheta'*inv_R*Y_Fmiu+...
                    Y_Fmiu'*dinv_R_dtheta*Y_Fmiu+...
                    Y_Fmiu'*inv_R*dU_dtheta)/x_num;
                dlnsigma2_dtheta=1/sigma2*dsigma2_dtheta;
                dlndetR=trace(inv_R*dR_dtheta);

                gradient(vari_i)=x_num/2*dlnsigma2_dtheta+0.5*dlndetR;
            end

            if simplify_hyp, gradient=sum(gradient);end
        end
    end

    function [cov,L_cov,beta,sigma_sq,inv_L_F_reg,inv_L_Y,inv_L_U]=calCovKRG...
            (X_dis_sq,Y,x_num,vari_num,theta,F_reg)
        % kriging interpolation kernel function
        % Y(x)=beta+Z(x)
        %

        % calculate covariance
        cov=zeros(x_num,x_num);
        for vari_i=1:vari_num
            cov=cov+X_dis_sq(:,:,vari_i)*theta(vari_i);
        end
        cov=exp(-cov/vari_num^2)+eye(x_num)*((1000+x_num)*eps);

        % coefficient calculation
        L_cov=chol(cov)'; % inv_cov=cov\eye(x_num);
        inv_L_F_reg=L_cov\F_reg;
        inv_L_Y=L_cov\Y; % inv_FTRF=(F_reg'*inv_cov*F_reg)\eye(size(F_reg,2));

        % basical bias
        beta=inv_L_F_reg\inv_L_Y; % beta=inv_FTRF*(F_reg'*inv_cov*Y);
        inv_L_U=inv_L_Y-inv_L_F_reg*beta; % U=Y-F_reg*beta;
        sigma_sq=sum(inv_L_U.^2)/x_num; % sigma_sq=(U'*inv_cov*U)/x_num;
    end

    function [Y_pred,Var_pred]=predictKRG...
            (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,theta,reg_fcn,...
            L_cov,beta,sigma_sq,gamma,inv_L_F_reg,inv_FTcovF)
        % Kriging surrogate predict function
        %
        % input:
        % X_pred (matrix): x_pred_num x vari_num matrix, predict X
        %
        % output:
        % Y_pred (matrix): x_pred_num x 1 matrix, value
        % Var_pred (matrix): x_pred_num x 1 matrix, variance
        %
        [x_pred_num,~]=size(X_pred);

        % normalize data
        X_pred_nomlz=(X_pred-aver_X)./stdD_X;

        % regression value
        fval_reg_pred_nomlz=(reg_fcn(X_pred)-aver_Y)./stdD_Y;

        % predict covariance
        cov_pred=zeros(x_num,x_pred_num);
        for vari_i=1:vari_num
            cov_pred=cov_pred+(X_nomlz(:,vari_i)-X_pred_nomlz(:,vari_i)').^2*theta(vari_i);
        end
        cov_pred=exp(-cov_pred/vari_num^2);

        % predict base fval
        Y_pred=fval_reg_pred_nomlz*beta+cov_pred'*gamma;

        % predict variance
        inv_L_r=L_cov\cov_pred;
        u=(inv_L_F_reg)'*inv_L_r-fval_reg_pred_nomlz';
        Var_pred=sigma_sq*(1+u'*inv_FTcovF*u-inv_L_r'*inv_L_r);
        Var_pred=diag(Var_pred);

        % renormalize data
        Y_pred=Y_pred*stdD_Y+aver_Y;
    end

end
