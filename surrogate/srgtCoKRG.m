function model_CoKRG=srgtCoKRG(X_LF,Y_LF,X_HF,Y_HF,model_option)
% generate Co-Kriging surrogate model
% input data will be normalize by average and standard deviation of data
% hyp include: theta_bias, rho, theta_LF
% notice: theta=exp(hyp), rho=exp(hyp)
%
% input:
% X_HF (matrix): x_HF_num x vari_num matrix
% Y_HF (matrix): x_HF_num x 1 matrix
% X_LF (matrix): x_LF_num x vari_num matrix
% Y_LF (matrix): x_LF_num x 1 matrix
% model_option (struct): optional input, include: optimize_hyp, simplify_hyp, hyp_bias, hyp_LF, optimize_option
%
% output:
% model_CoKRG (struct): a Co-Kriging model
%
% abbreviation:
% num: number, pred: predict, vari: variable, hyp: hyper parameter
% NLL: negative log likelihood
%
% reference: [1] FORRESTER A I J, SóBESTER A, KEANE A J. Multi-fidelity
% optimization via surrogate modelling [J]. Proceedings of the Royal
% Society A: Mathematical, Physical and Engineering Sciences, 2007,
% 463(3251-69.
%
% Copyright 2023.2 Adel
%
if nargin < 5,model_option=struct();end

% Kriging option
if isempty(model_option), model_option=struct();end
if ~isfield(model_option,'optimize_hyp'), model_option.('optimize_hyp')=true;end
if ~isfield(model_option,'simplify_hyp'), model_option.('simplify_hyp')=true;end
if model_option.('simplify_hyp'),FLAG_GRAD=false;else,FLAG_GRAD=true;end
if ~isfield(model_option,'model_option'), model_option.('optimize_option')=optimoptions...
        ('fminunc','Display','none',...
        'OptimalityTolerance',1e-6,...
        'FiniteDifferenceStepSize',1e-5,...,
        'MaxIterations',20,'SpecifyObjectiveGradient',FLAG_GRAD);end

% Co-Kriging option
if ~isfield(model_option,'hyp_bias'), model_option.('hyp_bias')=[];end
if ~isfield(model_option,'hyp_LF'), model_option.('hyp_LF')=[];end

% normalize data
X=[X_LF;X_HF];
Y=[Y_LF;Y_HF];
[x_num,vari_num]=size(X);
x_LF_num=size(X_LF,1);
x_HF_num=size(X_HF,1);
aver_X=mean(X);
stdD_X=std(X);stdD_X(stdD_X == 0)=1;
aver_Y=mean(Y);
stdD_Y=std(Y);stdD_Y(stdD_Y == 0)=1;
X_LF_nomlz=(X_LF-aver_X)./stdD_X;
Y_LF_nomlz=(Y_LF-aver_Y)./stdD_Y;
X_HF_nomlz=(X_HF-aver_X)./stdD_X;
Y_HF_nomlz=(Y_HF-aver_Y)./stdD_Y;
X_nomlz=[X_LF_nomlz;X_HF_nomlz];
Y_nomlz=[Y_LF_nomlz;Y_HF_nomlz];

% all initial X_dis_sq
X_dis_sq=zeros(x_num,x_num,vari_num);
for vari_idx=1:vari_num
    X_dis_sq(:,:,vari_idx)=(X_nomlz(:,vari_idx)-X_nomlz(:,vari_idx)').^2;
end
X_LF_dis_sq=X_dis_sq(1:x_LF_num,1:x_LF_num,:);
X_HF_dis_sq=X_dis_sq(x_LF_num+1:end,x_LF_num+1:end,:);

% regression function define
% reg_fcn=@(X) ones(size(X,1),1).*stdD_Y+aver_Y; % zero
reg_fcn=@(X) [ones(size(X,1),1),X-stdD_X].*stdD_Y+aver_Y; % linear

hyp_bias=model_option.('hyp_bias');
if isempty(hyp_bias),hyp_bias=[ones(1,vari_num),mean(Y_HF)/mean(Y_LF)];end
hyp_LF=model_option.('hyp_LF');
if isempty(hyp_LF),hyp_LF=ones(1,vari_num);end
simplify_hyp=model_option.('simplify_hyp');

% first step
% construct low-fidelity model

% calculate reg
fval_reg_nomlz_LF=(reg_fcn(X_LF)-aver_Y)./stdD_Y;

% if optimize hyperparameter
if model_option.optimize_hyp
    obj_fcn_hyp=@(hyp) probNLLKRG(X_LF_dis_sq,Y_LF_nomlz,x_LF_num,vari_num,hyp,fval_reg_nomlz_LF);

    if simplify_hyp
        hyp_LF=mean(hyp_LF);
        low_bou_hyp=-4;
        up_bou_hyp=4;
    else
        low_bou_hyp=-4*ones(1,vari_num);
        up_bou_hyp=4*ones(1,vari_num);
    end

    % [fval,gradient]=obj_fcn_hyp(hyp_LF)
    % [~,gradient_differ]=differ(obj_fcn_hyp,hyp_LF)
    % drawFcn(obj_fcn_hyp,low_bou_hyp,up_bou_hyp);

    [hyp_LF,~,~,~]=fminunc(obj_fcn_hyp,hyp_LF,model_option.('optimize_option'));
    hyp_LF=min(hyp_LF,up_bou_hyp);hyp_LF=max(hyp_LF,low_bou_hyp);

    if simplify_hyp, hyp_LF=hyp_LF*ones(1,vari_num);end
end

% get parameter
[cov_LF,L_cov_LF,beta_LF,sigma_sq_LF,inv_L_F_reg_LF,~,inv_L_U_LF]=calCovKRG...
    (X_LF_dis_sq,Y_LF_nomlz,x_LF_num,vari_num,exp(hyp_LF),fval_reg_nomlz_LF);
sigma_sq_LF=sigma_sq_LF*stdD_Y^2; % renormalize data
gamma_LF=L_cov_LF'\inv_L_U_LF;
inv_FTcovF_LF=(inv_L_F_reg_LF'*inv_L_F_reg_LF)\eye(size(fval_reg_nomlz_LF,2));

% initialization predict function
pred_fcn_LF=@(X_predict) predictKRG...
    (X_predict,X_LF_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_LF_num,vari_num,exp(hyp_LF),reg_fcn,...
    L_cov_LF,beta_LF,sigma_sq_LF,gamma_LF,inv_L_F_reg_LF,inv_FTcovF_LF);

% second step
% construct bias model
% notice rho is hyperparamter

% evaluate error in high fidelity point
Y_HF_pred_nomlz=(pred_fcn_LF(X_HF)-aver_Y)./stdD_Y;

% calculate reg
fval_reg_nomlz_bias=(reg_fcn(X_HF)-aver_Y)./stdD_Y;

% optimal to get hyperparameter
% if optimize hyperparameter
if model_option.optimize_hyp
    obj_fcn_hyp=@(hyp) probNLLBiasKRG(X_HF_dis_sq,Y_HF_nomlz,Y_HF_pred_nomlz,x_HF_num,vari_num,hyp,fval_reg_nomlz_bias);

    if simplify_hyp
        hyp_bias=[mean(hyp_bias(1:end-1)),hyp_bias(end)];
        low_bou_hyp=[-4,-10];
        up_bou_hyp=[4,10];
    else
        low_bou_hyp=[-4*ones(1,vari_num),-10];
        up_bou_hyp=[4*ones(1,vari_num),10];
    end

    % [fval,gradient]=obj_fcn_hyp(hyp_bias)
    % [~,gradient_differ]=differ(obj_fcn_hyp,hyp_bias)
    % drawFcn(obj_fcn_hyp,low_bou_hyp,up_bou_hyp);

    [hyp_bias,~,~,~]=fminunc(obj_fcn_hyp,hyp_bias,model_option.('optimize_option'));
    hyp_bias=min(hyp_bias,up_bou_hyp);hyp_bias=max(hyp_bias,low_bou_hyp);

    if simplify_hyp, hyp_bias=[hyp_bias(1)*ones(1,vari_num),hyp_bias(2)];end
end

% get parameter
Y_bias_nomlz=Y_HF_nomlz-hyp_bias(end)*Y_HF_pred_nomlz;
[~,~,~,sigma_sq_bias,~,~,~]=calCovKRG...
    (X_HF_dis_sq,Y_bias_nomlz,x_HF_num,vari_num,exp(hyp_bias),fval_reg_nomlz_bias);
sigma_sq_bias=sigma_sq_bias*stdD_Y^2; % renormalize data

% get total model parameter
fval_reg_nomlz=[
    fval_reg_nomlz_LF,zeros(size(fval_reg_nomlz_LF,1),size(fval_reg_nomlz_bias,2));
    fval_reg_nomlz_bias*hyp_bias(end),fval_reg_nomlz_bias;];
[~,L_cov,beta,~,inv_L_F_reg,~,inv_L_U]=calCovCoKRG...
    (X_dis_sq,Y_nomlz,x_num,x_HF_num,x_LF_num,vari_num,...
    exp(hyp_bias(1:end-1)),hyp_bias(end),exp(hyp_LF),...
    sigma_sq_bias,sigma_sq_LF,fval_reg_nomlz);
gamma=L_cov'\inv_L_U;
inv_FTcovF=(inv_L_F_reg'*inv_L_F_reg)\eye(size(fval_reg_nomlz,2));

% initialization predict function
pred_fcn=@(X_pred) predictCoKRG...
    (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_num,vari_num,x_HF_num,x_LF_num,exp(hyp_bias(1:end-1)),hyp_bias(end),exp(hyp_LF),reg_fcn,...
    L_cov,beta,sigma_sq_bias,sigma_sq_LF,gamma,inv_L_F_reg,inv_FTcovF);

model_CoKRG.X={X_LF,X_HF};
model_CoKRG.Y={Y_LF,Y_HF};

model_CoKRG.hyp_LF=hyp_LF;
model_CoKRG.hyp_bias=hyp_bias;

model_CoKRG.predict_list={pred_fcn_LF;pred_fcn};

model_CoKRG.predict=pred_fcn;

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
        if sigma2 == 0,fval=0;
        else,fval=x_num/2*log(sigma2)+sum(log(diag(L)));end

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

    function [fval,gradient]=probNLLBiasKRG(X_dis_sq,Y,Y_pred,x_num,vari_num,hyp,F_reg)
        % function to minimize negative likelihood
        %
        hyp=min(hyp,up_bou_hyp);hyp=max(hyp,low_bou_hyp);
        theta=exp(hyp(1:end-1));
        if simplify_hyp, theta=theta*ones(1,vari_num);end % extend hyp
        rho=hyp(end);
        Y_bias=Y-rho*Y_pred; % notice Y equal to d

        [R,L,Beta,sigma2,inv_L_F]=calCovKRG(X_dis_sq,Y_bias,x_num,vari_num,theta,F_reg);

        % calculation negative log likelihood
        if sigma2 == 0,fval=0;
        else,fval=x_num/2*log(sigma2)+sum(log(diag(L)));end

        % calculate gradient
        if nargout > 1
            % gradient
            gradient=zeros(vari_num+1,1);
            inv_R=L'\(L\eye(x_num));
            inv_FTRF=(inv_L_F'*inv_L_F)\eye(size(F_reg,2));
            Y_Fmiu=Y-F_reg*Beta;

            % theta d
            for vari_i=1:vari_num
                dR_dtheta=-(X_dis_sq(:,:,vari_i).*R)*theta(vari_i)/vari_num;
                dinv_R_dtheta=...
                    -inv_R*dR_dtheta*inv_R;
                dinv_FTRF_dtheta=-inv_FTRF*...
                    (F_reg'*dinv_R_dtheta*F_reg)*...
                    inv_FTRF;
                dbeta_dtheta=dinv_FTRF_dtheta*(F_reg'*inv_R*Y_bias)+...
                    inv_FTRF*(F_reg'*dinv_R_dtheta*Y_bias);
                dY_Fmiu_dtheta=-F_reg*dbeta_dtheta;
                dsigma2_dtheta=(dY_Fmiu_dtheta'*inv_R*Y_Fmiu+...
                    Y_Fmiu'*dinv_R_dtheta*Y_Fmiu+...
                    Y_Fmiu'*inv_R*dY_Fmiu_dtheta)/x_num;
                dlnsigma2_dtheta=1/sigma2*dsigma2_dtheta;
                dlndetR=trace(inv_R*dR_dtheta);

                gradient(vari_i)=x_num/2*dlnsigma2_dtheta+0.5*dlndetR;
            end

            % rho
            dY_drho=-Y_pred*rho;
            dbeta_drho=inv_FTRF*(F_reg'*inv_R*dY_drho);
            dY_Fmiu_drho=(dY_drho-F_reg*dbeta_drho);
            dsigma2_drho=(dY_Fmiu_drho'*inv_R*Y_Fmiu+...
                Y_Fmiu'*inv_R*dY_Fmiu_drho)/x_num;
            dlnsigma2_drho=1/sigma2*dsigma2_drho;

            gradient(end)=x_num/2*dlnsigma2_drho;

            if simplify_hyp, gradient=[sum(gradient(1:end-1)),gradient(end)];end
        end
    end

    function [cov,L_cov,beta,sigma_sq,inv_L_F_reg,inv_L_Y,inv_L_U]=calCovCoKRG...
            (X_dis_sq,Y,x_num,x_HF_num,x_LF_num,vari_num,...
            theta_bias,rho,theta_LF,sigma2_bias,sigma2_LF,F_reg)
        % calculate covariance of x with multi fidelity
        % hyp: theta_H, theta_L, rho
        %

        % exp of x__x with theta H
        cov_H=zeros(x_HF_num);
        for vari_i=1:vari_num
            cov_H=cov_H+X_dis_sq(x_LF_num+1:end,x_LF_num+1:end,vari_i)*theta_bias(vari_i);
        end
        cov_H=exp(-cov_H/vari_num)*sigma2_bias;

        % exp of x__x with theta L
        exp_disL=zeros(x_num);
        for vari_i=1:vari_num
            exp_disL=exp_disL+...
                X_dis_sq(:,:,vari_i)*theta_LF(vari_i);
        end
        exp_disL=exp(-exp_disL/vari_num)*sigma2_LF;
        % times rho: HH to rho2, HL to rho, LL to 1
        cov_L=exp_disL;
        cov_L(x_LF_num+1:end,x_LF_num+1:end)=...
            (rho*rho)*exp_disL(x_LF_num+1:end,x_LF_num+1:end);
        cov_L(x_LF_num+1:end,1:x_LF_num)=...
            rho*exp_disL(x_LF_num+1:end,1:x_LF_num);
        cov_L(1:x_LF_num,x_LF_num+1:end)=...
            cov_L(x_LF_num+1:end,1:x_LF_num)';

        cov=cov_L;
        cov(x_LF_num+1:end,x_LF_num+1:end)=cov(x_LF_num+1:end,x_LF_num+1:end)+cov_H;

        cov=cov+eye(x_num)*((1000+x_num)*eps);

        % coefficient calculation
        L_cov=chol(cov)'; % inv_cov=cov\eye(x_num);
        inv_L_F_reg=L_cov\F_reg;
        inv_L_Y=L_cov\Y; % inv_FTRF=(F_reg'*inv_cov*F_reg)\eye(size(F_reg,2));

        % basical bias
        beta=inv_L_F_reg\inv_L_Y; % beta=inv_FTRF*(F_reg'*inv_cov*Y);
        inv_L_U=inv_L_Y-inv_L_F_reg*beta; % U=Y-F_reg*beta;
        sigma_sq=(inv_L_U'*inv_L_U)/x_num; % sigma_sq=(U'*inv_cov*U)/x_num;
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

    function [Y_pred,Var_pred]=predictCoKRG...
            (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,x_HF_num,x_LF_num,theta_bias,rho,theta_LF,reg_fcn,...
            L_cov,beta,sigma_sq_bias,sigma_sq_LF,gamma,inv_L_F_reg,inv_FTcovF)
        % kriging surrogate predict function
        % input predict_x and kriging model
        % predict_x is row vector
        % output the predict value
        %
        [x_pred_num,~]=size(X_pred);
        fval_reg_pred_nomlz=[(reg_fcn(X_pred)-aver_Y)./stdD_Y*rho,...
            (reg_fcn(X_pred)-aver_Y)./stdD_Y];

        % normalize data
        X_pred_nomlz=(X_pred-aver_X)./stdD_X;

        % distance sq of X_predict
        X_pred_dis_sq=zeros(x_num,x_pred_num,vari_num);
        for vari_i=1:vari_num
            X_pred_dis_sq(:,:,vari_i)=(X_nomlz(:,vari_i)-X_pred_nomlz(:,vari_i)').^2;
        end

        % bias
        exp_dis_bias=zeros(x_HF_num,x_pred_num);
        for vari_i=1:vari_num
            exp_dis_bias=exp_dis_bias+X_pred_dis_sq(x_LF_num+1:end,:,vari_i)*theta_bias(vari_i);
        end
        exp_dis_bias=exp(-exp_dis_bias/vari_num);

        % LF
        exp_dis_LF=zeros(x_num,x_pred_num);
        for vari_i=1:vari_num
            exp_dis_LF=exp_dis_LF+X_pred_dis_sq(:,:,vari_i)*theta_LF(vari_i);
        end
        exp_dis_LF=exp(-exp_dis_LF/vari_num);

        % covariance of X_predict
        cov_pred=exp_dis_LF*rho*sigma_sq_LF;
        cov_pred(x_LF_num+1:end,:)=cov_pred(x_LF_num+1:end,:)*rho+...
            sigma_sq_bias*exp_dis_bias;

        % predict base fval
        Y_pred=fval_reg_pred_nomlz*beta+cov_pred'*gamma;

        % predict variance
        inv_L_r=L_cov\cov_pred;
        u=(inv_L_F_reg)'*inv_L_r-fval_reg_pred_nomlz';
        Var_pred=sigma_sq_LF*rho*rho+sigma_sq_bias+u'*inv_FTcovF*u-inv_L_r'*inv_L_r;
        Var_pred=diag(Var_pred);

        % normalize data
        Y_pred=Y_pred*stdD_Y+aver_Y;
    end

end
