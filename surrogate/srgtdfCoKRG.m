function srgt=srgtdfCoKRG(X_LF,Y_LF,X_HF,Y_HF,option)
% generate Co-Kriging surrogate model
% input data will be normalize by average and standard deviation of data
%
% input:
% X_LF (matrix): low fidelity trained X, x_num x vari_num
% Y_LF (vector): low fidelity trained Y, x_num x 1
% X_HF (matrix): high fidelity trained X, x_num x vari_num
% Y_HF (vector): high fidelity trained Y, x_num x 1
% option (struct): optional input, construct option
%
% option include:
% optimize_hyp (boolean): whether optimize hyperparameter
% simplify_hyp (boolean): whether simplify multi hyperparameter to one
% optimize_option (optimoptions): fminunc optimize option
% reg_fcn (function handle): basis function, default is linear
% cov_fcn (function handle): kernel function, default is gauss
% hyp_LF (array): hyperparameter value of kernel function for low fidelity
% hyp_bias (array): hyperparameter value of kernel function for bias
%
% output:
% srgt(struct): a Co-Kriging model
%
% reference:
% [1] Forrester A I J, Sóbester A, Keane A J. Multi-Fidelity Optimization
% via Surrogate Modelling[J]. Proceedings of the Royal Society A:
% Mathematical, Physical and Engineering Sciences, 2007, 463: 3251-3269.
%
% Copyright 2023.2 Adel
%
if nargin < 5,option=struct();end

% Kriging option
if isempty(option), option=struct();end
if ~isfield(option,'optimize_hyp'), option.('optimize_hyp')=true;end
if ~isfield(option,'simplify_hyp'), option.('simplify_hyp')=true;end
if option.('simplify_hyp'),FLAG_GRAD=false;else,FLAG_GRAD=true;end
if ~isfield(option,'optimize_option'), option.('optimize_option')=optimoptions...
        ('fminunc','Display','none',...
        'OptimalityTolerance',1e-6,...
        'FiniteDifferenceStepSize',1e-5,...
        'MaxIterations',20,'SpecifyObjectiveGradient',FLAG_GRAD);
end

if ~isfield(option,'reg_fcn'), option.('reg_fcn')=[];end
if ~isfield(option,'cov_fcn'), option.('cov_fcn')=[];end

% Co-Kriging option
if ~isfield(option,'hyp_bias'), option.('hyp_bias')=[];end
if ~isfield(option,'hyp_LF'), option.('hyp_LF')=[];end

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
X_LF_norm=(X_LF-aver_X)./stdD_X;
Y_LF_norm=(Y_LF-aver_Y)./stdD_Y;
X_HF_norm=(X_HF-aver_X)./stdD_X;
Y_HF_norm=(Y_HF-aver_Y)./stdD_Y;
X_norm=[X_LF_norm;X_HF_norm];
Y_norm=[Y_LF_norm;Y_HF_norm];

% all initial X_dis_sq
dr_sq=zeros(x_num,x_num,vari_num);
for vari_idx=1:vari_num
    dr_sq(:,:,vari_idx)=(X_norm(:,vari_idx)-X_norm(:,vari_idx)').^2;
end
dr_sq_LF=dr_sq(1:x_LF_num,1:x_LF_num,:);
dr_sq_HF=dr_sq(x_LF_num+1:end,x_LF_num+1:end,:);

% regression function define
reg_fcn=option.('reg_fcn');
if isempty(reg_fcn)
    if x_num < vari_num,reg_fcn=@(X) ones(size(X,1),1).*stdD_Y+aver_Y; % constant
    else,reg_fcn=@(X) [ones(size(X,1),1),X-aver_X].*stdD_Y+aver_Y;end % linear
end

% hyperparameter define
hyp_bias=option.('hyp_bias');
if isempty(hyp_bias),hyp_bias=[zeros(1,vari_num),mean(Y_HF)/mean(Y_LF)];end
hyp_LF=option.('hyp_LF');
if isempty(hyp_LF),hyp_LF=ones(1,vari_num);end
simplify_hyp=option.('simplify_hyp');

% first step
% construct low-fidelity model

% calculate reg
HLF_norm=(reg_fcn(X_LF)-aver_Y)./stdD_Y;

% if optimize hyperparameter
if option.optimize_hyp
    obj_fcn_hyp=@(hyp) probNLLKRG(dr_sq_LF,Y_LF_norm,x_LF_num,vari_num,hyp,HLF_norm);

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

    [hyp_LF,~,~,~]=fminunc(obj_fcn_hyp,hyp_LF,option.('optimize_option'));
    hyp_LF=min(hyp_LF,up_bou_hyp);hyp_LF=max(hyp_LF,low_bou_hyp);

    if simplify_hyp, hyp_LF=hyp_LF*ones(1,vari_num);end
end

% get parameter
[cov_LF,L_cov_LF,beta_LF,sigma_sq_LF,dLR_H_LF,~,dLR_U_LF]=calCovKRG...
    (dr_sq_LF,Y_LF_norm,x_LF_num,vari_num,exp(hyp_LF),HLF_norm);
sigma_sq_LF=sigma_sq_LF*stdD_Y^2; % renormalize data
gamma_LF=L_cov_LF'\dLR_U_LF;
inv_FTcovF_LF=(dLR_H_LF'*dLR_H_LF)\eye(size(HLF_norm,2));

% initialization predict function
pred_fcn_LF=@(X_predict) predictKRG...
    (X_predict,X_LF_norm,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_LF_num,vari_num,exp(hyp_LF),reg_fcn,...
    L_cov_LF,beta_LF,sigma_sq_LF,gamma_LF,dLR_H_LF,inv_FTcovF_LF);

% second step
% construct bias model
% notice rho is hyperparamter

% evaluate error in high fidelity point
Y_HF_pred_norm=(pred_fcn_LF(X_HF)-aver_Y)./stdD_Y;

% calculate reg
HD_norm=(reg_fcn(X_HF)-aver_Y)./stdD_Y;

% optimal to get hyperparameter
% if optimize hyperparameter
if option.optimize_hyp
    obj_fcn_hyp=@(hyp) probNLLBiasKRG(dr_sq_HF,Y_HF_norm,Y_HF_pred_norm,x_HF_num,vari_num,hyp,HD_norm);

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

    [hyp_bias,~,~,~]=fminunc(obj_fcn_hyp,hyp_bias,option.('optimize_option'));
    hyp_bias=min(hyp_bias,up_bou_hyp);hyp_bias=max(hyp_bias,low_bou_hyp);

    if simplify_hyp, hyp_bias=[hyp_bias(1)*ones(1,vari_num),hyp_bias(2)];end
end

% get parameter
Y_bias_norm=Y_HF_norm-hyp_bias(end)*Y_HF_pred_norm;
[cov_bias,L_cov_bias,~,sigma_sq_bias,~,~,~]=calCovKRG...
    (dr_sq_HF,Y_bias_norm,x_HF_num,vari_num,exp(hyp_bias),HD_norm);
sigma_sq_bias=sigma_sq_bias*stdD_Y^2; % renormalize data

% get total model parameter
H_norm=[
    HLF_norm,zeros(size(HLF_norm,1),size(HD_norm,2));
    HD_norm*hyp_bias(end),HD_norm;];
[cov,L_cov,beta,~,dLR_H,~,dLR_U]=calCovCoKRG...
    (dr_sq,Y_norm,x_num,x_HF_num,x_LF_num,vari_num,...
    exp(hyp_bias(1:end-1)),hyp_bias(end),exp(hyp_LF),...
    sigma_sq_bias,sigma_sq_LF,H_norm);
gamma=L_cov'\dLR_U;
inv_FTcovF=(dLR_H'*dLR_H)\eye(size(H_norm,2));

% initialization predict function
pred_fcn=@(X_pred) predictCoKRG...
    (X_pred,X_norm,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_num,vari_num,x_HF_num,x_LF_num,exp(hyp_bias(1:end-1)),hyp_bias(end),exp(hyp_LF),reg_fcn,...
    L_cov,beta,sigma_sq_bias,sigma_sq_LF,gamma,dLR_H,inv_FTcovF);

srgt=option;
srgt.X={X_LF,X_HF};
srgt.Y={Y_LF,Y_HF};
srgt.hyp_LF=hyp_LF;
srgt.hyp_bias=hyp_bias;
srgt.predict_list={pred_fcn_LF;pred_fcn};
srgt.predict=pred_fcn;

    function [fval,grad]=probNLLKRG(X_dis_sq,Y,x_num,vari_num,hyp,H)
        % function to minimize sigma_sq
        %
        hyp=min(hyp,up_bou_hyp);hyp=max(hyp,low_bou_hyp); % Prevent excessive hyp
        theta=exp(hyp);
        if simplify_hyp, theta=theta*ones(1,vari_num);end % extend hyp
        [R,L,B,s2,dLR_F]=calCovKRG...
            (X_dis_sq,Y,x_num,vari_num,theta,H);

        if s2 == 0
            fval=0;grad=zeros(vari_num,1);
            if simplify_hyp, grad=0;end
            return;
        end

        % calculation negative log likelihood
        if s2 == 0,fval=0;
        else,fval=(x_num*log(s2)+2*sum(log(diag(L))))/2;end

        if nargout > 1
            % calculate gradient
            grad=zeros(1,vari_num);
            invR=L'\(L\eye(x_num));
            invHiRH=(dLR_F'*dLR_F)\eye(size(H,2));
            Y_Fmiu=Y-H*B;

            for vari_i=1:vari_num
                dR_dtheta=-(X_dis_sq(:,:,vari_i).*R)*theta(vari_i)/vari_num^2;
                dinvR_dtheta=...
                    -invR*dR_dtheta*invR;
                dinvHiRH_dtheta=-invHiRH*...
                    (H'*dinvR_dtheta*H)*...
                    invHiRH;
                dmiu_dtheta=dinvHiRH_dtheta*(H'*invR*Y)+...
                    invHiRH*(H'*dinvR_dtheta*Y);
                dU_dtheta=-H*dmiu_dtheta;
                dsigma2_dtheta=(dU_dtheta'*invR*Y_Fmiu+...
                    Y_Fmiu'*dinvR_dtheta*Y_Fmiu+...
                    Y_Fmiu'*invR*dU_dtheta)/x_num;
                dlnsigma2_dtheta=1/s2*dsigma2_dtheta;
                dlndetR=trace(invR*dR_dtheta);

                grad(vari_i)=x_num/2*dlnsigma2_dtheta+0.5*dlndetR;
            end

            if simplify_hyp, grad=sum(grad);end
        end
    end

    function [cov,L_cov,beta,sigma_sq,dLR_H,dLR_Y,dLR_U]=calCovKRG...
            (X_dis_sq,Y,x_num,vari_num,theta,H)
        % kriging interpolation kernel function
        % y(x)=f(x)+z(x)
        %

        % calculate covariance
        cov=zeros(x_num,x_num);
        for vari_i=1:vari_num
            cov=cov+X_dis_sq(:,:,vari_i)*theta(vari_i);
        end
        cov=exp(-cov/vari_num^2)+eye(x_num)*((1000+x_num)*eps);

        % coefficient calculation
        L_cov=chol(cov)'; % inv_cov=cov\eye(x_num);
        dLR_H=L_cov\H;
        dLR_Y=L_cov\Y; % invHiRH=(H'*inv_cov*H)\eye(size(H,2));

        % basical bias
        beta=dLR_H\dLR_Y; % beta=invHiRH*(H'*inv_cov*Y);
        dLR_U=dLR_Y-dLR_H*beta; % U=Y-H*beta;
        sigma_sq=sum(dLR_U.^2)/x_num; % sigma_sq=(U'*inv_cov*U)/x_num;
    end

    function [fval,grad]=probNLLBiasKRG(X_dis_sq,Y,Y_pred,x_num,vari_num,hyp,H)
        % function to minimize negative likelihood
        %
        hyp=min(hyp,up_bou_hyp);hyp=max(hyp,low_bou_hyp);
        theta=exp(hyp(1:end-1));
        if simplify_hyp, theta=theta*ones(1,vari_num);end % extend hyp
        rho=hyp(end);
        Y_bias=Y-rho*Y_pred; % notice Y equal to d

        [R,L,Beta,s2,dLR_F]=calCovKRG(X_dis_sq,Y_bias,x_num,vari_num,theta,H);

        % calculation negative log likelihood
        if s2 == 0,fval=0;
        else,fval=(x_num*log(s2)+2*sum(log(diag(L))))/2;end

        % calculate gradient
        if nargout > 1
            % gradient
            grad=zeros(vari_num+1,1);
            invR=L'\(L\eye(x_num));
            invHiRH=(dLR_F'*dLR_F)\eye(size(H,2));
            Y_Fmiu=Y-H*Beta;

            % theta d
            for vari_i=1:vari_num
                dR_dtheta=-(X_dis_sq(:,:,vari_i).*R)*theta(vari_i)/vari_num;
                dinvR_dtheta=...
                    -invR*dR_dtheta*invR;
                dinvHiRH_dtheta=-invHiRH*...
                    (H'*dinvR_dtheta*H)*...
                    invHiRH;
                dbeta_dtheta=dinvHiRH_dtheta*(H'*invR*Y_bias)+...
                    invHiRH*(H'*dinvR_dtheta*Y_bias);
                dY_Fmiu_dtheta=-H*dbeta_dtheta;
                dsigma2_dtheta=(dY_Fmiu_dtheta'*invR*Y_Fmiu+...
                    Y_Fmiu'*dinvR_dtheta*Y_Fmiu+...
                    Y_Fmiu'*invR*dY_Fmiu_dtheta)/x_num;
                dlnsigma2_dtheta=1/s2*dsigma2_dtheta;
                dlndetR=trace(invR*dR_dtheta);

                grad(vari_i)=x_num/2*dlnsigma2_dtheta+0.5*dlndetR;
            end

            % rho
            dY_drho=-Y_pred*rho;
            dbeta_drho=invHiRH*(H'*invR*dY_drho);
            dY_Fmiu_drho=(dY_drho-H*dbeta_drho);
            dsigma2_drho=(dY_Fmiu_drho'*invR*Y_Fmiu+...
                Y_Fmiu'*invR*dY_Fmiu_drho)/x_num;
            dlnsigma2_drho=1/s2*dsigma2_drho;

            grad(end)=x_num/2*dlnsigma2_drho;

            if simplify_hyp, grad=[sum(grad(1:end-1)),grad(end)];end
        end
    end

    function [cov,L_cov,beta,sigma_sq,dLR_H,dLR_Y,dLR_U]=calCovCoKRG...
            (X_dis_sq,Y,x_num,x_HF_num,x_LF_num,vari_num,...
            theta_bias,rho,theta_LF,sigma2_bias,sigma2_LF,H)
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
        dLR_H=L_cov\H;
        dLR_Y=L_cov\Y; % invHiRH=(H'*inv_cov*H)\eye(size(H,2));

        % basical bias
        beta=dLR_H\dLR_Y; % beta=invHiRH*(H'*inv_cov*Y);
        dLR_U=dLR_Y-dLR_H*beta; % U=Y-H*beta;
        sigma_sq=(dLR_U'*dLR_U)/x_num; % sigma_sq=(U'*inv_cov*U)/x_num;
    end

    function [Y_pred,Var_pred]=predictKRG...
            (X_pred,X_norm,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,theta,reg_fcn,...
            L_cov,beta,sigma_sq,gamma,dLR_H,inv_FTcovF)
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
        X_pred_norm=(X_pred-aver_X)./stdD_X;

        % regression value
        fval_reg_pred_norm=(reg_fcn(X_pred)-aver_Y)./stdD_Y;

        % predict covariance
        cov_pred=zeros(x_num,x_pred_num);
        for vari_i=1:vari_num
            cov_pred=cov_pred+(X_norm(:,vari_i)-X_pred_norm(:,vari_i)').^2*theta(vari_i);
        end
        cov_pred=exp(-cov_pred/vari_num^2);

        % predict base fval
        Y_pred=fval_reg_pred_norm*beta+cov_pred'*gamma;

        % predict variance
        dLR_r=L_cov\cov_pred;
        u=(dLR_H)'*dLR_r-fval_reg_pred_norm';
        Var_pred=sigma_sq*(1+u'*inv_FTcovF*u-dLR_r'*dLR_r);
        Var_pred=diag(Var_pred);

        % renormalize data
        Y_pred=Y_pred*stdD_Y+aver_Y;
    end

    function [Y_pred,Var_pred]=predictCoKRG...
            (X_pred,X_norm,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,x_HF_num,x_LF_num,theta_bias,rho,theta_LF,reg_fcn,...
            L_cov,beta,sigma_sq_bias,sigma_sq_LF,gamma,dLR_H,inv_FTcovF)
        % kriging surrogate predict function
        % input predict_x and kriging model
        % predict_x is row vector
        % output the predict value
        %
        [x_pred_num,~]=size(X_pred);
        fval_reg_pred_norm=[(reg_fcn(X_pred)-aver_Y)./stdD_Y*rho,...
            (reg_fcn(X_pred)-aver_Y)./stdD_Y];

        % normalize data
        X_pred_norm=(X_pred-aver_X)./stdD_X;

        % distance sq of X_predict
        X_pred_dis_sq=zeros(x_num,x_pred_num,vari_num);
        for vari_i=1:vari_num
            X_pred_dis_sq(:,:,vari_i)=(X_norm(:,vari_i)-X_pred_norm(:,vari_i)').^2;
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
        Y_pred=fval_reg_pred_norm*beta+cov_pred'*gamma;

        % predict variance
        dLR_r=L_cov\cov_pred;
        u=(dLR_H)'*dLR_r-fval_reg_pred_norm';
        Var_pred=sigma_sq_LF*rho*rho+sigma_sq_bias+u'*inv_FTcovF*u-dLR_r'*dLR_r;
        Var_pred=diag(Var_pred);

        % normalize data
        Y_pred=Y_pred*stdD_Y+aver_Y;
    end

end
