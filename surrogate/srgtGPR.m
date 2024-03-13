function model_GPR=srgtGPR(X,Y,model_option)
% generate gaussian process regression model
% input data will be normalize by average and standard deviation of data
%
% input:
% X(x_num x vari_num matrix), Y(x_num x 1 matrix)
% model_option(optional, include: optimize_hyp, simplify_hyp, hyp, optimize_option)
%
% output:
% model_GPR(a gauss process regression model)
%
% abbreviation:
% num: number, pred: predict, vari: variable, hyp: hyper parameter
% NLL: negative log likelihood, obv: observe
%
% reference: [1] RASMUSSEN C E, WILLIAMS C K I. Gaussian Processes for
% Machine Learning [M/OL]. 2005
% [https://doi.org/10.7551/mitpress/3206.001.0001].
%
% Copyright 2023.2 Adel
%
if nargin < 3,model_option=struct();end
if ~isfield(model_option,'optimize_hyp'), model_option.('optimize_hyp')=true(1);end
if ~isfield(model_option,'simplify_hyp'), model_option.('simplify_hyp')=false(1);end
if ~isfield(model_option,'hyp'), model_option.('hyp')=[];end
if ~isfield(model_option,'sigma_obv'), model_option.('sigma_obv')=0;end
simplify_hyp=model_option.('simplify_hyp');
if simplify_hyp,FLAG_GRAD=false;else,FLAG_GRAD=true;end
if ~isfield(model_option,'optimize_option')
    model_option.('optimize_option')=optimoptions...
        ('fminunc','Display','none',...
        'OptimalityTolerance',1e-6,...
        'FiniteDifferenceStepSize',1e-5,...,
        'MaxIterations',20,'SpecifyObjectiveGradient',FLAG_GRAD);
end

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
for variable_index=1:vari_num
    X_dis_sq(:,:,variable_index)=...
        (X_nomlz(:,variable_index)-X_nomlz(:,variable_index)').^2;
end

hyp=model_option.('hyp');
% kernal function is sigma_sq*exp(-X_dis_sq/vari_num^2*len)))
if isempty(hyp),hyp=[ones(1,vari_num),log(1/sqrt(2))];end
sigma_obv=model_option.('sigma_obv')/stdD_Y.^2;

% if optimize hyperparameter
if model_option.optimize_hyp
    obj_fcn_hyp=@(hyp) probNLLGPR(X_dis_sq,Y,x_num,vari_num,hyp,sigma_obv);
    
    if simplify_hyp
        hyp=[mean(hyp(1:end-1)),hyp(end)];
        low_bou_hyp=[-4,-10];
        up_bou_hyp=[4,10];
    else
        low_bou_hyp=[-4*ones(1,vari_num),10];
        up_bou_hyp=[4*ones(1,vari_num),10];
    end

    % [fval,gradient]=obj_fcn_hyp(hyp)
    % [~,gradient_differ]=differ(obj_fcn_hyp,hyp)
    % drawFcn(obj_fcn_hyp,low_bou_hyp,up_bou_hyp);

    [hyp,~,~,~]=fminunc(obj_fcn_hyp,hyp,model_option.('optimize_option'));
    hyp=min(hyp,up_bou_hyp);hyp=max(hyp,low_bou_hyp);

    if simplify_hyp, hyp=[hyp(1)*ones(1,vari_num),hyp(end)];end
end

% obtian covariance matrix
[cov,L_cov,~]=calCovGPR...
    (X_dis_sq,x_num,vari_num,exp(hyp(1:end-1)),exp(hyp(end)),sigma_obv);
sigma_sq=exp(hyp(end))*stdD_Y^2; % renormalize data
sigma_obv=sigma_obv*stdD_Y^2;
gamma=L_cov'\(L_cov\Y_nomlz);

% initialization predict function
pred_fcn=@(X_pred) predictGPR...
    (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_num,vari_num,exp(hyp(1:end-1)),sigma_sq,sigma_obv,L_cov,gamma);

model_GPR.X=X;
model_GPR.Y=Y;
model_GPR.aver_X=aver_X;
model_GPR.stdD_X=stdD_X;
model_GPR.aver_Y=aver_Y;
model_GPR.stdD_Y=stdD_Y;

model_GPR.hyp=hyp;
model_GPR.cov=cov;
model_GPR.L_cov=L_cov;

model_GPR.gamma=gamma;


model_GPR.predict=pred_fcn;

    function [fval,gradient]=probNLLGPR...
            (X_dis_sq,Y,x_num,vari_num,hyp,sigma_obv)
        % hyperparameter is [len,sigma_sq]
        % notice to normalize X_dis for difference variable number
        % X_dis_sq will be divide by vari_num
        %
        hyp=min(hyp,up_bou_hyp);hyp=max(hyp,low_bou_hyp);
        len=exp(hyp(1:end-1));
        sigma2=exp(hyp(end));
        if simplify_hyp, len=len*ones(1,vari_num);end % extend hyp

        % obtian covariance matrix
        [R,L]=calCovGPR(X_dis_sq,x_num,vari_num,len,sigma2,sigma_obv);
        inv_L_Y=L\Y;

        % calculation negative log likelihood
        fval=0.5*(inv_L_Y'*inv_L_Y)+sum(log(diag(L)))+x_num/2*log(2*pi);

        if nargout > 1
            % get gradient
            % var: len_1, len_2, ..., len_n, sigma_sq
            gradient=zeros(1,vari_num+1);
            inv_R=L'\(L\eye(x_num));

            % len
            for vari_idx=1:vari_num
                dcov_dlen=-R.*X_dis_sq(:,:,vari_idx)*len(vari_idx)/vari_num^2;
                dinv_cov_dlen=-inv_R*dcov_dlen*inv_R;
                gradient(vari_idx)=0.5*Y'*dinv_cov_dlen*Y+...
                    0.5*trace(inv_R*dcov_dlen);
            end

            % sigma_sq
            dcov_dsigma_sq=R;
            dinv_cov_dsigma_sq=-inv_R*dcov_dsigma_sq*inv_R;
            gradient(end)=0.5*Y'*dinv_cov_dsigma_sq*Y+...
                0.5*trace(inv_R*dcov_dsigma_sq);

            if simplify_hyp, gradient=[sum(gradient(1:end-1)),gradient(end)];end
        end
    end

    function [cov,L,exp_dis]=calCovGPR...
            (X_dis_sq,x_num,vari_num,len,sigma_sq,sigma_obv)
        % obtain covariance of x
        % notice to normalize X_dis for difference variable number
        % X_dis_sq will be divide by vari_num
        % cov=K+sigma_obv*I
        %

        % calculate covariance
        exp_dis=zeros(x_num,x_num);
        for vari_index=1:vari_num
            exp_dis=exp_dis+X_dis_sq(:,:,vari_index)*len(vari_index);
        end
        exp_dis=exp(-exp_dis/vari_num^2)+eye(x_num)*((1000+x_num)*eps);

        cov=sigma_sq*exp_dis+sigma_obv*eye(x_num);
        L=chol(cov)';
    end

    function [Y_pred,Var_pred]=predictGPR...
            (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,len,sigma_sq,sigma_obv,L_cov,gamma)
        % gaussian process regression predict function
        % output the predict value and predict variance
        %
        [x_pred_num,~]=size(X_pred);
        
        % normalize data
        X_pred_nomlz=(X_pred-aver_X)./stdD_X;
        
        % predict covariance
        cov_pred=zeros(x_num,x_pred_num);
        for vari_index=1:vari_num
            cov_pred=cov_pred+...
                (X_nomlz(:,vari_index)-X_pred_nomlz(:,vari_index)').^2*len(vari_index);
        end
        cov_pred=sigma_sq/stdD_Y.^2*exp(-cov_pred/vari_num^2);

        % get miu and variance of predict x
        inv_L_r=L_cov\cov_pred;
        Y_pred=cov_pred'*gamma;
        Var_pred=(sigma_sq+sigma_obv)*eye(x_pred_num)-inv_L_r'*inv_L_r;

        % normalize data
        Y_pred=Y_pred*stdD_Y+aver_Y;
        Var_pred=diag(Var_pred)*stdD_Y*stdD_Y;
    end
end
