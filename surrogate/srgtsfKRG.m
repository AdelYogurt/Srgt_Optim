function srgt=srgtsfKRG(X,Y,option)
% generate Kriging surrogate model
% input data will be normalize by average and standard deviation of data
%
% input:
% X (matrix): trained X, x_num x vari_num
% Y (vector): trained Y, x_num x 1
% option (struct): optional input, construct option
%
% option include:
% optimize_hyp (boolean): whether optimize hyperparameter
% simplify_hyp (boolean): whether simplify multi hyperparameter to one
% optimize_option (optimoptions): fminunc optimize option
% reg_fcn (function handle): basis function, default is linear
% cov_fcn (function handle): kernel function, default is gauss
% hyp (array): hyperparameter value of kernel function
%
% output:
% srgt (struct): a Kriging model
%
% Copyright 2023.2 Adel
%
if nargin < 3,option=struct();end

% Kriging option
if ~isfield(option,'optimize_hyp'), option.('optimize_hyp')=true;end
if ~isfield(option,'simplify_hyp'), option.('simplify_hyp')=true;end
if option.('simplify_hyp'),FLAG_GRAD=false;else,FLAG_GRAD=true;end
if ~isfield(option,'optimize_option')
    option.('optimize_option')=optimoptions...
        ('fminunc','Display','none',...
        'OptimalityTolerance',1e-6,...
        'FiniteDifferenceStepSize',1e-5,...
        'MaxIterations',20,'SpecifyObjectiveGradient',FLAG_GRAD);
end

if ~isfield(option,'reg_fcn'), option.('reg_fcn')=[];end
if ~isfield(option,'cov_fcn'), option.('cov_fcn')=[];end
if ~isfield(option,'hyp'), option.('hyp')=[];end

% initialize data
R=[];L=[];U=[];P=[];
RUPD_flag=[];
dLRH=[];
dURH=[];
dLRY=[];
dURY=[];
dLRU=[];
dURU=[];
H_norm=[];
beta=[];
gamma=[];

% normalize data
[x_num,vari_num]=size(X);
aver_X=mean(X);
stdD_X=std(X);stdD_X(stdD_X == 0)=1;
X_norm=(X-aver_X)./stdD_X;
aver_Y=mean(Y);
stdD_Y=std(Y);stdD_Y(stdD_Y == 0)=1;
Y_norm=(Y-aver_Y)./stdD_Y;

% regression function define
reg_fcn=option.('reg_fcn');
if isempty(reg_fcn)
    if x_num < vari_num,reg_fcn=@(X) ones(size(X,1),1).*stdD_Y+aver_Y; % constant
    else,reg_fcn=@(X) [ones(size(X,1),1),X-aver_X].*stdD_Y+aver_Y;end % linear
end

% covarianve function define
cov_fcn=option.('cov_fcn');
if isempty(cov_fcn)
    % initialize dr_sq
    dr=zeros(x_num,x_num,vari_num);
    for vari_idx=1:vari_num
        dr(:,:,vari_idx)=abs(X_norm(:,vari_idx)-X_norm(:,vari_idx)');
    end
    dr_sq=dr.^2;
    % r=sqrt(sum(dr_sq,3));

    % cov_fcn=@(X,X_pred,hyp)covCuSpin(X,X_pred,hyp,dr);
    cov_fcn=@(X,X_pred,hyp)covGauss(X,X_pred,hyp,dr_sq);
    % cov_fcn=@(X,X_pred)covCubic(X,X_pred,r);
end
if nargin(cov_fcn) == 2,option.optimize_hyp=false;end

% hyperparameter define and optimize
hyp=option.('hyp');
if isempty(hyp), hyp=zeros(1,vari_num);end
if option.optimize_hyp % optimize hyperparameter
    simplify_hyp=option.('simplify_hyp');
    obj_fcn_hyp=@(hyp) probNLL(X,Y,reg_fcn,cov_fcn,hyp);

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
    % drawFcn([],obj_fcn_hyp,low_bou_hyp,up_bou_hyp);

    [hyp,~,~,~]=fminunc(obj_fcn_hyp,hyp,option.('optimize_option'));
    hyp=min(hyp,up_bou_hyp);hyp=max(hyp,low_bou_hyp);

    if simplify_hyp, hyp=hyp*ones(1,vari_num);end
end

% calculate KRG
[logdetR,R]=calKRG(X,Y,reg_fcn,cov_fcn,hyp);
sigma_sq=abs(sigma_sq)*stdD_Y^2; % renormalize data
invHiRH=(dURH'*dLRH)\eye(size(H_norm,2));

% initialization predict function
pred_fcn=@(X_pred)predictKRG(X,X_pred,reg_fcn,cov_fcn,hyp);

srgt=option;
srgt.X=X;
srgt.Y=Y;
% srgt.reg_fcn=reg_fcn;
% srgt.cov_fcn=cov_fcn;
srgt.hyp=hyp;
srgt.cov=R;
srgt.beta=beta;
srgt.gamma=gamma;
srgt.sigma_sq=sigma_sq;
srgt.predict=pred_fcn;

%% basical function

    function [fval,grad]=probNLL(X,Y,reg_fcn,cov_fcn,hyp)
        % negative logarithmic likelihood probability function
        %
        hyp=min(hyp,up_bou_hyp);hyp=max(hyp,low_bou_hyp); % prevent excessive hyp
        if simplify_hyp, hyp=hyp*ones(1,vari_num);end % extend hyp

        if nargout > 1
            % require gradient
            [logdetR,R,dR_dhyp]=calKRG(X,Y,reg_fcn,cov_fcn,hyp);
        else
            [logdetR,R]=calKRG(X,Y,reg_fcn,cov_fcn,hyp);
        end

        % calculation negative log likelihood
        if sigma_sq == 0
            fval=0;grad=zeros(vari_num,1);
            if simplify_hyp, grad=0;end
            return;
        end
        fval=(x_num*log(abs(sigma_sq))*sign(sigma_sq)+logdetR)/2;

        if nargout > 1
            % calculate gradient
            grad=zeros(1,vari_num);

            if RUPD_flag,invR=U\(L\P);
            else,invR=L\(L'\eye(x_num));end
            invHiRH=(dURH'*dLRH)\eye(size(H_norm,2));
            U_norm=Y_norm-H_norm*beta;
            for vari_i=1:vari_num
                dinvR_dtheta=...
                    -invR*dR_dhyp(:,:,vari_i)*invR;
                dinv_FTRF_dtheta=-invHiRH*...
                    (H_norm'*dinvR_dtheta*H_norm)*...
                    invHiRH;
                dmiu_dtheta=dinv_FTRF_dtheta*(H_norm'*invR*Y_norm)+...
                    invHiRH*(H_norm'*dinvR_dtheta*Y_norm);
                dU_dtheta=-H_norm*dmiu_dtheta;
                dsigma_sq_dtheta=(dU_dtheta'*invR*U_norm+...
                    U_norm'*dinvR_dtheta*U_norm+...
                    U_norm'*invR*dU_dtheta)/x_num;
                dlnsigma2_dtheta=1/sigma_sq*dsigma_sq_dtheta;
                dlndetR=trace(invR*dR_dhyp(:,:,vari_i));

                grad(vari_i)=x_num/2*dlnsigma2_dtheta+0.5*dlndetR;
            end

            if simplify_hyp, grad=sum(grad);end
        end
    end

    function [logdetR,R,dR_dhyp]=calKRG(X,~,reg_fcn,cov_fcn,hyp)
        % KRG train function, calculate beta, gamma, sigma
        % y(x)=f(x)+z(x)
        %
        % input:
        % X (matrix): trained X, x_num x vari_num
        % Y (vector): trained Y, x_num x 1
        % reg_fcn (function handle): basis function
        % cov_fcn (function handle): kernel function
        % hyp (array): hyperparameter value of kernel function
        %

        % calculate regression
        H_norm=(reg_fcn(X)-aver_Y)./stdD_Y;

        % calculate correlation matrix
        if nargin(cov_fcn) == 2
            R=cov_fcn(X,[]);
        else
            if nargout < 3,R=cov_fcn(X,[],hyp);
            else,[R,dR_dhyp]=cov_fcn(X,[],hyp);end
        end

        % calculate parameter
        
        [L,RUPD_flag]=chol(R);
        if RUPD_flag
            [L,U,P]=lu(R);
            du=diag(U);
            s=detMat1(P)*prod(sign(du));
            logdetR=s*sum(log(abs(du)));
            % invR=U\(L\P);

            % invHiRH=(H_norm'*invR*H_norm)\eye(size(H_norm,2));
            dLRH=L\(P*H_norm);
            dURH=U'\H_norm;

            % beta=invHiRH*(H_norm'*invR*Y_norm);
            dLRY=L\(P*Y_norm);
            dURY=U'\Y_norm;
            beta=dLRH\dLRY;

            dLRU=dLRY-dLRH*beta;
            dURU=dURY-dURH*beta;
        else
            U=L;
            logdetR=2*sum(log(diag(U)));
            % invR=L\(L'\eye(x_num));

            % invHiRH=(H_norm'*invR*H_norm)\eye(size(H_norm,2));
            dLRH=U'\H_norm;
            dURH=dLRH;

            % beta=invHiRH*(H_norm'*invR*Y_norm);
            dLRY=U'\Y_norm;
            dURY=dLRY;
            beta=dLRH\dLRY;

            dLRU=dLRY-dLRH*beta;
            dURU=dLRU;
        end

        gamma=U\dLRU; % gamma=invR*U_norm;
        sigma_sq=(dURU'*dLRU)/x_num; % sigma_sq=(U_norm'*gamma)/x_num;

        function d=detMat1(M)
            n=size(M,1);d=0;
            idx=find(M);[i,~]=ind2sub([n,n],idx);
            for k=1:n,d=d+sum(i(1:k) > i(k));end
            d=1-2*mod(d,2);
        end
    end

    function [Y_pred,Var_pred]=predictKRG(X,X_pred,reg_fcn,cov_fcn,hyp)
        % KRG predict function
        %
        % input:
        % X (matrix): trained X, x_num x vari_num
        % X_pred (matrix): predict X, x_pred_num x vari_num
        % reg_fcn (function handle): basis function
        % cov_fcn (function handle): kernel function
        % hyp (array): hyperparameter value of kernel function
        %
        % output:
        % Y_pred (vector): predict Y, x_pred_num x 1
        % Var_pred (vector): predict variance, x_pred_num x 1
        %
        max_batch=1024; % max size of each batch

        % allocate memory
        x_pred_num=size(X_pred,1);
        Y_pred=zeros(x_pred_num,1);
        Var_pred=zeros(x_pred_num,1);
        x_pred_idx=0;

        while x_pred_idx < x_pred_num
            batch=min(max_batch,x_pred_num-x_pred_idx); % limit each batch
            idx=(x_pred_idx+1):(x_pred_idx+batch);

            H_pred_norm=(reg_fcn(X_pred(idx,:))-aver_Y)./stdD_Y; % regression value
            if nargin(cov_fcn) == 2,R_pred=cov_fcn(X,X_pred(idx,:));R_self=cov_fcn([],[]); % predict covariance
            else,R_pred=cov_fcn(X,X_pred(idx,:),hyp);R_self=cov_fcn([],[],hyp);end

            % predict value
            Y_pred(idx,:)=(H_pred_norm*beta+R_pred'*gamma)*stdD_Y+aver_Y;

            % predict variance
            if nargout > 1
                if RUPD_flag
                    dLRr=L\(P*R_pred);
                    dURr=U'\R_pred;
                else
                    dLRr=U'\R_pred;
                    dURr=dLRr;
                end
                u=(dURH)'*dLRr-H_pred_norm';
                Var_pred(idx,:)=sigma_sq*diag(max((R_self+u'*invHiRH*u-dURr'*dLRr),0));
            end

            x_pred_idx=x_pred_idx+batch;
        end
    end

%% covariance function

    function [cov,dcov_dhyp]=covCuSpin(~,X_pred,hyp,dr)
        % cubic spline covariance
        %
        theta=exp(hyp);
        scale=size(X_norm,2);
        if isempty(X_pred) % self covariance
            if isempty(X) % diag covariance
                cov=1;
            else
                xi=reshape(theta,[1,1,size(X_norm,2)]).*dr/scale;
                cov_vari=zeros(size(xi));
                bool_inr=xi <= 0.2;cov_vari(bool_inr)=1-15*xi(bool_inr).^2+30*xi(bool_inr).^3;
                bool_mid=0.2 < xi & xi < 1;cov_vari(bool_mid)=1.25*(1-xi(bool_mid)).^3;
                cov=prod(cov_vari,3)+eye(x_num)*((1000+x_num)*eps);

                if nargout > 1
                    dcov_vari=zeros(size(xi));
                    dcov_vari(bool_inr)=-30*xi(bool_inr)+90*xi(bool_inr).^2;
                    dcov_vari(bool_mid)=-3.75*(1-xi(bool_mid)).^2;

                    dcov_dhyp=zeros(size(X_norm,1),size(X_norm,1),size(X_norm,2));
                    for vari_i=1:size(X_norm,2)
                        dcov_mat=cov_vari;dcov_mat(:,:,vari_i)=dcov_vari(:,:,vari_i);
                        dcov_dhyp(:,:,vari_i)=prod(dcov_mat,3).*xi(:,:,vari_i);
                    end
                end
            end
        else % predict covariance
            X_pred_norm=(X_pred-aver_X)./stdD_X;
            cov=ones(size(X_norm,1),size(X_pred_norm,1));
            for vari_i=1:size(X_norm,2)
                xi_page=abs(X_norm(:,vari_i)-X_pred_norm(:,vari_i)')*theta(vari_i)/scale;
                cov_page=zeros(size(xi_page));
                bool_inr=xi_page <= 0.2;cov_page(bool_inr)=1-15*xi_page(bool_inr).^2+30*xi_page(bool_inr).^3;
                bool_mid=0.2 < xi_page & xi_page < 1;cov_page(bool_mid)=1.25*(1-xi_page(bool_mid)).^3;
                cov=cov.*cov_page;
            end
        end
    end

    function [cov,dcov_dhyp]=covGauss(X,X_pred,hyp,dr_sq)
        % gaussian covariance
        %
        scale=size(X_norm,2)^2;
        theta=exp(hyp);
        if isempty(X_pred) % self covariance
            if isempty(X) % diag covariance
                cov=1;
            else
                cov=zeros(size(X_norm,1));
                for vari_i=1:size(X_norm,2)
                    cov=cov+dr_sq(:,:,vari_i)*theta(vari_i);
                end
                cov=exp(-cov/scale)+eye(x_num)*((1000+x_num)*eps);

                if nargout > 1
                    dcov_dhyp=zeros(size(X_norm,1),size(X_norm,1),size(X_norm,2));
                    for vari_i=1:size(X_norm,2)
                        dcov_dhyp(:,:,vari_i)=-(dr_sq(:,:,vari_i).*cov)*theta(vari_i)/scale;
                    end
                end
            end
        else % predict covariance
            X_pred_norm=(X_pred-aver_X)./stdD_X;
            cov=zeros(size(X_norm,1),size(X_pred_norm,1));
            for vari_i=1:size(X_norm,2)
                cov=cov+(X_norm(:,vari_i)-X_pred_norm(:,vari_i)').^2*theta(vari_i);
            end
            cov=exp(-cov/scale);
        end
    end

    function cov=covCubic(X,X_pred,r)
        % cubic covariance
        %
        if isempty(X_pred) % self covariance
            if isempty(X) % diag covariance
                cov=0;
            else
                cov=r.^3+eye(x_num)*((1000+x_num)*eps);
            end
        else % predice covariance
            X_pred_norm=(X_pred-aver_X)./stdD_X;
            cov=(dist(X_pred_norm,X_norm')').^3;
        end
    end
end
