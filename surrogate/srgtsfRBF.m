function srgt=srgtsfRBF(X,Y,option)
% generate radial basis function surrogate model
% input data will be normalize by average and standard deviation of data
%
% input:
% X (matrix): trained X, x_num x vari_num
% Y (vector): trained Y, x_num x 1
% option (struct): optional input, construct option
%
% option include:
% optimize_hyp (boolean): whether optimize hyperparameter
% optimize_option (optimoptions): fminbnd optimize option
% basis_fcn (function handle): basis function, default is r.^3
% hyp (double): hyperparameter value of basis function
%
% output:
% srgt(struct): a radial basis function model
%
% Copyright 2023.2 Adel
%
if nargin < 3,option=struct();end

% RBF option
if ~isfield(option,'optimize_hyp'), option.('optimize_hyp')=true;end
if ~isfield(option,'optimize_option')
    option.('optimize_option')=optimset('Display','none','TolFun',1e-6);
end

if ~isfield(option,'basis_fcn'), option.('basis_fcn')=[];end
if ~isfield(option,'hyp'), option.('hyp')=[];end

% initialize data
G=[];L=[];U=[];P=[];
RUPD_flag=[];
dLRY=[];
dURY=[];
weight=[];

% normalize data
[x_num,vari_num]=size(X);
aver_X=mean(X);
stdD_X=std(X);stdD_X(stdD_X == 0)=1;
aver_Y=mean(Y);
stdD_Y=std(Y);stdD_Y(stdD_Y == 0)=1;
X_norm=(X-aver_X)./stdD_X;
Y_norm=(Y-aver_Y)./stdD_Y;

% basis function define
basis_fcn=option.('basis_fcn');
if isempty(basis_fcn)
    % initialize r
    r_sq=zeros(x_num,x_num);
    for vari_idx=1:vari_num
        r_sq=r_sq+(X_norm(:,vari_idx)-X_norm(:,vari_idx)').^2;
    end
    r=sqrt(r_sq);

    % basis_fcn=@(X,X_pred,hyp)basisGauss(X,X_pred,hyp,r_sq);
    basis_fcn=@(X,X_pred)basisCubic(X,X_pred,r);
end
if nargin(basis_fcn) == 2,option.optimize_hyp=false;end

% hyperparameter define and optimize
hyp=option.('hyp');
if isempty(hyp), hyp=0;end
if option.optimize_hyp % optimize hyperparameter
    obj_fcn_hyp=@(hyp) probNLL(X,Y,basis_fcn,hyp);
    low_bou_hyp=-4;
    up_bou_hyp=4;

    % drawFcn([],obj_fcn_hyp,low_bou_hyp,up_bou_hyp);

    [hyp,~,~,~]=fminbnd(obj_fcn_hyp,low_bou_hyp,up_bou_hyp,option.('optimize_option'));
end

% calculate RBF
[logdetG,G]=calRBF(X,Y,basis_fcn,hyp);
sigma_sq=abs(sigma_sq)*stdD_Y^2; % renormalize data

% initialization predict function
pred_fcn=@(X_pred)predictRBF(X,X_pred,basis_fcn,hyp);

srgt=option;
srgt.X=X;
srgt.Y=Y;
% srgt.basis_fcn=basis_fcn;
srgt.gram=G;
srgt.weight=weight;
srgt.sigma_sq=sigma_sq;
srgt.predict=pred_fcn;
srgt.loo_err=calLOOErr();

%% basical function

    function fval=probNLL(X,Y,basis_fcn,hyp)
        % negative logarithmic likelihood probability function
        %
        [logdetG,G]=calRBF(X,Y,basis_fcn,hyp);

        % % calculation negative log likelihood
        % if sigma_sq == 0
        %     fval=0;
        %     return;
        % end
        % fval=(x_num*log(abs(sigma_sq))*sign(sigma_sq)+logdetG)/2;

        % calculation negative R2
        loo_err=calLOOErr();
        R2=1-sum(loo_err.^2)/stdD_Y^2;
        fval=-R2;

        % % calculate gradient
        % if nargout > 1
        %     dinvG_dhyp=-invG*dG_dhyp*invG;
        %     dloo_err_dhyp=zeros(x_number,1);
        %     I=eye(x_number);
        %     for x_idx=1:x_num
        %         dloo_err_dhyp(x_idx)=(I(x_idx,:)*dinvG_dhyp*Y_norm)/...
        %             invG(x_idx,x_idx)-...
        %             weight(x_idx)*(I(x_idx,:)*dinvG_dhyp*I(:,x_idx))/...
        %             invG(x_idx,x_idx)^2;
        %     end
        % 
        %     grad=2*sum(loo_err.*dloo_err_dhyp);
        % end
    end

    function [logdetG,G]=calRBF(X,~,basis_fcn,hyp)
        % RBF train function, calculate weight, sigma
        % y(x)=z(x)
        %
        % input:
        % X (matrix): trained X, x_num x vari_num
        % Y (vector): trained Y, x_num x 1
        % basis_fcn (function handle): basis function
        % hyp (array): hyperparameter value of basis function
        %

        % calculate gram matrix
        if nargin(basis_fcn) == 2,G=basis_fcn(X,[]);
        else,G=basis_fcn(X,[],hyp);end

        % calculate parameter
        [L,RUPD_flag]=chol(G);
        if RUPD_flag
            [L,U,P]=lu(G);
            du=diag(U);
            s=detMat1(P)*prod(sign(du));
            logdetG=s*sum(log(abs(du)));
            % invG=U\(L\(P'\eye(x_num)));

            dLRY=L\(P*Y_norm);
            dURY=U'\Y_norm;
        else
            U=L;
            logdetG=2*sum(log(diag(L)));
            % invG=L\(L'\eye(x_num));

            dLRY=U'\Y_norm;
            dURY=dLRY;
        end

        weight=U\dLRY;
        sigma_sq=(dURY'*dLRY)/x_num;

        function d=detMat1(M)
            n=size(M,1);d=0;
            idx=find(M);[i,~]=ind2sub([n,n],idx);
            for k=1:n,d=d+sum(i(1:k) > i(k));end
            d=1-2*mod(d,2);
        end
    end

    function [Y_pred,Var_pred]=predictRBF(X,X_pred,basis_fcn,hyp)
        % RBF predict function
        %
        % input:
        % X (matrix): trained X, x_num x vari_num
        % X_pred (matrix): predict X, x_pred_num x vari_num
        % basis_fcn (function handle): basis function
        % hyp (array): hyperparameter value of basis function
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

            if nargin(basis_fcn) == 2,G_pred=basis_fcn(X,X_pred(idx,:));G_self=basis_fcn([],[]); % predict covariance
            else,G_pred=basis_fcn(X,X_pred(idx,:),hyp);G_self=basis_fcn([],[],hyp);end

            % predict value
            Y_pred(idx,:)=(G_pred'*weight)*stdD_Y+aver_Y;

            if nargout > 1
                if RUPD_flag
                    dLRr=L\(P*G_pred);
                    dURr=U'\G_pred;
                else
                    dLRr=U'\G_pred;
                    dURr=dLRr;
                end

                % if RUPD_flag,invG=U\(L\P);
                % else,invG=L\(L'\eye(x_num));end
                % Var_pred(idx,:)=sigma_sq*diag(G_self-G_pred'*invG*G_pred);

                Var_pred(idx,:)=sigma_sq*diag(max(G_self-dURr'*dLRr,0));
            end

            x_pred_idx=x_pred_idx+batch;
        end
    end

    function loo_err=calLOOErr()
        % analysis method to quickly calculate LOO of RBF surrogate model
        %
        % reference:
        % [1] Rippa S. An Algorithm for Selecting a Good Value for the
        % Parameter c in Radial Basis Function Interpolation [J]. Advances
        % in Computational Mathematics, 1999, 11(2-3): 193-210.
        %
        if RUPD_flag,invG=U\(L\P);
        else,invG=L\(L'\eye(x_num));end
        loo_err=weight*stdD_Y./diag(invG);
    end

%% radial basis function

    function gram=basisGauss(X,X_pred,hyp,r_sq)
        % gauss basis
        %
        scale=size(X_norm,2)^2;
        theta=exp(hyp);
        if isempty(X_pred) % self covariance
            if isempty(X) % diag covariance
                gram=1;
            else
                gram=exp(-r_sq*theta/scale)+eye(size(X_norm,1))*((1000+size(X_norm,1))*eps);
            end
        else % predict covariance
            X_pred_norm=(X_pred-aver_X)./stdD_X;
            gram=zeros(size(X_norm,1),size(X_pred_norm,1));
            for vari_i=1:size(X_norm,2)
                gram=gram+(X_norm(:,vari_i)-X_pred_norm(:,vari_i)').^2;
            end
            gram=exp(-gram*theta/scale);
        end
    end

    function gram=basisCubic(X,X_pred,r)
        % cubic basis
        %
        if isempty(X_pred) % self covariance
            if isempty(X) % diag covariance
                gram=0;
            else
                gram=r.^3+eye(x_num)*((1000+x_num)*eps);
            end
        else % predice covariance
            X_pred_norm=(X_pred-aver_X)./stdD_X;
            gram=(dist(X_pred_norm,X_norm')').^3;
        end
    end
end
