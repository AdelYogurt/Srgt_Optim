function model_GPC=classifyGPC(X,Y,model_option)
% generate gaussian process classifier model
% assembly of gpml-3.6 with EP infer method
% only support binary classification, 0 and 1
%
% input:
% X(x_num x vari_num matrix), Y(x_num x 1 matrix),...
% model_option(optional, include: optimize_hyp, simplify_hyp, hyp,...
% optimize_option)
%
% output:
% model_GPC(a gaussian process classifier model)
%
% abbreviation:
% num: number, pred: predict, vari: variable, hyp: hyper parameter
% NLL: negative log likelihood, nomlz: normalization, var: variance
%
if nargin < 3,model_option=struct();end
if ~isfield(model_option,'optimize_hyp'), model_option.('optimize_hyp')=true(1);end
if ~isfield(model_option,'simplify_hyp'), model_option.('simplify_hyp')=false(1);end
if ~isfield(model_option,'hyp'), model_option.('hyp')=[];end
if ~isfield(model_option,'optimize_option')
    model_option.('optimize_option')=optimoptions...
        ('fminunc','Display','none',...
        'OptimalityTolerance',1e-6,...
        'FiniteDifferenceStepSize',1e-5,...,
        'MaxIterations',20,'SpecifyObjectiveGradient',true);
end

% normalization data
[x_num,vari_num]=size(X);
aver_X=mean(X);
stdD_X=std(X);stdD_X(stdD_X == 0)=1;
X_nomlz=(X-aver_X)./stdD_X;
Y(Y==0)=-1;

% initial X_dis_sq
X_dis_sq=zeros(x_num,x_num,vari_num);
for variable_index=1:vari_num
    X_dis_sq(:,:,variable_index)=...
        (X_nomlz(:,variable_index)-X_nomlz(:,variable_index)').^2;
end

% regression function define
% notice reg_fcn process no normalization data
reg_fcn=@(X) regMinus(X);
% reg_fcn=@(X) regZero(X);
% reg_fcn=@(X) regLinear(X);

% calculate reg
fval_reg=reg_fcn(X);

hyp=model_option.('hyp');
% kernal function is sigma_sq*exp(-X_dis_sq/vari_num*len))
if isempty(hyp),hyp=[log(x_num)*ones(1,vari_num),log(0.5)];end
simplify_hyp=model_option.('simplify_hyp');

% if optimize hyperparameter
if model_option.optimize_hyp
    obj_fcn_hyp=@(hyp) probNLLGPC(X_dis_sq,Y,x_num,vari_num,hyp,fval_reg);

    if simplify_hyp
        hyp=[mean(hyp(1:end-1)),hyp(end)];
        low_bou_hyp=-3*ones(1,2);
        up_bou_hyp=3*ones(1,2);
    else
        low_bou_hyp=-3*ones(1,vari_num+1);
        up_bou_hyp=3*ones(1,vari_num+1);
    end

    % [fval,gradient]=obj_fcn_hyp(hyp)
    % [~,gradient_differ]=differ(obj_fcn_hyp,hyp)

    [hyp,~,~,~]=fminunc(obj_fcn_hyp,hyp,model_option.('optimize_option'));
    hyp=min(hyp,up_bou_hyp);hyp=max(hyp,low_bou_hyp);

    if simplify_hyp, hyp=[hyp(1)*ones(1,vari_num),hyp(end)];end
end

% initialization predict function
covariance=calCovGPC(X_dis_sq,x_num,vari_num,exp(hyp(1:end-1)),exp(hyp(end)));
[~,alpha,sW,L]=calInfGPC(covariance,Y,x_num,fval_reg);
pred_fcn=@(X_pred) predictGPC(X_pred,X_nomlz,aver_X,stdD_X,...
    x_num,vari_num,exp(hyp(1:end-1)),exp(hyp(end)),alpha,sW,L,reg_fcn);

model_GPC.X=X;
model_GPC.Y=Y;
model_GPC.aver_X=aver_X;
model_GPC.stdD_X=stdD_X;

model_GPC.hyp=hyp;

model_GPC.covariance=covariance;
model_GPC.alpha=alpha;
model_GPC.sW=sW;
model_GPC.L=L;

model_GPC.predict=pred_fcn;

    function [fval,gradient]=probNLLGPC(X_dis_sq,Y,x_num,vari_num,hyp,fval_reg)
        % hyperparameter is [len,eta]
        % X_dis_sq will be divide by vari_num
        %
        hyp=min(hyp,up_bou_hyp);hyp=max(hyp,low_bou_hyp);
        len=exp(hyp(1:end-1));
        eta=exp(hyp(end));
        if simplify_hyp, len=len*ones(1,vari_num);end % extend hyp

        cov=calCovGPC(X_dis_sq,x_num,vari_num,len,eta);
        [fval,alpha,sW,L]=calInfGPC(cov,Y,x_num,fval_reg);

        if nargout > 1
            gradient=zeros(1,vari_num+1);                                   % allocate space for derivatives
            F=alpha*alpha'-repmat(sW,1,x_num).*(L\(L'\diag(sW)));   % covariance hypers

            for len_idx=1:vari_num
                dcov_dhyp=-cov.*X_dis_sq(:,:,len_idx)*len(len_idx)/vari_num;
                gradient(len_idx)=-sum(sum(F.*dcov_dhyp))/2;
            end
            dcov_dhyp=cov;
            gradient(end)=-sum(sum(F.*dcov_dhyp))/2;

            if simplify_hyp, gradient=[sum(gradient(1:end-1)),gradient(end)];end
        end
    end

    function [Y_pred,Prob_pred,Miu_pred,Var_pred]=predictGPC...
            (X_pred,X_nomlz,aver_X,stdD_X,x_num,vari_num,...
            len,eta,alpha,sW,L,reg_fcn)
        % GPC predict function
        %
        [x_pred_num,~]=size(X_pred);
        X_pred_nomlz=(X_pred-aver_X)./stdD_X;
        Y_targ_pred=ones(x_pred_num,1); % predict target class

        % predict covariance
        cov_self=eta;              % self-variance
        cov_pred=zeros(x_num,x_pred_num);
        for vari_idx=1:vari_num
            cov_pred=cov_pred+(X_nomlz(:,vari_idx)-X_pred_nomlz(:,vari_idx)').^2*len(vari_idx);
        end
        cov_pred=eta*exp(-cov_pred/vari_num);

        % predict liklihoood
        fval_reg_pred=reg_fcn(X_pred_nomlz);
        Miu_pred=fval_reg_pred+cov_pred'*alpha;        % conditional mean fs|f
        V =L'\(repmat(sW,1,x_pred_num).*cov_pred);
        Var_pred=cov_self-sum(V.*V,1)';                       % predictive variances
        Var_pred=max(Var_pred,0);   % remove numerical noise i.e. negative variances

        [Prob_pred,ymu,ys2]=calLikGPC(Y_targ_pred,Miu_pred,Var_pred);
        Prob_pred=exp(Prob_pred);
        Y_pred=ones(x_pred_num,1);
        Y_pred(Prob_pred < 0.5)=0;
    end

    function [cov,exp_dis]=calCovGPC(X_dis_sq,x_num,vari_num,len,eta)
        % obtain covariance of x
        % cov: eta,len(equal to 1/len.^2)
        %
        % k=eta*exp(-sum(x_dis*len)/vari_num);
        %

        % exp of x__x with theta
        exp_dis=zeros(x_num);
        for len_idx=1:vari_num
            exp_dis=exp_dis+X_dis_sq(:,:,len_idx)*len(len_idx);
        end
        exp_dis=exp(-exp_dis/vari_num)+eye(x_num)*10000*eps;
        cov=exp_dis*eta;
    end

    function [nlZ,alpha,sW,L]=calInfGPC(cov,Y,x_num,fval_reg)
        % Expectation Propagation approximation to the posterior Gaussian Process.
        % The function takes a specified covariance function (see covFunctions.m) and
        % likelihood function (see likFunctions.m),and is designed to be used with
        % gp.m. See also infMethods.m. In the EP algorithm,the sites are
        % updated in random order,for better performance when cases are ordered
        % according to the targets.
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch 2013-09-13.
        %
        % See also INFMETHODS.M.
        %
        persistent last_ttau last_tnu              % keep tilde parameters between calls
        tol=1e-4; max_sweep=10; min_sweep=2;     % tolerance to stop EP iterations

        inf='infEP';
        % A note on naming: variables are given short but descriptive names in
        % accordance with Rasmussen & Williams "GPs for Machine Learning" (2006): mu
        % and s2 are mean and variance,nu and tau are natural parameters. A leading t
        % means tilde,a subscript _ni means "not i" (for cavity parameters),or _n
        % for a vector of cavity parameters. N(f|mu,Sigma) is the posterior.

        % marginal likelihood for ttau=tnu=zeros(n,1); equals n*log(2) for likCum*
        nlZ0=-sum(calLikGPC(Y,fval_reg,diag(cov),inf));
        if any(size(last_ttau) ~= [x_num 1])      % find starting point for tilde parameters
            ttau=zeros(x_num,1); tnu =zeros(x_num,1);        % init to zero if no better guess
            Sigma=cov;                     % initialize Sigma and mu,the parameters of ..
            mu=fval_reg; nlZ=nlZ0;                  % .. the Gaussian posterior approximation
        else
            ttau=last_ttau; tnu =last_tnu;   % try the tilde values from previous call
            [Sigma,mu,L,alpha,nlZ]=epComputeParams(cov,Y,ttau,tnu,fval_reg,inf);
            if nlZ > nlZ0                                           % if zero is better ..
                ttau=zeros(x_num,1); tnu =zeros(x_num,1);       % .. then init with zero instead
                Sigma=cov;                   % initialize Sigma and mu,the parameters of ..
                mu=fval_reg; nlZ=nlZ0;                % .. the Gaussian posterior approximation
            end
        end

        nlZ_old=Inf; sweep=0;               % converged,max. sweeps or min. sweeps?
        while (abs(nlZ-nlZ_old) > tol && sweep < max_sweep) || sweep<min_sweep
            nlZ_old=nlZ; sweep=sweep+1;
            for i=randperm(x_num)       % iterate EP updates (in random order) over examples
                tau_ni=1/Sigma(i,i)-ttau(i);      %  first find the cavity distribution ..
                nu_ni=mu(i)/Sigma(i,i)-tnu(i);                % .. params tau_ni and nu_ni

                % compute the desired derivatives of the indivdual log partition function
                [lZ,dlZ,d2lZ]=calLikGPC(Y(i),nu_ni/tau_ni,1/tau_ni,inf);
                ttau_old=ttau(i); tnu_old=tnu(i);  % find the new tilde params,keep old
                ttau(i)=-d2lZ/(1+d2lZ/tau_ni);
                ttau(i)=max(ttau(i),0); % enforce positivity i.e. lower bound ttau by zero
                tnu(i) =(dlZ-nu_ni/tau_ni*d2lZ)/(1+d2lZ/tau_ni);

                dtt=ttau(i)-ttau_old; dtn=tnu(i)-tnu_old;      % rank-1 update Sigma ..
                si=Sigma(:,i); ci=dtt/(1+dtt*si(i));
                Sigma=Sigma-ci*(si*si');                         % takes 70% of total time
                mu=mu-(ci*(mu(i)+si(i)*dtn)-dtn)*si;               % .. and recompute mu
            end
            % recompute since repeated rank-one updates can destroy numerical precision
            [Sigma,mu,L,alpha,nlZ]=epComputeParams(cov,Y,ttau,tnu,fval_reg,inf);
        end

%         if sweep == max_sweep && abs(nlZ-nlZ_old) > tol
%             error('maximum number of sweeps exceeded in function infEP')
%         end

        last_ttau=ttau; last_tnu=tnu;                       % remember for next call
        sW=sqrt(ttau);  % return posterior params
    end

    function [Sigma,mu,L,alpha,nlZ]=epComputeParams(K,y,ttau,tnu,m,inf_fcn)
        % function to compute the parameters of the Gaussian approximation,Sigma and
        % mu,and the negative log marginal likelihood,nlZ,from the current site
        % parameters,ttau and tnu. Also returns L (useful for predictions).
        %
        n=length(y);                                      % number of training cases
        sW=sqrt(ttau);                                        % compute Sigma and mu
        L=chol(eye(n)+sW*sW'.*K);                            % L'*L=B=eye(n)+sW*K*sW
        V=L'\(repmat(sW,1,n).*K);
        Sigma=K-V'*V;
        alpha=tnu-sW.*(L\(L'\(sW.*(K*tnu+m))));
        mu=K*alpha+m; v=diag(Sigma);

        tau_n=1./diag(Sigma)-ttau;             % compute the log marginal likelihood
        nu_n =mu./diag(Sigma)-tnu;                    % vectors of cavity parameters
        lZ=calLikGPC(y,nu_n./tau_n,1./tau_n,inf_fcn);
        p=tnu-m.*ttau; q=nu_n-m.*tau_n;                        % auxiliary vectors
        nlZ=sum(log(diag(L)))-sum(lZ)-p'*Sigma*p/2+(v'*p.^2)/2 ...
           -q'*((ttau./tau_n.*q-2*p).*v)/2-sum(log(1+ttau./tau_n))/2;
    end

    function [varargout]=calLikGPC(y,mu,s2,inf_fcn)
        % likErf-Error function or cumulative Gaussian likelihood function for binary
        % classification or probit regression. The expression for the likelihood is
        %   likErf(t)=(1+erf(t/sqrt(2)))/2=normcdf(t).
        %
        % Several modes are provided,for computing likelihoods,derivatives and moments
        % respectively,see likFunctions.m for the details. In general,care is taken
        % to avoid numerical issues when the arguments are extreme.
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2014-03-19.
        %
        % See also LIKFUNCTIONS.M.
        %
        if nargin<3,varargout={'0'}; return; end   % report number of hyperparameters
        if nargin>1,y=sign(y); y(y==0)=1; else y=1; end % allow only +/- 1 values
        if numel(y)==0,y=1; end

        if nargin<4                              % prediction mode if inf is not present
            y=y.*ones(size(mu));                                       % make y a vector
            s2zero=1; if nargin>3&&numel(s2)>0&&norm(s2)>eps,s2zero=0; end  % s2==0 ?
            if s2zero                                         % log probability evaluation
                lp=logphi(y.*mu);
            else                                                              % prediction
                lp=calLikGPC(y,mu,s2,'infEP');
            end
            p=exp(lp); ymu={}; ys2={};
            if nargout>1
                ymu=2*p-1;                                                % first y moment
                if nargout>2
                    ys2=4*p.*(1-p);                                        % second y moment
                end
            end
            varargout={lp,ymu,ys2};
        else                                                            % inference mode
            % no derivative mode
            z=mu./sqrt(1+s2); dlZ={}; d2lZ={};
            if numel(y)>0,z=z.*y; end
            if nargout <= 1,lZ=logphi(z);                         % log part function
            else          [lZ,n_p]=logphi(z); end
            if nargout > 1
                if numel(y)==0,y=1; end
                dlZ=y.*n_p./sqrt(1+s2);                      % 1st derivative wrt mean
                if nargout>2,d2lZ=-n_p.*(z+n_p)./(1+s2); end         % 2nd derivative
            end
            varargout={lZ,dlZ,d2lZ};
        end
    end

    function [lp,dlp,d2lp,d3lp]=logphi(z)
        % Safe computation of logphi(z)=log(normcdf(z)) and its derivatives
        %                    dlogphi(z)=normpdf(x)/normcdf(x).
        % The function is based on index 5725 in Hart et al. and gsl_sf_log_erfc_e.
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2013-11-13.
        %
        z=real(z);                                 % support for real arguments only
        lp=zeros(size(z));                                         % allocate memory
        id1=z.*z<0.0492;                                 % first case: close to zero
        lp0=-z(id1)/sqrt(2*pi);
        c=[ 0.00048204; -0.00142906; 0.0013200243174; 0.0009461589032;
            -0.0045563339802; 0.00556964649138; 0.00125993961762116;
            -0.01621575378835404; 0.02629651521057465; -0.001829764677455021;
            2*(1-pi/3); (4-pi)/3; 1; 1];
        f=0; for i=1:14,f=lp0.*(c(i)+f); end,lp(id1)=-2*f-log(2);
        id2=z<-11.3137;                                    % second case: very small
        r=[ 1.2753666447299659525; 5.019049726784267463450;
            6.1602098531096305441; 7.409740605964741794425;
            2.9788656263939928886 ];
        q=[ 2.260528520767326969592;  9.3960340162350541504;
            12.048951927855129036034; 17.081440747466004316;
            9.608965327192787870698;  3.3690752069827527677 ];
        num=0.5641895835477550741; for i=1:5,num=-z(id2).*num/sqrt(2)+r(i); end
        den=1.0;                   for i=1:6,den=-z(id2).*den/sqrt(2)+q(i); end
        e=num./den; lp(id2)=log(e/2)-z(id2).^2/2;
        id3=~id2 & ~id1; lp(id3)=log(erfc(-z(id3)/sqrt(2))/2);  % third case: rest
        if nargout>1                                        % compute first derivative
            dlp=zeros(size(z));                                      % allocate memory
            dlp(id2)=abs(den./num) * sqrt(2/pi); % strictly positive first derivative
            dlp(~id2)=exp(-z(~id2).*z(~id2)/2-lp(~id2))/sqrt(2*pi); % safe computation
            if nargout>2                                     % compute second derivative
                d2lp=-dlp.*abs(z+dlp);             % strictly negative second derivative
                if nargout>3                                    % compute third derivative
                    d3lp=-d2lp.*abs(z+2*dlp)-dlp;     % strictly positive third derivative
                end
            end
        end
    end

    function F_reg=regMinus(X)
        % zero order base funcion
        %
        F_reg=zeros(size(X,1),1); % zero
    end

    function F_reg=regZero(X)
        % zero order base funcion
        %
        F_reg=ones(size(X,1),1); % zero
    end

    function F_reg=regLinear(X)
        % first order base funcion
        %
        F_reg=[ones(size(X,1),1),X]; % linear
    end
end
