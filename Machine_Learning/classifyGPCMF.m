clc;
% clear;
close all hidden;

% func_LF = @(x) LFFcn(x);
% func_HF = @(x) HFFcn(x);
% 
% func_LF_bou = @(x) 0.45+sin(2.2*pi*x)/2.5;
% func_HF_bou = @(x) 0.5+sin(2.5*pi*x)/3;
% 
% variable_number = 2;
% 
% x_LF_number = 30;
% X_LF = lhsdesign(x_LF_number,variable_number);
% Class_LF = zeros(x_LF_number,1);
% for x_index = 1:x_LF_number
%     Class_LF(x_index) = func_LF(X_LF(x_index,:));
% end
% 
% [Class_LF,order] = sort(Class_LF);
% X_LF = X_LF(order,:);
% 
% x_HF_number = 5;
% X_HF = lhsdesign(x_HF_number,variable_number);
% Class_HF = zeros(x_HF_number,1);
% for x_index = 1:x_HF_number
%     Class_HF(x_index) = func_HF(X_HF(x_index,:));
% end
% 
% [Class_HF,order] = sort(Class_HF);
% X_HF = X_HF(order,:);
% 
% low_bou = [0,0];
% up_bou = [1,1];

% load('CMF_5.mat');
% 
% [predict_function_MF,CGPMF_model] = classifyGaussProcessMultiFidelity...
%     (X_HF,Class_HF,X_LF,Class_LF);
% [predict_function,CGP_model] = classifyGaussProcess...
%     (X_HF,Class_HF);



fig_hdl=figure(1);
fig_hdl.set('Position',[488   200   680  420])

subplot(1,2,1)
axes_handle=gca;
axes_handle.set('Position',[0.0800,0.1500,0.38,0.6750]);
% draw zero value line
grid_number=100;
d_bou=(up_bou-low_bou)/grid_number;
[X_draw,Y_draw]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));

% generate all predict list
predict_x=[X_draw(:),Y_draw(:)];
[predict_class,predict_probability]=predict_function(predict_x);
predict_probability=reshape(predict_probability,grid_number+1,grid_number+1);

contourf(axes_handle,X_draw,Y_draw,predict_probability*2-1,'LineStyle','none');
shading interp;
load('color.mat','my_color_map_light');
colormap(my_color_map_light);

hold on;
% draw bou
X_draw=0:0.01:1;
scatter(X_HF((Class_HF>0),1),X_HF((Class_HF>0),2),'or')
scatter(X_HF((Class_HF<0),1),X_HF((Class_HF<0),2),'ok')
line(axes_handle,X_draw,func_HF_bou(X_draw),'LineStyle','-');

xlabel('x_1');ylabel('x_2');grid on;


subplot(1,2,2)
axes_handle=gca;
axes_handle.set('Position',[0.5400,0.1500,0.38,0.6750]);
% draw zero value line
grid_number=100;
d_bou=(up_bou-low_bou)/grid_number;
[X_draw,Y_draw]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));

% generate all predict list
predict_x=[X_draw(:),Y_draw(:)];
[predict_class,predict_probability]=predict_function_MF(predict_x);
predict_probability=reshape(predict_probability,grid_number+1,grid_number+1);

contourf(axes_handle,X_draw,Y_draw,predict_probability*2-1,'LineStyle','none');
shading interp;
load('color.mat','my_color_map_light');
colormap(my_color_map_light);

hold on;
% draw bou
X_draw=0:0.01:1;
SP_HF=scatter(X_HF((Class_HF>0),1),X_HF((Class_HF>0),2),'or');
SN_HF=scatter(X_HF((Class_HF<0),1),X_HF((Class_HF<0),2),'ok');
line_HF=line(axes_handle,X_draw,func_HF_bou(X_draw),'LineStyle','-');
SP_LF=scatter(X_LF((Class_LF>0),1),X_LF((Class_LF>0),2),'*r');
SN_LF=scatter(X_LF((Class_LF<0),1),X_LF((Class_LF<0),2),'*k');
line_LF=line(axes_handle,X_draw,func_LF_bou(X_draw),'LineStyle','--');

xlabel('x_1');ylabel('x_2');grid on;

legend([SP_HF,SN_HF,line_HF,SP_LF,SN_LF,line_LF],'高精度正样本点','高精度负样本点','高精度分类边界','低精度正样本点','低精度负样本点','低精度分类边界','Location','northwest','Orientation','horizontal','NumColumns',3)


function [predict_function,CGPMF_model] = classifyGaussProcessMultiFidelity...
    (X_HF,Class_HF,X_LF,Class_LF,hyp)
% generate multi fidelity gaussian process classifier model
% version 6,this version is assembly of gpml-3.6 EP method
% X,XHF,XLF is x_number x variable_number matirx
% Class,ClassH,ClassLF is x_number x 1 matrix
% low_bou,up_bou is 1 x variable_number matrix
%
% input:
% XHF,ClassHF,XLF,ClassLF,hyp(mean,cov(lenD,etaD,lenL,etaL,rho))
%
% abbreviation:
% pred: predicted,nomlz: normalization,num: number
% var: variance
%
% reference: [1]FSCAB C,PP D,EK E,et al. Multi-fidelity classification
% using Gaussian processes: Accelerating the prediction of large-scale
% computational models [J]. Computer Methods in Applied Mechanics and
% Engineering,357(C): 112602-.
%
X = [X_HF;X_LF];
Class = [Class_HF;Class_LF];
[x_number,variable_number] = size(X);
x_HF_number = size(X_HF,1);
x_LF_number = size(X_LF,1);
if nargin < 5
    hyp.mean = 0;
    hyp.cov = zeros(1,variable_number*2+3);
end

% normalization data
aver_X = mean(X);
stdD_X = std(X);
index__ = find(stdD_X  ==  0);
if  ~isempty(index__),stdD_X(index__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;

object_function = @(x) objectNLLMFGPC(x,{@infEP},{@meanConst},{@calCovMF},{@likErf},X_nomlz,Class);
hyp_x = [hyp.mean,hyp.cov];

% [fval,gradient] = object_function(hyp_x)
% [fval_differ,gradient_differ] = differ(object_function,hyp_x)

low_bou_hyp = -4*ones(1,2*variable_number+4);
up_bou_hyp = 4*ones(1,2*variable_number+4);
hyp_x = fmincon(object_function,hyp_x,[],[],[],[],low_bou_hyp,up_bou_hyp,[],...
    optimoptions('fmincon','Display','iter','SpecifyObjectiveGradient',true,...
    'MaxFunctionEvaluations',20,'OptimalityTolerance',1e-6));

hyp.mean = hyp_x(1);
hyp.cov = hyp_x(2:end);
hyp.lik = [];
post = infEP(hyp,{@meanConst},{@calCovMF},{@likErf},X_nomlz,Class);
predict_function = @(x_pred) classifyGaussPredictor...
    (x_pred,hyp,{@meanConst},{@calCovMF},{@likErf},post,X_nomlz,aver_X,stdD_X);

% output model
X = {X_HF,X_LF};
Class = {Class_HF,Class_LF};
CGPMF_model.X = X;
CGPMF_model.Class = Class;
CGPMF_model.aver_X = aver_X;
CGPMF_model.stdD_X = stdD_X;
CGPMF_model.predict_function = predict_function;
CGPMF_model.hyp = hyp;
CGPMF_model.post = post;

    function [fval,gradient] = objectNLLMFGPC(x,inf,mean,cov,lik,X,Y)
        hyp_iter.mean = x(1);
        hyp_iter.cov = x(2:end);
        hyp_iter.lik = [];

        if nargout < 2
            [~,nlZ] = feval(inf{:},hyp_iter,mean,cov,lik,X,Y);
            fval = nlZ;
        elseif nargout < 3
            [~,nlZ,dnlZ] = feval(inf{:},hyp_iter,mean,cov,lik,X,Y);
            fval = nlZ;
            gradient = [dnlZ.mean,dnlZ.cov];
        end
    end

    function [class,possibility,miu_pre,var_pre] = classifyGaussPredictor...
            (X_pred,hyp,mean,cov,lik,post,X,aver_X,stdD_X)
        % predict function
        %
        X_pred_nomlz = (X_pred-aver_X)./stdD_X;
        pred_num = size(X_pred_nomlz,1);
        ys = ones(pred_num,1);

        alpha = post.alpha; L = post.L; sW = post.sW;
        %verify whether L contains valid Cholesky decomposition or something different
        Lchol = isnumeric(L) && all(all(tril(L,-1) == 0)&diag(L)'>0&isreal(diag(L))');
        ns = size(X_pred_nomlz,1);                                       % number of data points
        nperbatch = 1000;                       % number of data points per mini batch
        nact = 0;                       % number of already processed test data points
        ymu = zeros(ns,1); ys2 = ymu; miu_pre = ymu; var_pre = ymu; possibility = ymu;   % allocate mem
        while nact<ns               % process minibatches of test cases to save memory
            id = (nact+1):min(nact+nperbatch,ns);               % data points to process
            kss = feval(cov{:},hyp.cov,X_pred_nomlz(id,:),'diag');              % self-variance
            Ks = feval(cov{:},hyp.cov,X,X_pred_nomlz(id,:));        % avoid computation
            ms = feval(mean{:},hyp.mean,X_pred_nomlz(id,:));
            N = size(alpha,2);  % number of alphas (usually 1; more in case of sampling)
            Fmu = repmat(ms,1,N) + Ks'*full(alpha);        % conditional mean fs|f
            miu_pre(id) = sum(Fmu,2)/N;                                   % predictive means
            if Lchol    % L contains chol decomp  = > use Cholesky parameters (alpha,sW,L)
                V  = L'\(repmat(sW,1,length(id)).*Ks);
                var_pre(id) = kss - sum(V.*V,1)';                       % predictive variances
            else                % L is not triangular  = > use alternative parametrisation
                if isnumeric(L),LKs = L*Ks; else LKs = L(Ks); end    % matrix or callback
                var_pre(id) = kss + sum(Ks.*LKs,1)';                    % predictive variances
            end
            var_pre(id) = max(var_pre(id),0);   % remove numerical noise i.e. negative variances
            Fs2 = repmat(var_pre(id),1,N);     % we have multiple values in case of sampling
            if nargin<9
                [Lp,Ymu,Ys2] = feval(lik{:},hyp.lik,[],Fmu(:),Fs2(:));
            else
                Ys = repmat(ys(id),1,N);
                [Lp,Ymu,Ys2] = feval(lik{:},hyp.lik,Ys(:),Fmu(:),Fs2(:));
            end
            possibility(id)  = sum(reshape(Lp,[],N),2)/N;    % log probability; sample averaging
            ymu(id) = sum(reshape(Ymu,[],N),2)/N;          % predictive mean ys|y and ..
            ys2(id) = sum(reshape(Ys2,[],N),2)/N;                          % .. variance
            nact = id(end);          % set counter to index of last processed data point
        end

        possibility = exp(possibility);
        class = ones(pred_num,1);
        index_list = possibility < 0.5;
        class(index_list) = -1;
    end

    function [K,dK_dcov] = calCovMF(cov,X,Z)
        % obtain covariance of x
        % cov: lenD,etaD,lenL,etaL,rho
        % len equal to 1/len_origin.^2
        %
        % % k = eta*exp(-sum(x_dis*theta)/vari_num);
        %
        [x_num,vari_num] = size(X);

        lenD = exp(cov(1:vari_num));
        etaD = exp(cov(vari_num+1));
        lenL = exp(cov(vari_num+1+(1:vari_num)));
        etaL = exp(cov(2*(vari_num+1)));
        rho = exp(cov(end));

        if nargin > 2 && nargout < 2 && ~isempty(Z)
            if strcmp(Z,'diag')
                K = rho*rho*etaL+etaD;
                return
            end
        end

        % predict
        if nargin > 2 && nargout < 2 && ~isempty(Z)
            [z_num,vari_num] = size(Z);
            % initializate square of X inner distance sq/ vari_num
            sq_dis_v = zeros(x_num,z_num,vari_num);
            for len_index = 1:vari_num
                sq_dis_v(:,:,len_index) = (X(:,len_index)-Z(:,len_index)').^2/vari_num;
            end

            % exp of x__x with D
            exp_disD = zeros(x_HF_number,z_num);
            for len_index = 1:vari_num
                exp_disD = exp_disD+...
                    sq_dis_v(1:x_HF_number,:,len_index)*lenD(len_index);
            end
            exp_disD = exp(-exp_disD);

            % exp of x__x with L
            exp_disL = zeros(x_num,z_num);
            for len_index = 1:vari_num
                exp_disL = exp_disL+...
                    sq_dis_v(1:x_num,:,len_index)*lenL(len_index);
            end
            exp_disL = exp(-exp_disL);

            K = exp_disL;
            K(1:x_HF_number,:) = rho*rho*etaL*K(1:x_HF_number,:)+etaD*exp_disD;
            K(x_HF_number+1:end,:) = rho*etaL*K(x_HF_number+1:end,:);
        else
            % initializate square of X inner distance sq/ vari_num
            sq_dis_v = zeros(x_num,x_num,vari_num);
            for len_index = 1:vari_num
                sq_dis_v(:,:,len_index) = (X(:,len_index)-X(:,len_index)').^2/vari_num;
            end

            % exp of x__x with H
            exp_disD = zeros(x_num);
            for len_index = 1:vari_num
                exp_disD(1:x_HF_number,1:x_HF_number) = exp_disD(1:x_HF_number,1:x_HF_number)+...
                    sq_dis_v(1:x_HF_number,1:x_HF_number,len_index)*lenD(len_index);
            end
            exp_disD(1:x_HF_number,1:x_HF_number) = exp(-exp_disD(1:x_HF_number,1:x_HF_number));
            KD = etaD*exp_disD;

            % exp of x__x with L
            exp_disL = zeros(x_num);
            for len_index = 1:vari_num
                exp_disL = exp_disL+...
                    sq_dis_v(1:end,1:end,len_index)*lenL(len_index);
            end
            exp_disL = exp(-exp_disL);
            eta_exp_disL = etaL*exp_disL;

            % times rho: HH to rho2,HL to rho,LL to 1
            KL = eta_exp_disL;
            KL(1:x_HF_number,1:x_HF_number) = ...
                (rho*rho)*eta_exp_disL(1:x_HF_number,1:x_HF_number);
            KL(1:x_HF_number,(x_HF_number+1):end) = ...
                rho*eta_exp_disL(1:x_HF_number,(x_HF_number+1):end);
            KL((x_HF_number+1):end,1:x_HF_number) = ...
                KL(1:x_HF_number,(x_HF_number+1):end)';

            K = KL+KD;

            if nargout >= 2
                dK_dcov = cell(1,2*vari_num+3);

                % len D
                for len_index = 1:vari_num
                    dK_dlenD = zeros(x_num);
                    dK_dlenD(1:x_HF_number,1:x_HF_number) = -KD(1:x_HF_number,1:x_HF_number).*...
                        sq_dis_v(1:x_HF_number,1:x_HF_number,len_index)*lenD(len_index);
                    dK_dcov{len_index} = dK_dlenD;
                end

                % eta D
                dK_dcov{vari_num+1} = KD;

                % len L
                for len_index = 1:vari_num
                    dK_dlenL = -KL.*sq_dis_v(:,:,len_index)*lenL(len_index);
                    dK_dcov{(vari_num+1)+len_index} = dK_dlenL;
                end

                % eta L
                dK_dcov{2*(vari_num+1)} = KL;

                % rho
                dK_drho = zeros(x_num);
                dK_drho(1:x_HF_number,1:x_HF_number) = ...
                    2*rho*rho*eta_exp_disL(1:x_HF_number,1:x_HF_number);
                dK_drho(1:x_HF_number,(x_HF_number+1):end) = ...
                    rho*eta_exp_disL(1:x_HF_number,(x_HF_number+1):end);
                dK_drho((x_HF_number+1):end,1:x_HF_number) = ...
                    dK_drho(1:x_HF_number,(x_HF_number+1):end)';
                dK_dcov{end} = dK_drho;
            end
        end

    end

    function [post,nlZ,dnlZ] = infEP(hyp,mean,cov,lik,x,y)
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
        tol = 1e-4; max_sweep = 10; min_sweep = 2;     % tolerance to stop EP iterations

        inf = 'infEP';
        if isnumeric(cov),K = cov;                    % use provided covariance matrix
        else K = feval(cov{:},hyp.cov,x); end       % evaluate the covariance matrix

        n = size(x,1);

        if isnumeric(mean),m = mean;                         % use provided mean vector
        else m = feval(mean{:},hyp.mean,x); end             % evaluate the mean vector

        % A note on naming: variables are given short but descriptive names in
        % accordance with Rasmussen & Williams "GPs for Machine Learning" (2006): mu
        % and s2 are mean and variance,nu and tau are natural parameters. A leading t
        % means tilde,a subscript _ni means "not i" (for cavity parameters),or _n
        % for a vector of cavity parameters. N(f|mu,Sigma) is the posterior.

        % marginal likelihood for ttau = tnu = zeros(n,1); equals n*log(2) for likCum*
        nlZ0 = -sum(feval(lik{:},hyp.lik,y,m,diag(K),inf));
        if any(size(last_ttau) ~= [n 1])      % find starting point for tilde parameters
            ttau = zeros(n,1); tnu  = zeros(n,1);        % init to zero if no better guess
            Sigma = K;                     % initialize Sigma and mu,the parameters of ..
            mu = m; nlZ = nlZ0;                  % .. the Gaussian posterior approximation
        else
            ttau = last_ttau; tnu  = last_tnu;   % try the tilde values from previous call
            [Sigma,mu,L,alpha,nlZ] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,inf);
            if nlZ > nlZ0                                           % if zero is better ..
                ttau = zeros(n,1); tnu  = zeros(n,1);       % .. then init with zero instead
                Sigma = K;                   % initialize Sigma and mu,the parameters of ..
                mu = m; nlZ = nlZ0;                % .. the Gaussian posterior approximation
            end
        end

        nlZ_old = Inf; sweep = 0;               % converged,max. sweeps or min. sweeps?
        while (abs(nlZ-nlZ_old) > tol && sweep < max_sweep) || sweep<min_sweep
            nlZ_old = nlZ; sweep = sweep+1;
            for i = randperm(n)       % iterate EP updates (in random order) over examples
                tau_ni = 1/Sigma(i,i)-ttau(i);      %  first find the cavity distribution ..
                nu_ni = mu(i)/Sigma(i,i)-tnu(i);                % .. params tau_ni and nu_ni

                % compute the desired derivatives of the indivdual log partition function
                [lZ,dlZ,d2lZ] = feval(lik{:},hyp.lik,y(i),nu_ni/tau_ni,1/tau_ni,inf);
                ttau_old = ttau(i); tnu_old = tnu(i);  % find the new tilde params,keep old
                ttau(i) = -d2lZ /(1+d2lZ/tau_ni);
                ttau(i) = max(ttau(i),0); % enforce positivity i.e. lower bound ttau by zero
                tnu(i)  = ( dlZ - nu_ni/tau_ni*d2lZ )/(1+d2lZ/tau_ni);

                dtt = ttau(i)-ttau_old; dtn = tnu(i)-tnu_old;      % rank-1 update Sigma ..
                si = Sigma(:,i); ci = dtt/(1+dtt*si(i));
                Sigma = Sigma - ci*si*si';                         % takes 70% of total time
                mu = mu - (ci*(mu(i)+si(i)*dtn)-dtn)*si;               % .. and recompute mu
            end
            % recompute since repeated rank-one updates can destroy numerical precision
            [Sigma,mu,L,alpha,nlZ] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,inf);
        end

        if sweep  ==  max_sweep && abs(nlZ-nlZ_old) > tol
            error('maximum number of sweeps exceeded in function infEP')
        end

        last_ttau = ttau; last_tnu = tnu;                       % remember for next call
        post.alpha = alpha; post.sW = sqrt(ttau); post.L = L;  % return posterior params

        if nargout>2                                           % do we want derivatives?
            dnlZ = hyp;                                   % allocate space for derivatives
            tau_n = 1./diag(Sigma)-ttau;             % compute the log marginal likelihood
            nu_n  = mu./diag(Sigma)-tnu;                    % vectors of cavity parameters
            sW = sqrt(ttau);
            F = alpha*alpha'-repmat(sW,1,n).*(L\(L'\diag(sW)));   % covariance hypers
            [K,dK] = feval(cov{:},hyp.cov,x,[]);
            for i = 1:length(hyp.cov)
                dnlZ.cov(i) = -sum(sum(F.*dK{i}))/2;
            end
            for i = 1:numel(hyp.lik)                                   % likelihood hypers
                dlik = feval(lik{:},hyp.lik,y,nu_n./tau_n,1./tau_n,inf,i);
                dnlZ.lik(i) = -sum(dlik);
            end
            [junk,dlZ] = feval(lik{:},hyp.lik,y,nu_n./tau_n,1./tau_n,inf);% mean hyps
            for i = 1:numel(hyp.mean)
                dm = feval(mean{:},hyp.mean,x,i);
                dnlZ.mean(i) = -dlZ'*dm;
            end
        end
    end

    function [Sigma,mu,L,alpha,nlZ] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,inf)
        % function to compute the parameters of the Gaussian approximation,Sigma and
        % mu,and the negative log marginal likelihood,nlZ,from the current site
        % parameters,ttau and tnu. Also returns L (useful for predictions).
        %
        n = length(y);                                      % number of training cases
        sW = sqrt(ttau);                                        % compute Sigma and mu
        L = chol(eye(n)+sW*sW'.*K);                            % L'*L = B = eye(n)+sW*K*sW
        V = L'\(repmat(sW,1,n).*K);
        Sigma = K - V'*V;
        alpha = tnu-sW.*(L\(L'\(sW.*(K*tnu+m))));
        mu = K*alpha+m; v = diag(Sigma);

        tau_n = 1./diag(Sigma)-ttau;             % compute the log marginal likelihood
        nu_n  = mu./diag(Sigma)-tnu;                    % vectors of cavity parameters
        lZ = feval(lik{:},hyp.lik,y,nu_n./tau_n,1./tau_n,inf);
        p = tnu-m.*ttau; q = nu_n-m.*tau_n;                        % auxiliary vectors
        nlZ = sum(log(diag(L))) - sum(lZ) - p'*Sigma*p/2 + (v'*p.^2)/2 ...
            - q'*((ttau./tau_n.*q-2*p).*v)/2 - sum(log(1+ttau./tau_n))/2;

    end

    function A = meanConst(hyp,x,i)
        % Constant mean function. The mean function is parameterized as:
        %
        % m(x) = c
        %
        % The hyperparameter is:
        %
        % hyp = [ c ]
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2010-08-04.
        %
        % See also MEANFUNCTIONS.M.
        %
        if nargin<2,A = '1'; return; end             % report number of hyperparameters
        if numel(hyp)~=1,error('Exactly one hyperparameter needed.'),end
        c = hyp;
        if nargin == 2
            A = c*ones(size(x,1),1);                                       % evaluate mean
        else
            if i == 1
                A = ones(size(x,1),1);                                          % derivative
            else
                A = zeros(size(x,1),1);
            end
        end
    end

    function [varargout] = likErf(hyp,y,mu,s2,inf)
        % likErf - Error function or cumulative Gaussian likelihood function for binary
        % classification or probit regression. The expression for the likelihood is
        %   likErf(t) = (1+erf(t/sqrt(2)))/2 = normcdf(t).
        %
        % Several modes are provided,for computing likelihoods,derivatives and moments
        % respectively,see likFunctions.m for the details. In general,care is taken
        % to avoid numerical issues when the arguments are extreme.
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2014-03-19.
        %
        % See also LIKFUNCTIONS.M.
        %
        if nargin<3,varargout = {'0'}; return; end   % report number of hyperparameters
        if nargin>1,y = sign(y); y(y == 0) = 1; else y = 1; end % allow only +/- 1 values
        if numel(y) == 0,y = 1; end

        if nargin<5                              % prediction mode if inf is not present
            y = y.*ones(size(mu));                                       % make y a vector
            s2zero = 1; if nargin>3&&numel(s2)>0&&norm(s2)>eps,s2zero = 0; end  % s2 == 0 ?
            if s2zero                                         % log probability evaluation
                lp = logphi(y.*mu);
            else                                                              % prediction
                lp = likErf(hyp,y,mu,s2,'infEP');
            end
            p = exp(lp); ymu = {}; ys2 = {};
            if nargout>1
                ymu = 2*p-1;                                                % first y moment
                if nargout>2
                    ys2 = 4*p.*(1-p);                                        % second y moment
                end
            end
            varargout = {lp,ymu,ys2};
        else                                                            % inference mode
            switch inf
                case 'infLaplace'
                    if nargin<6                                             % no derivative mode
                        f = mu; yf = y.*f;                            % product latents and labels
                        varargout = cell(nargout,1); [varargout{:}] = logphi(yf);   % query logphi
                        if nargout>1
                            varargout{2} = y.*varargout{2};
                            if nargout>3,varargout{4} = y.*varargout{4}; end
                        end
                    else                                                       % derivative mode
                        varargout = {[],[],[]};                         % derivative w.r.t. hypers
                    end

                case 'infEP'
                    if nargin<6                                             % no derivative mode
                        z = mu./sqrt(1+s2); dlZ = {}; d2lZ = {};
                        if numel(y) > 0,z = z.*y; end
                        if nargout <= 1,lZ = logphi(z);                         % log part function
                        else          [lZ,n_p] = logphi(z); end
                        if nargout > 1
                            if numel(y) == 0,y = 1; end
                            dlZ = y.*n_p./sqrt(1+s2);                      % 1st derivative wrt mean
                            if nargout>2,d2lZ = -n_p.*(z+n_p)./(1+s2); end         % 2nd derivative
                        end
                        varargout = {lZ,dlZ,d2lZ};
                    else                                                       % derivative mode
                        varargout = {[]};                                     % deriv. wrt hyp.lik
                    end
            end
        end
    end

    function [lp,dlp,d2lp,d3lp] = logphi(z)
        % Safe computation of logphi(z) = log(normcdf(z)) and its derivatives
        %                    dlogphi(z) = normpdf(x)/normcdf(x).
        % The function is based on index 5725 in Hart et al. and gsl_sf_log_erfc_e.
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2013-11-13.
        %
        z = real(z);                                 % support for real arguments only
        lp = zeros(size(z));                                         % allocate memory
        id1 = z.*z<0.0492;                                 % first case: close to zero
        lp0 = -z(id1)/sqrt(2*pi);
        c = [ 0.00048204; -0.00142906; 0.0013200243174; 0.0009461589032;
            -0.0045563339802; 0.00556964649138; 0.00125993961762116;
            -0.01621575378835404; 0.02629651521057465; -0.001829764677455021;
            2*(1-pi/3); (4-pi)/3; 1; 1];
        f = 0; for i = 1:14,f = lp0.*(c(i)+f); end,lp(id1) = -2*f-log(2);
        id2 = z<-11.3137;                                    % second case: very small
        r = [ 1.2753666447299659525; 5.019049726784267463450;
            6.1602098531096305441; 7.409740605964741794425;
            2.9788656263939928886 ];
        q = [ 2.260528520767326969592;  9.3960340162350541504;
            12.048951927855129036034; 17.081440747466004316;
            9.608965327192787870698;  3.3690752069827527677 ];
        num = 0.5641895835477550741; for i = 1:5,num = -z(id2).*num/sqrt(2) + r(i); end
        den = 1.0;                   for i = 1:6,den = -z(id2).*den/sqrt(2) + q(i); end
        e = num./den; lp(id2) = log(e/2) - z(id2).^2/2;
        id3 = ~id2 & ~id1; lp(id3) = log(erfc(-z(id3)/sqrt(2))/2);  % third case: rest
        if nargout>1                                        % compute first derivative
            dlp = zeros(size(z));                                      % allocate memory
            dlp( id2) = abs(den./num) * sqrt(2/pi); % strictly positive first derivative
            dlp(~id2) = exp(-z(~id2).*z(~id2)/2-lp(~id2))/sqrt(2*pi); % safe computation
            if nargout>2                                     % compute second derivative
                d2lp = -dlp.*abs(z+dlp);             % strictly negative second derivative
                if nargout>3                                    % compute third derivative
                    d3lp = -d2lp.*abs(z+2*dlp)-dlp;     % strictly positive third derivative
                end
            end
        end
    end

end

% function [predict_function,CGPMF_model] = classifyGaussProcessMultiFidelity...
%     (X_HF,Class_HF,X_LF,Class_LF,hyp)
% % generate multi fidelity gaussian process classifier model
% % version 6,this version is assembly of gpml-3.6 EP method
% % X,XHF,XLF is x_number x variable_number matirx
% % Class,ClassH,ClassLF is x_number x 1 matrix
% % low_bou,up_bou is 1 x variable_number matrix
% %
% % input:
% % XHF,ClassHF,XLF,ClassLF,hyp(mean,cov(lenD,etaD,lenL,etaL,rho))
% %
% % abbreviation:
% % pred: predicted,nomlz: normalization,num: number
% % var: variance
% %
% % reference: [1]FSCAB C,PP D,EK E,et al. Multi-fidelity classification
% % using Gaussian processes: Accelerating the prediction of large-scale
% % computational models [J]. Computer Methods in Applied Mechanics and
% % Engineering,357(C): 112602-.
% %
% X = [X_HF;X_LF];
% Class = [Class_HF;Class_LF];
% [x_number,variable_number] = size(X);
% x_HF_number = size(X_HF,1);
% x_LF_number = size(X_LF,1);
% if nargin < 5
%     hyp.mean = 0;
%     hyp.cov = zeros(1,5);
% end
% 
% % normalization data
% aver_X = mean(X);
% stdD_X = std(X);
% index__ = find(stdD_X  ==  0);
% if  ~isempty(index__),stdD_X(index__) = 1; end
% X_nomlz = (X-aver_X)./stdD_X;
% 
% object_function = @(x) objectNLLMFGPC(x,{@infEP},{@meanConst},{@calCovMF},{@likErf},X_nomlz,Class);
% hyp_x = [hyp.mean,hyp.cov];
% 
% % [fval,gradient] = object_function(hyp_x)
% % [fval_differ,gradient_differ] = differ(object_function,hyp_x)
% 
% low_bou_hyp = -3*ones(1,6);
% up_bou_hyp = 3*ones(1,6);
% hyp_x = fmincon(object_function,hyp_x,[],[],[],[],low_bou_hyp,up_bou_hyp,[],...
%     optimoptions('fmincon','Display','iter','SpecifyObjectiveGradient',true,...
%     'MaxFunctionEvaluations',20,'OptimalityTolerance',1e-6));
% 
% hyp.mean = hyp_x(1);
% hyp.cov = hyp_x(2:end);
% hyp.lik = [];
% post = infEP(hyp,{@meanConst},{@calCovMF},{@likErf},X_nomlz,Class);
% predict_function = @(x_pred) classifyGaussPredictor...
%     (x_pred,hyp,{@meanConst},{@calCovMF},{@likErf},post,X_nomlz,aver_X,stdD_X);
% 
% % output model
% X = {X_HF,X_LF};
% Class = {Class_HF,Class_LF};
% CGPMF_model.X = X;
% CGPMF_model.Class = Class;
% CGPMF_model.aver_X = aver_X;
% CGPMF_model.stdD_X = stdD_X;
% CGPMF_model.predict_function = predict_function;
% CGPMF_model.hyp = hyp;
% CGPMF_model.post = post;
% 
%     function [fval,gradient] = objectNLLMFGPC(x,inf,mean,cov,lik,X,Y)
%         hyp_iter.mean = x(1);
%         hyp_iter.cov = x(2:end);
%         hyp_iter.lik = [];
% 
%         if nargout < 2
%             [~,nlZ] = feval(inf{:},hyp_iter,mean,cov,lik,X,Y);
%             fval = nlZ;
%         elseif nargout < 3
%             [~,nlZ,dnlZ] = feval(inf{:},hyp_iter,mean,cov,lik,X,Y);
%             fval = nlZ;
%             gradient = [dnlZ.mean,dnlZ.cov];
%         end
%     end
% 
%     function [class,possibility,miu_pre,var_pre] = classifyGaussPredictor...
%             (X_pred,hyp,mean,cov,lik,post,X,aver_X,stdD_X)
%         % predict function
%         %
%         X_pred_nomlz = (X_pred-aver_X)./stdD_X;
%         pred_num = size(X_pred_nomlz,1);
%         ys = ones(pred_num,1);
% 
%         alpha = post.alpha; L = post.L; sW = post.sW;
%         %verify whether L contains valid Cholesky decomposition or something different
%         Lchol = isnumeric(L) && all(all(tril(L,-1) == 0)&diag(L)'>0&isreal(diag(L))');
%         ns = size(X_pred_nomlz,1);                                       % number of data points
%         nperbatch = 1000;                       % number of data points per mini batch
%         nact = 0;                       % number of already processed test data points
%         ymu = zeros(ns,1); ys2 = ymu; miu_pre = ymu; var_pre = ymu; possibility = ymu;   % allocate mem
%         while nact<ns               % process minibatches of test cases to save memory
%             id = (nact+1):min(nact+nperbatch,ns);               % data points to process
%             kss = feval(cov{:},hyp.cov,X_pred_nomlz(id,:),'diag');              % self-variance
%             Ks = feval(cov{:},hyp.cov,X,X_pred_nomlz(id,:));        % avoid computation
%             ms = feval(mean{:},hyp.mean,X_pred_nomlz(id,:));
%             N = size(alpha,2);  % number of alphas (usually 1; more in case of sampling)
%             Fmu = repmat(ms,1,N) + Ks'*full(alpha);        % conditional mean fs|f
%             miu_pre(id) = sum(Fmu,2)/N;                                   % predictive means
%             if Lchol    % L contains chol decomp  = > use Cholesky parameters (alpha,sW,L)
%                 V  = L'\(repmat(sW,1,length(id)).*Ks);
%                 var_pre(id) = kss - sum(V.*V,1)';                       % predictive variances
%             else                % L is not triangular  = > use alternative parametrisation
%                 if isnumeric(L),LKs = L*Ks; else LKs = L(Ks); end    % matrix or callback
%                 var_pre(id) = kss + sum(Ks.*LKs,1)';                    % predictive variances
%             end
%             var_pre(id) = max(var_pre(id),0);   % remove numerical noise i.e. negative variances
%             Fs2 = repmat(var_pre(id),1,N);     % we have multiple values in case of sampling
%             if nargin<9
%                 [Lp,Ymu,Ys2] = feval(lik{:},hyp.lik,[],Fmu(:),Fs2(:));
%             else
%                 Ys = repmat(ys(id),1,N);
%                 [Lp,Ymu,Ys2] = feval(lik{:},hyp.lik,Ys(:),Fmu(:),Fs2(:));
%             end
%             possibility(id)  = sum(reshape(Lp,[],N),2)/N;    % log probability; sample averaging
%             ymu(id) = sum(reshape(Ymu,[],N),2)/N;          % predictive mean ys|y and ..
%             ys2(id) = sum(reshape(Ys2,[],N),2)/N;                          % .. variance
%             nact = id(end);          % set counter to index of last processed data point
%         end
% 
%         possibility = exp(possibility);
%         class = ones(pred_num,1);
%         index_list = possibility < 0.5;
%         class(index_list) = -1;
%     end
% 
%     function [K,dK_dcov] = calCovMF(cov,X,Z)
%         % obtain covariance of x
%         % cov: lenD,etaD,lenL,etaL,rho
%         % len equal to 1/len_origin.^2
%         %
%         % % k = eta*exp(-sum(x_dis*theta)/vari_num);
%         %
%         [x_num,vari_num] = size(X);
% 
%         lenD = exp(cov(1));
%         etaD = exp(cov(2));
%         lenL = exp(cov(3));
%         etaL = exp(cov(4));
%         rho = exp(cov(5));
% 
%         if nargin > 2 && nargout < 2 && ~isempty(Z)
%             % predict
%             if strcmp(Z,'diag')
%                 K = rho*rho*etaL+etaD;
%             else
%                 [z_num,vari_num] = size(Z);
%                 % initializate square of X inner distance sq/ vari_num
%                 sq_dis_v = zeros(x_num,z_num,vari_num);
%                 for len_index = 1:vari_num
%                     sq_dis_v(:,:,len_index) = (X(:,len_index)-Z(:,len_index)').^2/vari_num;
%                 end
% 
%                 % exp of x__x with D
%                 exp_disD = zeros(x_HF_number,z_num);
%                 for len_index = 1:vari_num
%                     exp_disD = exp_disD+...
%                         sq_dis_v(1:x_HF_number,:,len_index)*lenD;
%                 end
%                 exp_disD = exp(-exp_disD);
% 
%                 % exp of x__x with L
%                 exp_disL = zeros(x_num,z_num);
%                 for len_index = 1:vari_num
%                     exp_disL = exp_disL+...
%                         sq_dis_v(1:x_num,:,len_index)*lenL;
%                 end
%                 exp_disL = exp(-exp_disL);
% 
%                 % covariance
%                 K = exp_disL;
%                 K(1:x_HF_number,:) = rho*rho*etaL*K(1:x_HF_number,:)+etaD*exp_disD;
%                 K(x_HF_number+1:end,:) = rho*etaL*K(x_HF_number+1:end,:);
%             end
%         else
%             % initializate square of X inner distance sq/ vari_num
%             sq_dis_v = zeros(x_num,x_num,vari_num);
%             for len_index = 1:vari_num
%                 sq_dis_v(:,:,len_index) = (X(:,len_index)-X(:,len_index)').^2/vari_num;
%             end
% 
%             % exp of x__x with H
%             exp_disD = zeros(x_num);
%             for len_index = 1:vari_num
%                 exp_disD(1:x_HF_number,1:x_HF_number) = exp_disD(1:x_HF_number,1:x_HF_number)+...
%                     sq_dis_v(1:x_HF_number,1:x_HF_number,len_index)*lenD;
%             end
%             exp_disD(1:x_HF_number,1:x_HF_number) = exp(-exp_disD(1:x_HF_number,1:x_HF_number));
%             KD = etaD*exp_disD;
% 
%             % exp of x__x with L
%             exp_disL = zeros(x_num);
%             for len_index = 1:vari_num
%                 exp_disL = exp_disL+...
%                     sq_dis_v(:,:,len_index)*lenL;
%             end
%             exp_disL = exp(-exp_disL);
%             eta_exp_disL = etaL*exp_disL;
% 
%             % times rho: HH to rho2,HL to rho,LL to 1
%             KL = eta_exp_disL;
%             KL(1:x_HF_number,1:x_HF_number) = ...
%                 (rho*rho)*eta_exp_disL(1:x_HF_number,1:x_HF_number);
%             KL(1:x_HF_number,(x_HF_number+1):end) = ...
%                 rho*eta_exp_disL(1:x_HF_number,(x_HF_number+1):end);
%             KL((x_HF_number+1):end,1:x_HF_number) = ...
%                 KL(1:x_HF_number,(x_HF_number+1):end)';
% 
%             K = KL+KD;
% 
%             if nargout >= 2
%                 dK_dcov = cell(1,5);
% 
%                 % len D
%                 dK_dlenD = zeros(x_num);
%                 for len_index = 1:vari_num
%                     dK_dlenD(1:x_HF_number,1:x_HF_number) = dK_dlenD(1:x_HF_number,1:x_HF_number) + ...
%                         sq_dis_v(1:x_HF_number,1:x_HF_number,len_index);
%                 end
%                 dK_dlenD(1:x_HF_number,1:x_HF_number) = -dK_dlenD(1:x_HF_number,1:x_HF_number).*...
%                     KD(1:x_HF_number,1:x_HF_number)*lenD;
%                 dK_dcov{1} = dK_dlenD;
% 
%                 % eta D
%                 dK_dcov{2} = KD;
% 
%                 % len L
%                 dK_dlenL = zeros(x_num,x_num);
%                 for len_index = 1:vari_num
%                     dK_dlenL = dK_dlenL + sq_dis_v(:,:,len_index);
%                 end
%                 dK_dlenL = -dK_dlenL.*KL.*lenL;
%                 dK_dcov{3} = dK_dlenL;
% 
%                 % eta L
%                 dK_dcov{4} = KL;
% 
%                 % rho
%                 dK_drho = zeros(x_num);
%                 dK_drho(1:x_HF_number,1:x_HF_number) = ...
%                     2*rho*rho*eta_exp_disL(1:x_HF_number,1:x_HF_number);
%                 dK_drho(1:x_HF_number,(x_HF_number+1):end) = ...
%                     rho*eta_exp_disL(1:x_HF_number,(x_HF_number+1):end);
%                 dK_drho((x_HF_number+1):end,1:x_HF_number) = ...
%                     dK_drho(1:x_HF_number,(x_HF_number+1):end)';
%                 dK_dcov{5} = dK_drho;
%             end
%         end
% 
%     end
% 
%     function [post,nlZ,dnlZ] = infEP(hyp,mean,cov,lik,x,y)
%         % Expectation Propagation approximation to the posterior Gaussian Process.
%         % The function takes a specified covariance function (see covFunctions.m) and
%         % likelihood function (see likFunctions.m),and is designed to be used with
%         % gp.m. See also infMethods.m. In the EP algorithm,the sites are
%         % updated in random order,for better performance when cases are ordered
%         % according to the targets.
%         %
%         % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch 2013-09-13.
%         %
%         % See also INFMETHODS.M.
%         %
%         persistent last_ttau last_tnu              % keep tilde parameters between calls
%         tol = 1e-4; max_sweep = 10; min_sweep = 2;     % tolerance to stop EP iterations
% 
%         inf = 'infEP';
%         if isnumeric(cov),K = cov;                    % use provided covariance matrix
%         else K = feval(cov{:},hyp.cov,x); end       % evaluate the covariance matrix
% 
%         n = size(x,1);
% 
%         if isnumeric(mean),m = mean;                         % use provided mean vector
%         else m = feval(mean{:},hyp.mean,x); end             % evaluate the mean vector
% 
%         % A note on naming: variables are given short but descriptive names in
%         % accordance with Rasmussen & Williams "GPs for Machine Learning" (2006): mu
%         % and s2 are mean and variance,nu and tau are natural parameters. A leading t
%         % means tilde,a subscript _ni means "not i" (for cavity parameters),or _n
%         % for a vector of cavity parameters. N(f|mu,Sigma) is the posterior.
% 
%         % marginal likelihood for ttau = tnu = zeros(n,1); equals n*log(2) for likCum*
%         nlZ0 = -sum(feval(lik{:},hyp.lik,y,m,diag(K),inf));
%         if any(size(last_ttau) ~= [n 1])      % find starting point for tilde parameters
%             ttau = zeros(n,1); tnu  = zeros(n,1);        % init to zero if no better guess
%             Sigma = K;                     % initialize Sigma and mu,the parameters of ..
%             mu = m; nlZ = nlZ0;                  % .. the Gaussian posterior approximation
%         else
%             ttau = last_ttau; tnu  = last_tnu;   % try the tilde values from previous call
%             [Sigma,mu,L,alpha,nlZ] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,inf);
%             if nlZ > nlZ0                                           % if zero is better ..
%                 ttau = zeros(n,1); tnu  = zeros(n,1);       % .. then init with zero instead
%                 Sigma = K;                   % initialize Sigma and mu,the parameters of ..
%                 mu = m; nlZ = nlZ0;                % .. the Gaussian posterior approximation
%             end
%         end
% 
%         nlZ_old = Inf; sweep = 0;               % converged,max. sweeps or min. sweeps?
%         while (abs(nlZ-nlZ_old) > tol && sweep < max_sweep) || sweep<min_sweep
%             nlZ_old = nlZ; sweep = sweep+1;
%             for i = 1:n       % iterate EP updates (in random order) over examples
% %             for i = randperm(n)       % iterate EP updates (in random order) over examples
%                 tau_ni = 1/Sigma(i,i)-ttau(i);      %  first find the cavity distribution ..
%                 nu_ni = mu(i)/Sigma(i,i)-tnu(i);                % .. params tau_ni and nu_ni
% 
%                 % compute the desired derivatives of the indivdual log partition function
%                 [lZ,dlZ,d2lZ] = feval(lik{:},hyp.lik,y(i),nu_ni/tau_ni,1/tau_ni,inf);
%                 ttau_old = ttau(i); tnu_old = tnu(i);  % find the new tilde params,keep old
%                 ttau(i) = -d2lZ /(1+d2lZ/tau_ni);
%                 ttau(i) = max(ttau(i),0); % enforce positivity i.e. lower bound ttau by zero
%                 tnu(i)  = ( dlZ - nu_ni/tau_ni*d2lZ )/(1+d2lZ/tau_ni);
% 
%                 dtt = ttau(i)-ttau_old; dtn = tnu(i)-tnu_old;      % rank-1 update Sigma ..
%                 si = Sigma(:,i); ci = dtt/(1+dtt*si(i));
%                 Sigma = Sigma - ci*si*si';                         % takes 70% of total time
%                 mu = mu - (ci*(mu(i)+si(i)*dtn)-dtn)*si;               % .. and recompute mu
%             end
%             % recompute since repeated rank-one updates can destroy numerical precision
%             [Sigma,mu,L,alpha,nlZ] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,inf);
%         end
% 
%         if sweep  ==  max_sweep && abs(nlZ-nlZ_old) > tol
%             error('maximum number of sweeps exceeded in function infEP')
%         end
% 
%         last_ttau = ttau; last_tnu = tnu;                       % remember for next call
%         post.alpha = alpha; post.sW = sqrt(ttau); post.L = L;  % return posterior params
% 
%         if nargout>2                                           % do we want derivatives?
%             dnlZ = hyp;                                   % allocate space for derivatives
%             tau_n = 1./diag(Sigma)-ttau;             % compute the log marginal likelihood
%             nu_n  = mu./diag(Sigma)-tnu;                    % vectors of cavity parameters
%             sW = sqrt(ttau);
%             F = alpha*alpha'-repmat(sW,1,n).*(L\(L'\diag(sW)));   % covariance hypers
%             [K,dK] = feval(cov{:},hyp.cov,x,[]);
%             for i = 1:length(hyp.cov)
%                 dnlZ.cov(i) = -sum(sum(F.*dK{i}))/2;
%             end
%             for i = 1:numel(hyp.lik)                                   % likelihood hypers
%                 dlik = feval(lik{:},hyp.lik,y,nu_n./tau_n,1./tau_n,inf,i);
%                 dnlZ.lik(i) = -sum(dlik);
%             end
%             [junk,dlZ] = feval(lik{:},hyp.lik,y,nu_n./tau_n,1./tau_n,inf);% mean hyps
%             for i = 1:numel(hyp.mean)
%                 dm = feval(mean{:},hyp.mean,x,i);
%                 dnlZ.mean(i) = -dlZ'*dm;
%             end
%         end
%     end
% 
%     function [Sigma,mu,L,alpha,nlZ] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,inf)
%         % function to compute the parameters of the Gaussian approximation,Sigma and
%         % mu,and the negative log marginal likelihood,nlZ,from the current site
%         % parameters,ttau and tnu. Also returns L (useful for predictions).
%         %
%         n = length(y);                                      % number of training cases
%         sW = sqrt(ttau);                                        % compute Sigma and mu
%         L = chol(eye(n)+sW*sW'.*K);                            % L'*L = B = eye(n)+sW*K*sW
%         V = L'\(repmat(sW,1,n).*K);
%         Sigma = K - V'*V;
%         alpha = tnu-sW.*(L\(L'\(sW.*(K*tnu+m))));
%         mu = K*alpha+m; v = diag(Sigma);
% 
%         tau_n = 1./diag(Sigma)-ttau;             % compute the log marginal likelihood
%         nu_n  = mu./diag(Sigma)-tnu;                    % vectors of cavity parameters
%         lZ = feval(lik{:},hyp.lik,y,nu_n./tau_n,1./tau_n,inf);
%         p = tnu-m.*ttau; q = nu_n-m.*tau_n;                        % auxiliary vectors
%         nlZ = sum(log(diag(L))) - sum(lZ) - p'*Sigma*p/2 + (v'*p.^2)/2 ...
%             - q'*((ttau./tau_n.*q-2*p).*v)/2 - sum(log(1+ttau./tau_n))/2;
% 
%     end
% 
%     function A = meanConst(hyp,x,i)
%         % Constant mean function. The mean function is parameterized as:
%         %
%         % m(x) = c
%         %
%         % The hyperparameter is:
%         %
%         % hyp = [ c ]
%         %
%         % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2010-08-04.
%         %
%         % See also MEANFUNCTIONS.M.
%         %
%         if nargin<2,A = '1'; return; end             % report number of hyperparameters
%         if numel(hyp)~=1,error('Exactly one hyperparameter needed.'),end
%         c = hyp;
%         if nargin == 2
%             A = c*ones(size(x,1),1);                                       % evaluate mean
%         else
%             if i == 1
%                 A = ones(size(x,1),1);                                          % derivative
%             else
%                 A = zeros(size(x,1),1);
%             end
%         end
%     end
% 
%     function [varargout] = likErf(hyp,y,mu,s2,inf)
%         % likErf - Error function or cumulative Gaussian likelihood function for binary
%         % classification or probit regression. The expression for the likelihood is
%         %   likErf(t) = (1+erf(t/sqrt(2)))/2 = normcdf(t).
%         %
%         % Several modes are provided,for computing likelihoods,derivatives and moments
%         % respectively,see likFunctions.m for the details. In general,care is taken
%         % to avoid numerical issues when the arguments are extreme.
%         %
%         % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2014-03-19.
%         %
%         % See also LIKFUNCTIONS.M.
%         %
%         if nargin<3,varargout = {'0'}; return; end   % report number of hyperparameters
%         if nargin>1,y = sign(y); y(y == 0) = 1; else y = 1; end % allow only +/- 1 values
%         if numel(y) == 0,y = 1; end
% 
%         if nargin<5                              % prediction mode if inf is not present
%             y = y.*ones(size(mu));                                       % make y a vector
%             s2zero = 1; if nargin>3&&numel(s2)>0&&norm(s2)>eps,s2zero = 0; end  % s2 == 0 ?
%             if s2zero                                         % log probability evaluation
%                 lp = logphi(y.*mu);
%             else                                                              % prediction
%                 lp = likErf(hyp,y,mu,s2,'infEP');
%             end
%             p = exp(lp); ymu = {}; ys2 = {};
%             if nargout>1
%                 ymu = 2*p-1;                                                % first y moment
%                 if nargout>2
%                     ys2 = 4*p.*(1-p);                                        % second y moment
%                 end
%             end
%             varargout = {lp,ymu,ys2};
%         else                                                            % inference mode
%             switch inf
%                 case 'infLaplace'
%                     if nargin<6                                             % no derivative mode
%                         f = mu; yf = y.*f;                            % product latents and labels
%                         varargout = cell(nargout,1); [varargout{:}] = logphi(yf);   % query logphi
%                         if nargout>1
%                             varargout{2} = y.*varargout{2};
%                             if nargout>3,varargout{4} = y.*varargout{4}; end
%                         end
%                     else                                                       % derivative mode
%                         varargout = {[],[],[]};                         % derivative w.r.t. hypers
%                     end
% 
%                 case 'infEP'
%                     if nargin<6                                             % no derivative mode
%                         z = mu./sqrt(1+s2); dlZ = {}; d2lZ = {};
%                         if numel(y) > 0,z = z.*y; end
%                         if nargout <= 1,lZ = logphi(z);                         % log part function
%                         else          [lZ,n_p] = logphi(z); end
%                         if nargout > 1
%                             if numel(y) == 0,y = 1; end
%                             dlZ = y.*n_p./sqrt(1+s2);                      % 1st derivative wrt mean
%                             if nargout>2,d2lZ = -n_p.*(z+n_p)./(1+s2); end         % 2nd derivative
%                         end
%                         varargout = {lZ,dlZ,d2lZ};
%                     else                                                       % derivative mode
%                         varargout = {[]};                                     % deriv. wrt hyp.lik
%                     end
%             end
%         end
%     end
% 
%     function [lp,dlp,d2lp,d3lp] = logphi(z)
%         % Safe computation of logphi(z) = log(normcdf(z)) and its derivatives
%         %                    dlogphi(z) = normpdf(x)/normcdf(x).
%         % The function is based on index 5725 in Hart et al. and gsl_sf_log_erfc_e.
%         %
%         % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2013-11-13.
%         %
%         z = real(z);                                 % support for real arguments only
%         lp = zeros(size(z));                                         % allocate memory
%         id1 = z.*z<0.0492;                                 % first case: close to zero
%         lp0 = -z(id1)/sqrt(2*pi);
%         c = [ 0.00048204; -0.00142906; 0.0013200243174; 0.0009461589032;
%             -0.0045563339802; 0.00556964649138; 0.00125993961762116;
%             -0.01621575378835404; 0.02629651521057465; -0.001829764677455021;
%             2*(1-pi/3); (4-pi)/3; 1; 1];
%         f = 0; for i = 1:14,f = lp0.*(c(i)+f); end,lp(id1) = -2*f-log(2);
%         id2 = z<-11.3137;                                    % second case: very small
%         r = [ 1.2753666447299659525; 5.019049726784267463450;
%             6.1602098531096305441; 7.409740605964741794425;
%             2.9788656263939928886 ];
%         q = [ 2.260528520767326969592;  9.3960340162350541504;
%             12.048951927855129036034; 17.081440747466004316;
%             9.608965327192787870698;  3.3690752069827527677 ];
%         num = 0.5641895835477550741; for i = 1:5,num = -z(id2).*num/sqrt(2) + r(i); end
%         den = 1.0;                   for i = 1:6,den = -z(id2).*den/sqrt(2) + q(i); end
%         e = num./den; lp(id2) = log(e/2) - z(id2).^2/2;
%         id3 = ~id2 & ~id1; lp(id3) = log(erfc(-z(id3)/sqrt(2))/2);  % third case: rest
%         if nargout>1                                        % compute first derivative
%             dlp = zeros(size(z));                                      % allocate memory
%             dlp( id2) = abs(den./num) * sqrt(2/pi); % strictly positive first derivative
%             dlp(~id2) = exp(-z(~id2).*z(~id2)/2-lp(~id2))/sqrt(2*pi); % safe computation
%             if nargout>2                                     % compute second derivative
%                 d2lp = -dlp.*abs(z+dlp);             % strictly negative second derivative
%                 if nargout>3                                    % compute third derivative
%                     d3lp = -d2lp.*abs(z+2*dlp)-dlp;     % strictly positive third derivative
%                 end
%             end
%         end
%     end
% 
% end

function [predict_function,CGP_model] = classifyGaussProcess...
    (X,Class,hyp)
% generate gaussian process classifier model
% version 6,this version is assembly of gpml-3.6 EP method
% X is x_number x variable_number matirx,Y is x_number x 1 matrix
% low_bou,up_bou is 1 x variable_number matrix
% only support binary classification,-1 and 1
%
% input:
% X,Class,hyp(mean,cov(len,eta))
%
% abbreviation:
% pred: predicted,nomlz: normalization,num: number
% var: variance
%
[x_number,variable_number] = size(X);
if nargin < 5
    hyp.mean = 0;
    hyp.cov = zeros(1,variable_number+1);
end

% normalization data
aver_X = mean(X);
stdD_X = std(X);
index__ = find(stdD_X == 0);
if  ~isempty(index__),stdD_X(index__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;

object_function = @(x) objectNLLGPC(x,{@infEP},{@meanConst},{@calCov},{@likErf},X_nomlz,Class);
hyp_x = [hyp.mean,hyp.cov];

% [fval,gradient] = object_function(hyp_x)
% [fval_differ,gradient_differ] = differ(object_function,hyp_x)

hyp_low_bou = -3*ones(1,variable_number+2);
hyp_up_bou = 3*ones(1,variable_number+2);
hyp_x = fmincon(object_function,hyp_x,[],[],[],[],hyp_low_bou,hyp_up_bou,[],...
    optimoptions('fmincon','Display','iter-detailed','SpecifyObjectiveGradient',true,...
    'MaxFunctionEvaluations',20,'OptimalityTolerance',1e-6));

hyp.mean = hyp_x(1);
hyp.cov = hyp_x(2:end);
hyp.lik = [];
post = infEP(hyp,{@meanConst},{@calCov},{@likErf},X_nomlz,Class);
predict_function = @(x_pred) classifyGaussPredictor...
    (x_pred,hyp,{@meanConst},{@calCov},{@likErf},post,X_nomlz,aver_X,stdD_X);

% output model
CGP_model.X = X;
CGP_model.Class = Class;
CGP_model.X_nomlz = X_nomlz;
CGP_model.aver_X = aver_X;
CGP_model.stdD_X = stdD_X;
CGP_model.predict_function = predict_function;
CGP_model.hyp = hyp;

    function [fval,gradient] = objectNLLGPC(x,inf,mean,cov,lik,X,Y)
        hyp_iter.mean = x(1);
        hyp_iter.cov = x(2:end);
        hyp_iter.lik = [];

        if nargout < 2
            [~,nlZ] = feval(inf{:},hyp_iter,mean,cov,lik,X,Y);
            fval = nlZ;
        elseif nargout < 3
            [~,nlZ,dnlZ] = feval(inf{:},hyp_iter,mean,cov,lik,X,Y);
            fval = nlZ;
            gradient = [dnlZ.mean,dnlZ.cov];
        end
    end

    function [class,possibility,miu_pre,var_pre] = classifyGaussPredictor...
            (X_pred,hyp,mean,cov,lik,post,X,aver_X,stdD_X)
        % predict function
        %
        X_pred_nomlz = (X_pred-aver_X)./stdD_X;
        pred_num = size(X_pred_nomlz,1);
        ys = ones(pred_num,1);

        alpha = post.alpha; L = post.L; sW = post.sW;
        nz = true(size(alpha,1),1);               % non-sparse representation
        %verify whether L contains valid Cholesky decomposition or something different
        Lchol = isnumeric(L) && all(all(tril(L,-1)==0)&diag(L)'>0&isreal(diag(L))');
        ns = size(X_pred_nomlz,1);                                       % number of data points
        nperbatch = 1000;                       % number of data points per mini batch
        nact = 0;                       % number of already processed test data points
        ymu = zeros(ns,1); ys2 = ymu; miu_pre = ymu; var_pre = ymu; possibility = ymu;   % allocate mem
        while nact<ns               % process minibatches of test cases to save memory
            id = (nact+1):min(nact+nperbatch,ns);               % data points to process
            kss = feval(cov{:},hyp.cov,X_pred_nomlz(id,:),'diag');              % self-variance
            Ks = feval(cov{:},hyp.cov,X(nz,:),X_pred_nomlz(id,:));        % avoid computation
            ms = feval(mean{:},hyp.mean,X_pred_nomlz(id,:));
            N = size(alpha,2);  % number of alphas (usually 1; more in case of sampling)
            Fmu = repmat(ms,1,N) + Ks'*full(alpha(nz,:));        % conditional mean fs|f
            miu_pre(id) = sum(Fmu,2)/N;                                   % predictive means
            if Lchol    % L contains chol decomp => use Cholesky parameters (alpha,sW,L)
                V  = L'\(repmat(sW,1,length(id)).*Ks);
                var_pre(id) = kss - sum(V.*V,1)';                       % predictive variances
            else                % L is not triangular => use alternative parametrisation
                if isnumeric(L),LKs = L*Ks; else LKs = L(Ks); end    % matrix or callback
                var_pre(id) = kss + sum(Ks.*LKs,1)';                    % predictive variances
            end
            var_pre(id) = max(var_pre(id),0);   % remove numerical noise i.e. negative variances
            Fs2 = repmat(var_pre(id),1,N);     % we have multiple values in case of sampling
            if nargin<9
                [Lp,Ymu,Ys2] = feval(lik{:},hyp.lik,[],Fmu(:),Fs2(:));
            else
                Ys = repmat(ys(id),1,N);
                [Lp,Ymu,Ys2] = feval(lik{:},hyp.lik,Ys(:),Fmu(:),Fs2(:));
            end
            possibility(id)  = sum(reshape(Lp,[],N),2)/N;    % log probability; sample averaging
            ymu(id) = sum(reshape(Ymu,[],N),2)/N;          % predictive mean ys|y and ..
            ys2(id) = sum(reshape(Ys2,[],N),2)/N;                          % .. variance
            nact = id(end);          % set counter to index of last processed data point
        end

        possibility = exp(possibility);
        class = ones(pred_num,1);
        index_list = find(possibility < 0.5);
        class(index_list) = -1;
    end

    function [K,dK_dcov] = calCov(cov,X,Z)
        % obtain covariance of x
        % cov: eta,len(equal to 1/len.^2)
        %
        % k = eta*exp(-sum(x_dis*len)/vari_num);
        %
        [x_num,vari_num] = size(X);

        len = exp(cov(1:vari_num));
        eta = exp(cov(vari_num+1));

        % predict
        if nargin > 2 && nargout < 2 && ~isempty(Z)
            if strcmp(Z,'diag')
                K = eta;
            else
                [z_number,vari_num] = size(Z);
                % initializate square of X inner distance/ vari_num
                K = zeros(x_num,z_number);
                for len_index = 1:vari_num
                    K = K+(X(:,len_index)-Z(:,len_index)').^2*len(len_index)/vari_num;
                end
                K = eta*exp(-K);
            end
        else
            % initializate square of X inner distance sq
            sq_dis_v = zeros(x_num,x_num,vari_num);
            for len_index = 1:vari_num
                sq_dis_v(:,:,len_index) = (X(:,len_index)-X(:,len_index)').^2/vari_num;
            end

            % exp of x__x with theta
            exp_dis = zeros(x_num);
            for len_index = 1:vari_num
                exp_dis = exp_dis+sq_dis_v(:,:,len_index)*len(len_index);
            end
            exp_dis = exp(-exp_dis);
            K = exp_dis*eta;

            if nargout >= 2
                dK_dcov = cell(1,vari_num+1);
                for len_index = 1:vari_num
                    dK_dcov{len_index} = -K.*sq_dis_v(:,:,len_index)*len(len_index);
                end

                dK_dcov{vari_num+1} = K;
            end
        end

    end

    function [post nlZ dnlZ] = infEP(hyp,mean,cov,lik,x,y)
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
        tol = 1e-4; max_sweep = 10; min_sweep = 2;     % tolerance to stop EP iterations

        inf = 'infEP';
        n = size(x,1);
        if isnumeric(cov),K = cov;                    % use provided covariance matrix
        else K = feval(cov{:},hyp.cov,x); end       % evaluate the covariance matrix
        if isnumeric(mean),m = mean;                         % use provided mean vector
        else m = feval(mean{:},hyp.mean,x); end             % evaluate the mean vector

        % A note on naming: variables are given short but descriptive names in
        % accordance with Rasmussen & Williams "GPs for Machine Learning" (2006): mu
        % and s2 are mean and variance,nu and tau are natural parameters. A leading t
        % means tilde,a subscript _ni means "not i" (for cavity parameters),or _n
        % for a vector of cavity parameters. N(f|mu,Sigma) is the posterior.

        % marginal likelihood for ttau = tnu = zeros(n,1); equals n*log(2) for likCum*
        nlZ0 = -sum(feval(lik{:},hyp.lik,y,m,diag(K),inf));
        if any(size(last_ttau) ~= [n 1])      % find starting point for tilde parameters
            ttau = zeros(n,1); tnu  = zeros(n,1);        % init to zero if no better guess
            Sigma = K;                     % initialize Sigma and mu,the parameters of ..
            mu = m; nlZ = nlZ0;                  % .. the Gaussian posterior approximation
        else
            ttau = last_ttau; tnu  = last_tnu;   % try the tilde values from previous call
            [Sigma,mu,L,alpha,nlZ] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,inf);
            if nlZ > nlZ0                                           % if zero is better ..
                ttau = zeros(n,1); tnu  = zeros(n,1);       % .. then init with zero instead
                Sigma = K;                   % initialize Sigma and mu,the parameters of ..
                mu = m; nlZ = nlZ0;                % .. the Gaussian posterior approximation
            end
        end

        nlZ_old = Inf; sweep = 0;               % converged,max. sweeps or min. sweeps?
        while (abs(nlZ-nlZ_old) > tol && sweep < max_sweep) || sweep<min_sweep
            nlZ_old = nlZ; sweep = sweep+1;
            for i = randperm(n)       % iterate EP updates (in random order) over examples
                tau_ni = 1/Sigma(i,i)-ttau(i);      %  first find the cavity distribution ..
                nu_ni = mu(i)/Sigma(i,i)-tnu(i);                % .. params tau_ni and nu_ni

                % compute the desired derivatives of the indivdual log partition function
                [lZ,dlZ,d2lZ] = feval(lik{:},hyp.lik,y(i),nu_ni/tau_ni,1/tau_ni,inf);
                ttau_old = ttau(i); tnu_old = tnu(i);  % find the new tilde params,keep old
                ttau(i) =                     -d2lZ  /(1+d2lZ/tau_ni);
                ttau(i) = max(ttau(i),0); % enforce positivity i.e. lower bound ttau by zero
                tnu(i)  = ( dlZ - nu_ni/tau_ni*d2lZ )/(1+d2lZ/tau_ni);

                dtt = ttau(i)-ttau_old; dtn = tnu(i)-tnu_old;      % rank-1 update Sigma ..
                si = Sigma(:,i); ci = dtt/(1+dtt*si(i));
                Sigma = Sigma - ci*si*si';                         % takes 70% of total time
                mu = mu - (ci*(mu(i)+si(i)*dtn)-dtn)*si;               % .. and recompute mu
            end
            % recompute since repeated rank-one updates can destroy numerical precision
            [Sigma,mu,L,alpha,nlZ] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,inf);
        end

        if sweep == max_sweep && abs(nlZ-nlZ_old) > tol
            error('maximum number of sweeps exceeded in function infEP')
        end

        last_ttau = ttau; last_tnu = tnu;                       % remember for next call
        post.alpha = alpha; post.sW = sqrt(ttau); post.L = L;  % return posterior params

        if nargout>2                                           % do we want derivatives?
            dnlZ = hyp;                                   % allocate space for derivatives
            tau_n = 1./diag(Sigma)-ttau;             % compute the log marginal likelihood
            nu_n  = mu./diag(Sigma)-tnu;                    % vectors of cavity parameters
            sW = sqrt(ttau);
            F = alpha*alpha'-repmat(sW,1,n).*(L\(L'\diag(sW)));   % covariance hypers
            [K,dK] = feval(cov{:},hyp.cov,x,[]);
            for i = 1:length(hyp.cov)
                dnlZ.cov(i) = -sum(sum(F.*dK{i}))/2;
            end
            for i = 1:numel(hyp.lik)                                   % likelihood hypers
                dlik = feval(lik{:},hyp.lik,y,nu_n./tau_n,1./tau_n,inf,i);
                dnlZ.lik(i) = -sum(dlik);
            end
            [junk,dlZ] = feval(lik{:},hyp.lik,y,nu_n./tau_n,1./tau_n,inf);% mean hyps
            for i = 1:numel(hyp.mean)
                dm = feval(mean{:},hyp.mean,x,i);
                dnlZ.mean(i) = -dlZ'*dm;
            end
        end
    end

    function [Sigma,mu,L,alpha,nlZ] = epComputeParams(K,y,ttau,tnu,lik,hyp,m,inf)
        % function to compute the parameters of the Gaussian approximation,Sigma and
        % mu,and the negative log marginal likelihood,nlZ,from the current site
        % parameters,ttau and tnu. Also returns L (useful for predictions).
        %
        n = length(y);                                      % number of training cases
        sW = sqrt(ttau);                                        % compute Sigma and mu
        L = chol(eye(n)+sW*sW'.*K);                            % L'*L = B = eye(n)+sW*K*sW
        V = L'\(repmat(sW,1,n).*K);
        Sigma = K - V'*V;
        alpha = tnu-sW.*(L\(L'\(sW.*(K*tnu+m))));
        mu = K*alpha+m; v = diag(Sigma);

        tau_n = 1./diag(Sigma)-ttau;             % compute the log marginal likelihood
        nu_n  = mu./diag(Sigma)-tnu;                    % vectors of cavity parameters
        lZ = feval(lik{:},hyp.lik,y,nu_n./tau_n,1./tau_n,inf);
        p = tnu-m.*ttau; q = nu_n-m.*tau_n;                        % auxiliary vectors
        nlZ = sum(log(diag(L))) - sum(lZ) - p'*Sigma*p/2 + (v'*p.^2)/2 ...
            - q'*((ttau./tau_n.*q-2*p).*v)/2 - sum(log(1+ttau./tau_n))/2;
    end

    function A = meanConst(hyp,x,i)

        % Constant mean function. The mean function is parameterized as:
        %
        % m(x) = c
        %
        % The hyperparameter is:
        %
        % hyp = [ c ]
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2010-08-04.
        %
        % See also MEANFUNCTIONS.M.

        if nargin<2,A = '1'; return; end             % report number of hyperparameters
        if numel(hyp)~=1,error('Exactly one hyperparameter needed.'),end
        c = hyp;
        if nargin==2
            A = c*ones(size(x,1),1);                                       % evaluate mean
        else
            if i==1
                A = ones(size(x,1),1);                                          % derivative
            else
                A = zeros(size(x,1),1);
            end
        end
    end

    function [varargout] = likErf(hyp,y,mu,s2,inf,i)
        % likErf - Error function or cumulative Gaussian likelihood function for binary
        % classification or probit regression. The expression for the likelihood is
        %   likErf(t) = (1+erf(t/sqrt(2)))/2 = normcdf(t).
        %
        % Several modes are provided,for computing likelihoods,derivatives and moments
        % respectively,see likFunctions.m for the details. In general,care is taken
        % to avoid numerical issues when the arguments are extreme.
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2014-03-19.
        %
        % See also LIKFUNCTIONS.M.
        %
        if nargin<3,varargout = {'0'}; return; end   % report number of hyperparameters
        if nargin>1,y = sign(y); y(y==0) = 1; else y = 1; end % allow only +/- 1 values
        if numel(y)==0,y = 1; end

        if nargin<5                              % prediction mode if inf is not present
            y = y.*ones(size(mu));                                       % make y a vector
            s2zero = 1; if nargin>3&&numel(s2)>0&&norm(s2)>eps,s2zero = 0; end  % s2==0 ?
            if s2zero                                         % log probability evaluation
                lp = logphi(y.*mu);
            else                                                              % prediction
                lp = likErf(hyp,y,mu,s2,'infEP');
            end
            p = exp(lp); ymu = {}; ys2 = {};
            if nargout>1
                ymu = 2*p-1;                                                % first y moment
                if nargout>2
                    ys2 = 4*p.*(1-p);                                        % second y moment
                end
            end
            varargout = {lp,ymu,ys2};
        else                                                            % inference mode
            switch inf
                case 'infLaplace'
                    if nargin<6                                             % no derivative mode
                        f = mu; yf = y.*f;                            % product latents and labels
                        varargout = cell(nargout,1); [varargout{:}] = logphi(yf);   % query logphi
                        if nargout>1
                            varargout{2} = y.*varargout{2};
                            if nargout>3,varargout{4} = y.*varargout{4}; end
                        end
                    else                                                       % derivative mode
                        varargout = {[],[],[]};                         % derivative w.r.t. hypers
                    end

                case 'infEP'
                    if nargin<6                                             % no derivative mode
                        z = mu./sqrt(1+s2); dlZ = {}; d2lZ = {};
                        if numel(y)>0,z = z.*y; end
                        if nargout <= 1,lZ = logphi(z);                         % log part function
                        else          [lZ,n_p] = logphi(z); end
                        if nargout > 1
                            if numel(y)==0,y = 1; end
                            dlZ = y.*n_p./sqrt(1+s2);                      % 1st derivative wrt mean
                            if nargout>2,d2lZ = -n_p.*(z+n_p)./(1+s2); end         % 2nd derivative
                        end
                        varargout = {lZ,dlZ,d2lZ};
                    else                                                       % derivative mode
                        varargout = {[]};                                     % deriv. wrt hyp.lik
                    end
            end
        end
    end

    function [lp,dlp,d2lp,d3lp] = logphi(z)
        % Safe computation of logphi(z) = log(normcdf(z)) and its derivatives
        %                    dlogphi(z) = normpdf(x)/normcdf(x).
        % The function is based on index 5725 in Hart et al. and gsl_sf_log_erfc_e.
        %
        % Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch,2013-11-13.
        %
        z = real(z);                                 % support for real arguments only
        lp = zeros(size(z));                                         % allocate memory
        id1 = z.*z<0.0492;                                 % first case: close to zero
        lp0 = -z(id1)/sqrt(2*pi);
        c = [ 0.00048204; -0.00142906; 0.0013200243174; 0.0009461589032;
            -0.0045563339802; 0.00556964649138; 0.00125993961762116;
            -0.01621575378835404; 0.02629651521057465; -0.001829764677455021;
            2*(1-pi/3); (4-pi)/3; 1; 1];
        f = 0; for i = 1:14,f = lp0.*(c(i)+f); end,lp(id1) = -2*f-log(2);
        id2 = z<-11.3137;                                    % second case: very small
        r = [ 1.2753666447299659525; 5.019049726784267463450;
            6.1602098531096305441; 7.409740605964741794425;
            2.9788656263939928886 ];
        q = [ 2.260528520767326969592;  9.3960340162350541504;
            12.048951927855129036034; 17.081440747466004316;
            9.608965327192787870698;  3.3690752069827527677 ];
        num = 0.5641895835477550741; for i = 1:5,num = -z(id2).*num/sqrt(2) + r(i); end
        den = 1.0;                   for i = 1:6,den = -z(id2).*den/sqrt(2) + q(i); end
        e = num./den; lp(id2) = log(e/2) - z(id2).^2/2;
        id3 = ~id2 & ~id1; lp(id3) = log(erfc(-z(id3)/sqrt(2))/2);  % third case: rest
        if nargout>1                                        % compute first derivative
            dlp = zeros(size(z));                                      % allocate memory
            dlp( id2) = abs(den./num) * sqrt(2/pi); % strictly positive first derivative
            dlp(~id2) = exp(-z(~id2).*z(~id2)/2-lp(~id2))/sqrt(2*pi); % safe computation
            if nargout>2                                     % compute second derivative
                d2lp = -dlp.*abs(z+dlp);             % strictly negative second derivative
                if nargout>3                                    % compute third derivative
                    d3lp = -d2lp.*abs(z+2*dlp)-dlp;     % strictly positive third derivative
                end
            end
        end
    end

end


function fval = LFFcn(x)
if 0.45+sin(2.2*pi*x(1))/2.5-x(2) > 0
    fval = 1;
else
    fval = -1;
end
end
function fval = HFFcn(x)
if 0.5+sin(2.5*pi*x(1))/3-x(2) > 0
    fval = 1;
else
    fval = -1;
end
end
