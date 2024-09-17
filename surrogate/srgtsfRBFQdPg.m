function srgt=srgtsfRBFQdPg(X,Y)
% generate ensemle radial basis function surrogate model
% input data will be normalize by average and standard deviation of data
% using quadratic programming to calculate weigth of each sub model
%
% input:
% X (matrix): trained X, x_num x vari_num
% Y (vector): trained Y, x_num x 1
%
% output:
% srgt(a ensemle radial basis function model)
%
% reference:
% [1] Shi R, Liu L, Long T, et al. An Efficient Ensemble of Radial Basis
% Functions Method Based on Quadratic Programming[J]. Engineering
% Optimization, 2016, 48: 1202-1225.
%
% Copyright 2023.2 Adel
%

% normalize data
[x_num,vari_num]=size(X);
aver_X=mean(X);
stdD_X=std(X);stdD_X(stdD_X == 0)=1;
aver_Y=mean(Y);
stdD_Y=std(Y);stdD_Y(stdD_Y == 0)=1;
X_norm=(X-aver_X)./stdD_X;
Y_norm=(Y-aver_Y)./stdD_Y;

% initialization distance of all X
r=zeros(x_num,x_num);
for vari_idx=1:vari_num
    r=r+(X_norm(:,vari_idx)-X_norm(:,vari_idx)').^2;
end
r=sqrt(r);

% linr kernal function
basis_fcn_linr=@(r,c) r+c;
obj_fcn=@(c) probNLL(r,Y_norm,x_num,basis_fcn_linr,c);
[c_linr,~,~,~]=fminbnd(obj_fcn,-1e2,1e2,optimset('Display','none'));

% gauss kernal function
basis_fcn_gauss=@(r,c) exp(-c*r.^2);
obj_fcn=@(c) probNLL(r,Y_norm,x_num,basis_fcn_gauss,c);
[c_gauss,~,~,~]=fminbnd(obj_fcn,1e-2,1e2,optimset('Display','none'));

% spin kernal function
basis_fcn_spin=@(r,c) r.^2.*log(r.^2*c+1e-3);
obj_fcn=@(c) probNLL(r,Y_norm,x_num,basis_fcn_spin,c);
[c_spin,~,~,~]=fminbnd(obj_fcn,1e-2,1e2,optimset('Display','none'));

% cubic kernal function
basis_fcn_cubic=@(r,c) (r+c).^3;
obj_fcn=@(c) probNLL(r,Y_norm,x_num,basis_fcn_cubic,c);
[c_cubic,~,~,~]=fminbnd(obj_fcn,1e-2,1e2,optimset('Display','none'));

% multiquadric kernal function
basis_fcn_mtqd=@(r,c) sqrt(r+c);
obj_fcn=@(c) probNLL(r,Y_norm,x_num,basis_fcn_mtqd,c);
[c_mtqd,~,~,~]=fminbnd(obj_fcn,1e-2,1e2,optimset('Display','none'));

% inverse multiquadric kernal function
basis_fcn_ivmq=@(r,c) 1./sqrt(r+c);
obj_fcn=@(c) probNLL(r,Y_norm,x_num,basis_fcn_ivmq,c);
[c_ivmq,~,~,~]=fminbnd(obj_fcn,1e-2,1e2,optimset('Display','none'));

% generate total model
basis_fcn_list={
    @(r) basis_fcn_linr(r,c_linr);
    @(r) basis_fcn_gauss(r,c_gauss);
    @(r) basis_fcn_spin(r,c_spin);
    @(r) basis_fcn_cubic(r,c_cubic);
    @(r) basis_fcn_mtqd(r,c_mtqd);
    @(r) basis_fcn_ivmq(r,c_ivmq);};
c_list=[c_linr;c_gauss;c_spin;c_cubic;c_mtqd;c_ivmq];

mdl_num=size(basis_fcn_list,1);
beta_list=zeros(x_num,mdl_num);
G_list=zeros(x_num,x_num,mdl_num);
invG_list=zeros(x_num,x_num,mdl_num);

% calculate model matrix and error
loo_err_list=zeros(x_num,mdl_num);
for mdl_idx=1:mdl_num
    basis_fcn=basis_fcn_list{mdl_idx};
    [beta,G,invG]=calRBF(r,Y_norm,basis_fcn,x_num);
    beta_list(:,mdl_idx)=beta;
    G_list(:,:,mdl_idx)=G;
    invG_list(:,:,mdl_idx)=invG;

    loo_err_list(:,mdl_idx)=(beta./diag(invG));
end

% calculate weight of each model
C=loo_err_list'*loo_err_list;
eta=trace(C)/x_num;
I_mdl=eye(mdl_num);
i_mdl=ones(mdl_num,1);
weight=(C+eta*I_mdl)\i_mdl/(i_mdl'*((C+eta*I_mdl)\i_mdl));
while min(weight) < -0.05
    % minimal weight cannot less than zero too much
    eta=eta*10;
    weight=(C+eta*I_mdl)\i_mdl/(i_mdl'*((C+eta*I_mdl)\i_mdl));
end

% initialization predict function
pred_fcn=@(X_pred) predictRBFQdPg(X_pred,mdl_num,beta_list,basis_fcn_list,weight);

srgt.X=X;
srgt.Y=Y;
% srgt.basis_fcn_list=basis_fcn_list;
srgt.c_list=c_list;
srgt.beta_list=beta_list;
srgt.gram_list=G_list;
srgt.inv_gram_list=invG_list;
srgt.loo_err_list=loo_err_list;
srgt.weight=weight;
srgt.predict=pred_fcn;
srgt.loo_err=calLOOErr();

%% basical function

    function fval=probNLL(r,Y,x_num,basis_fcn,c)
        % MSE_CV function, simple approximation to RMS
        % basis_fcn input is c and x_sq
        %
        basis_fcn=@(r) basis_fcn(r,c);
        [beta,~,invG]=calRBF(r,Y,basis_fcn,x_num);

        loo_err=beta./diag(invG);
        fval=sum(loo_err.^2);
    end

    function [beta,G,invG]=calRBF(r,Y,basis_fcn,x_num)
        % interp polynomial responed surface core function
        % calculation beta
        %
        % Copyright 2022 Adel
        %
        G=basis_fcn(r)+eye(x_num)*(1000+x_num)*eps;

        % solve beta
        invG=G\eye(x_num);
        beta=invG*Y;
    end

    function Y_pred=predictRBFQdPg(X_pred,model_num,beta_list,basis_fcn_list,w)
        % ensemle radial basis function interpolation predict function
        %
        [x_pred_num,~]=size(X_pred);

        % normalize data
        X_pred_norm=(X_pred-aver_X)./stdD_X;

        % calculate distance
        r_pred=zeros(x_pred_num,x_num);
        for vari_index=1:vari_num
            r_pred=r_pred+(X_pred_norm(:,vari_index)-X_norm(:,vari_index)').^2;
        end
        r_pred=sqrt(r_pred);

        % calculate each sub model predict fval and get predict_y
        y_pred_norm_list=zeros(x_pred_num,model_num);
        for model_idx=1:model_num
            y_pred_norm_list(:,model_idx)=basis_fcn_list{model_idx}(r_pred)*beta_list(:,model_idx);
        end
        Y_pred_norm=y_pred_norm_list*w;

        % normalize data
        Y_pred=Y_pred_norm*stdD_Y+aver_Y;
    end

    function loo_err=calLOOErr()
        % analysis method to quickly calculate R^2 of RBF surrogate model
        %
        % reference:
        % [1] Rippa S. An Algorithm for Selecting a Good Value for the
        % Parameter c in Radial Basis Function Interpolation [J]. Advances
        % in Computational Mathematics, 1999, 11(2-3): 193-210.
        %
        loo_err=sum(loo_err_list*stdD_Y.*weight',2);
    end
end
