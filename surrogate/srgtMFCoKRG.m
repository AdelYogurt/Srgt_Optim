function model_CoKRG=srgtmfCoKRG(X_list,Y_list,option)
% generate Multi-Level Co-Kriging surrogate model
% input data will be normalize by average and standard deviation of data
%
% input:
% X_list (cell): x_num x vari_num matrix, the lower, the higher fidelity
% Y_list (cell): x_num x 1 vector, the lower, the higher fidelity
% model_option (struct): optional, include: optimize_hyp, simplify_hyp, hyp, optimize_option
%
% output:
% srgt(struct): a Multi-Level Co-Kriging model
%
% reference:
% [1] Forrester A I J, Sóbester A, Keane A J. Multi-Fidelity Optimization
% via Surrogate Modelling[J]. Proceedings of the Royal Society A:
% Mathematical, Physical and Engineering Sciences, 2007, 463: 3251-3269.
%
% Copyright 2023.2 Adel
%
if nargin < 3,option=struct();end

% Kriging option
if isempty(option), option=struct();end
if ~isfield(option,'optimize_hyp'), option.('optimize_hyp')=true;end
if ~isfield(option,'simplify_hyp'), option.('simplify_hyp')=true;end
if option.('simplify_hyp'),FLAG_GRAD=false;else,FLAG_GRAD=true;end
if ~isfield(option,'model_option')
    option.('optimize_option')=optimoptions...
        ('fminunc','Display','none',...
        'OptimalityTolerance',1e-6,...
        'FiniteDifferenceStepSize',1e-5,...
        'MaxIterations',20,'SpecifyObjectiveGradient',FLAG_GRAD);
end

% Co-Kriging option
if ~isfield(option,'hyp_list'), option.('hyp_list')={};end
if ~isfield(option,'reg_fcn_list'), option.('reg_fcn_list')={};end

% load exist data
fid_num=length(X_list);
hyp_list=option.('hyp_list');
hyp_list=[hyp_list;repmat({[]},fid_num-length(hyp_list),1)];
reg_fcn_list=option.('reg_fcn_list');
reg_fcn_list=[reg_fcn_list;repmat({[]},fid_num-length(reg_fcn_list),1)];
predict_list=cell(fid_num,1);
sigma_sq_list=zeros(fid_num,1);

x_idx_list=zeros(fid_num,2);
h_idx_list=zeros(fid_num,2);

% load all data and normalize
fid_idx=1;
XALL=X_list{fid_idx};YALL=Y_list{fid_idx};
x_idx_list(fid_idx,1)=1;
x_idx_list(fid_idx,2)=length(Y_list{fid_idx});
fid_idx=fid_idx+1;
while fid_idx <= fid_num
    XALL=[XALL;X_list{fid_idx}];
    YALL=[YALL;Y_list{fid_idx}];
    x_idx_list(fid_idx,1)=x_idx_list(fid_idx-1,2)+1;
    x_idx_list(fid_idx,2)=x_idx_list(fid_idx-1,2)+length(Y_list{fid_idx});
    fid_idx=fid_idx+1;
end
[x_total_num,vari_num]=size(XALL);
aver_X=mean(XALL);
stdD_X=std(XALL);stdD_X(stdD_X == 0)=1;
aver_Y=mean(YALL);
stdD_Y=std(YALL);stdD_Y(stdD_Y == 0)=1;
XALL_norm=(XALL-aver_X)./stdD_X;
YALL_norm=(YALL-aver_Y)./stdD_Y;

% all initial dr_sq
dr_sq_all=zeros(x_total_num,x_total_num,vari_num);
for vari_idx=1:vari_num
    dr_sq_all(:,:,vari_idx)=(XALL_norm(:,vari_idx)-XALL_norm(:,vari_idx)').^2;
end
rho_list=zeros(fid_num,1);

simplify_hyp=option.('simplify_hyp');
pred_fcn=@(X_pred) zeros(size(X_pred,1),1);

% generate all fidelity Kriging
for fid_idx=1:fid_num
    % load fidelity data
    x_idx=x_idx_list(fid_idx,:);
    dr_sq=dr_sq_all(x_idx(1):x_idx(2),x_idx(1):x_idx(2),:);
    hyp=hyp_list{fid_idx};
    reg_fcn=reg_fcn_list{fid_idx};
    X=X_list{fid_idx};x_num=size(X,1);

    % regression function define
    if isempty(reg_fcn)
        if x_num < vari_num,reg_fcn=@(X) ones(size(X,1),1).*stdD_Y+aver_Y; % constant
        else,reg_fcn=@(X) [ones(size(X,1),1),X-aver_X].*stdD_Y+aver_Y;end % linear
    end
    reg_fcn_list{fid_idx}=reg_fcn;
    H_norm=(reg_fcn(X)-aver_Y)./stdD_Y;
    if fid_idx == 1
        h_idx_list(fid_idx,1)=1;
        h_idx_list(fid_idx,2)=size(H_norm,2);
    else
        h_idx_list(fid_idx,1)=h_idx_list(fid_idx-1,2)+1;
        h_idx_list(fid_idx,2)=h_idx_list(fid_idx-1,2)+size(H_norm,2);
    end

    % hyperparameter define
    if fid_idx == 1, rho_init=0;
    else, rho_init=mean(Y_list{fid_idx})/mean(Y_list{fid_idx-1});end
    if isempty(hyp), hyp=[zeros(1,vari_num),rho_init];end

    % predict high fidelity point by low fidelity model
    Y_pred_norm=(pred_fcn(X)-aver_Y)./stdD_Y;

    Y_norm=YALL_norm(x_idx(1):x_idx(2));
    % optimal to get hyperparameter
    % if optimize hyperparameter
    if option.optimize_hyp
        obj_fcn_hyp=@(hyp) probNLLBiasKRG(dr_sq,Y_norm,Y_pred_norm,x_num,vari_num,hyp,H_norm);

        if simplify_hyp
            hyp=[mean(hyp(1:end-1)),hyp(end)];
            low_bou_hyp=[-4,-10];
            up_bou_hyp=[4,10];
        else
            low_bou_hyp=[-4*ones(1,vari_num),-10];
            up_bou_hyp=[4*ones(1,vari_num),10];
        end

        % [fval,gradient]=obj_fcn_hyp(hyp)
        % [~,gradient_differ]=differ(obj_fcn_hyp,hyp)
        % drawFcn(obj_fcn_hyp,low_bou_hyp,up_bou_hyp);

        [hyp,~,~,~]=fminunc(obj_fcn_hyp,hyp,option.('optimize_option'));
        hyp=min(hyp,up_bou_hyp);hyp=max(hyp,low_bou_hyp);

        if simplify_hyp, hyp=[hyp(1)*ones(1,vari_num),hyp(2)];end
    end

    % get bias kriging parameter
    D_norm=Y_norm-hyp(end)*Y_pred_norm;
    [cov,~,~,sigma_sq,~,~,~]=calCovKRG(dr_sq,D_norm,x_num,vari_num,exp(hyp(1:vari_num)),H_norm);
    sigma_sq=sigma_sq*stdD_Y^2; % renormalize data
    hyp_list{fid_idx}=hyp;
    sigma_sq_list(fid_idx)=sigma_sq;
    rho_list(fid_idx)=hyp(end);

    % calculate accumulate covariance
    Y_cum_norm=YALL_norm(1:x_idx(2));
    if fid_idx == 1
        cov_cum=cov*sigma_sq;
        H_cum_norm=H_norm;
    else
        % extend covariance
        x_cum_num=size(cov_cum,1);
        cov_cum=[cov_cum,zeros(x_cum_num,x_num);
            zeros(x_num,x_cum_num),zeros(x_num,x_num)];

        % calculate each block of covariance
        l=fid_idx;
        for fid_jdx=1:fid_idx-1
            x_jdx=x_idx_list(fid_jdx,:);
            cov_block=zeros(x_num,x_jdx(2)-x_jdx(1)+1);
            k=fid_jdx;
            % calculate each item of block
            for fid_kdx=1:fid_jdx
                % calculate covariance
                bi=k-fid_kdx+1;theta_bi=exp(hyp_list{bi});
                cov_fid=zeros(x_num,x_jdx(2)-x_jdx(1)+1);
                for vari_idx=1:vari_num
                    cov_fid=cov_fid+dr_sq_all(x_idx(1):x_idx(2),x_jdx(1):x_jdx(2),vari_idx)*theta_bi(vari_idx);
                end
                cov_fid=exp(-cov_fid/vari_num^2)*sigma_sq_list(bi);

                rho_prod_k=1;bi=k-fid_kdx+1;
                if bi < k,rho_prod_k=prod(rho_list(bi+1:k));end
                rho_prod_l=prod(rho_list(bi+1:l));
                cov_block=cov_block+rho_prod_k*rho_prod_l*cov_fid;
            end
            cov_cum(x_idx(1):x_idx(2),x_jdx(1):x_jdx(2))=cov_block;
            cov_cum(x_jdx(1):x_jdx(2),x_idx(1):x_idx(2))=cov_block';
        end

        % calculate center block
        x_jdx=x_idx_list(fid_idx,:);
        cov_block=zeros(x_num,x_num);
        k=fid_idx;
        % calculate each item of block
        for fid_kdx=1:fid_idx-1
            % calculate covariance
            bi=k-fid_kdx;theta_bi=exp(hyp_list{bi});
            cov_fid=zeros(x_num,x_jdx(2)-x_jdx(1)+1);
            for vari_idx=1:vari_num
                cov_fid=cov_fid+dr_sq_all(x_idx(1):x_idx(2),x_jdx(1):x_jdx(2),vari_idx)*theta_bi(vari_idx);
            end
            cov_fid=exp(-cov_fid/vari_num^2)*sigma_sq_list(bi);

            rho_prod_k=1;bi=k-fid_kdx;
            if bi < k,rho_prod_k=prod(rho_list(bi+1:k));end
            cov_block=cov_block+(rho_prod_k)^2*cov_fid;
        end
        cov_block=cov_block+cov*sigma_sq;
        cov_cum(x_idx(1):x_idx(2),x_idx(1):x_idx(2))=cov_block;

        % extend H
        reg_num=size(H_norm,2);reg_cum_num=size(H_cum_norm,2);
        H_cum_norm=[H_cum_norm,zeros(x_cum_num,reg_num);
            zeros(x_num,reg_cum_num),zeros(x_num,reg_num)];

        for fid_jdx=1:fid_idx-1
            f_jdx=h_idx_list(fid_jdx,:);
            reg_fcn_j=reg_fcn_list{fid_jdx};
            H_norm_fid=prod(rho_list(fid_jdx+1:fid_idx))*(reg_fcn_j(X)-aver_Y)./stdD_Y;
            H_cum_norm(x_idx(1):x_idx(2),f_jdx(1):f_jdx(2))=H_norm_fid;
        end
        f_idx=h_idx_list(fid_idx,:);
        H_cum_norm(x_idx(1):x_idx(2),f_idx(1):f_idx(2))=H_norm;
    end
    L_cov_cum=chol(cov_cum+eye(size(cov_cum,1))*(1000*eps))';
    dLR_H_cum=L_cov_cum\H_cum_norm;
    dLR_Y_cum=L_cov_cum\Y_cum_norm;
    beta_cum=dLR_H_cum\dLR_Y_cum; % beta=inv_FTRF*(H'*inv_cov*Y);
    gamma_cum=L_cov_cum'\(dLR_Y_cum-dLR_H_cum*beta_cum); % U=Y-H*beta;
    inv_FTcovF_cum=(dLR_H_cum'*dLR_H_cum)\eye(size(H_cum_norm,2));

    % initialization predict function
    X_cum_norm=XALL_norm(1:x_idx(2),:);
    fid_cum_num=fid_idx;
    x_idx_cum=x_idx_list(1:fid_idx,:);
    f_idx_cum=h_idx_list(1:fid_idx,:);
    hyp_cum=hyp_list(1:fid_idx);
    sigma_sq_cum=sigma_sq_list(1:fid_idx);
    rho_cum=rho_list(1:fid_idx);
    reg_fcn_cum=reg_fcn_list(1:fid_idx);
    pred_fcn=@(X_pred) predictCoKRG...
        (X_cum_norm,X_pred,aver_X,stdD_X,aver_Y,stdD_Y,...
        fid_cum_num,vari_num,x_idx_cum,f_idx_cum,hyp_cum,rho_cum,sigma_sq_cum,...
        reg_fcn_cum,L_cov_cum,beta_cum,gamma_cum,dLR_H_cum,inv_FTcovF_cum);

    predict_list{fid_idx}=pred_fcn;
end

var_fid_fcn=@(X_pred,fid_l) varFidelity(X_pred,X_cum_norm,fid_l,fid_num,vari_num,...
            aver_X,stdD_X,aver_Y,stdD_Y,...
            x_idx_list,h_idx_list,hyp_list,rho_list,sigma_sq_list,...
            reg_fcn_list,L_cov_cum,dLR_H_cum,inv_FTcovF_cum);

model_CoKRG=option;
model_CoKRG.X=X_list;
model_CoKRG.Y=Y_list;
model_CoKRG.reg_fcn_list=reg_fcn_list;
model_CoKRG.hyp_list=hyp_list;
model_CoKRG.rho_list=rho_list;
model_CoKRG.sigma_sq_list=sigma_sq_list;
model_CoKRG.predict_list=predict_list;
model_CoKRG.predict=pred_fcn;
model_CoKRG.var_fid=var_fid_fcn;

    function [fval,grad]=probNLLBiasKRG(dr_sq,Y,Y_pred,x_num,vari_num,hyp,H)
        % function to minimize negative likelihood
        %
        hyp=min(hyp,up_bou_hyp);hyp=max(hyp,low_bou_hyp);
        theta=exp(hyp(1:end-1));
        if simplify_hyp, theta=theta*ones(1,vari_num);end % extend hyp
        rho=hyp(end);
        Y_bias=Y-rho*Y_pred; % notice Y equal to d

        [R,L,B,s2,dLR_F]=calCovKRG(dr_sq,Y_bias,x_num,vari_num,theta,H);

        % calculation negative log likelihood
        if s2 == 0,fval=0;
        else,fval=x_num/2*log(s2)+sum(log(diag(L)));end

        % calculate gradient
        if nargout > 1
            % gradient
            grad=zeros(vari_num+1,1);
            inv_R=L'\(L\eye(x_num));
            inv_FTRF=(dLR_F'*dLR_F)\eye(size(H,2));
            Y_Fmiu=Y-H*B;

            % theta d
            for vari_i=1:vari_num
                dR_dtheta=-(dr_sq(:,:,vari_i).*R)*theta(vari_i)/vari_num;
                dinv_R_dtheta=...
                    -inv_R*dR_dtheta*inv_R;
                dinv_FTRF_dtheta=-inv_FTRF*...
                    (H'*dinv_R_dtheta*H)*...
                    inv_FTRF;
                dbeta_dtheta=dinv_FTRF_dtheta*(H'*inv_R*Y_bias)+...
                    inv_FTRF*(H'*dinv_R_dtheta*Y_bias);
                dY_Fmiu_dtheta=-H*dbeta_dtheta;
                dsigma2_dtheta=(dY_Fmiu_dtheta'*inv_R*Y_Fmiu+...
                    Y_Fmiu'*dinv_R_dtheta*Y_Fmiu+...
                    Y_Fmiu'*inv_R*dY_Fmiu_dtheta)/x_num;
                dlnsigma2_dtheta=1/s2*dsigma2_dtheta;
                dlndetR=trace(inv_R*dR_dtheta);

                grad(vari_i)=x_num/2*dlnsigma2_dtheta+0.5*dlndetR;
            end

            % rho
            dY_drho=-Y_pred*rho;
            dbeta_drho=inv_FTRF*(H'*inv_R*dY_drho);
            dY_Fmiu_drho=(dY_drho-H*dbeta_drho);
            dsigma2_drho=(dY_Fmiu_drho'*inv_R*Y_Fmiu+...
                Y_Fmiu'*inv_R*dY_Fmiu_drho)/x_num;
            dlnsigma2_drho=1/s2*dsigma2_drho;

            grad(end)=x_num/2*dlnsigma2_drho;

            if simplify_hyp, grad=[sum(grad(1:end-1)),grad(end)];end
        end
    end

    function [cov,L_cov,beta,sigma_sq,dLR_H,dLR_Y,dLR_U]=calCovKRG...
            (dr_sq,Y,x_num,vari_num,theta,H)
        % kriging interpolation kernel function
        % y(x)=f(x)+z(x)
        %

        % calculate covariance
        cov=zeros(x_num,x_num);
        for vari_i=1:vari_num
            cov=cov+dr_sq(:,:,vari_i)*theta(vari_i);
        end
        cov=exp(-cov/vari_num^2)+eye(x_num)*((1000+x_num)*eps);

        % coefficient calculation
        L_cov=chol(cov)'; % inv_cov=cov\eye(x_num);
        dLR_H=L_cov\H;
        dLR_Y=L_cov\Y; % inv_FTRF=(H'*inv_cov*H)\eye(size(H,2));

        % basical bias
        beta=dLR_H\dLR_Y; % beta=inv_FTRF*(H'*inv_cov*Y);
        dLR_U=dLR_Y-dLR_H*beta; % U=Y-H*beta;
        sigma_sq=sum(dLR_U.^2)/x_num; % sigma_sq=(U'*inv_cov*U)/x_num;
    end

    function [Y_pred,Var_pred]=predictCoKRG(X_norm,X_pred,aver_X,stdD_X,aver_Y,stdD_Y,...
            fid_num,vari_num,x_idx_list,f_idx_list,hyp_list,rho_list,sigma_sq_list,...
            reg_fcn_list,L_cov,beta,gamma,dLR_H,inv_FTcovF)
        % Co-Kriging surrogate predict function
        %
        % input:
        % X_pred (matrix): x_pred_num x vari_num matrix, predict X
        %
        % output:
        % Y_pred (matrix): x_pred_num x 1 matrix, value
        % Var_pred (matrix): x_pred_num x 1 matrix, variance
        %
        
        % normalize data
        X_pred_norm=(X_pred-aver_X)./stdD_X;
        FR_pred_norm=calRegFid(X_pred,aver_Y,stdD_Y,rho_list,fid_num,reg_fcn_list,f_idx_list);

        cov_pred=calCovFid(X_norm,X_pred_norm,fid_num,fid_num,vari_num,...
            x_idx_list,hyp_list,rho_list,sigma_sq_list);

        % predict base fval
        Y_pred=FR_pred_norm*beta+cov_pred'*gamma;

        % predict variance
        dLR_r=L_cov\cov_pred;
        u_norm=(dLR_H)'*dLR_r-FR_pred_norm';
        Var_pred=0;
        for fid_bdx=1:fid_num
            rho_prod_kp=1;add_idx=fid_bdx;
            if add_idx < fid_num,rho_prod_kp=prod(rho_list(add_idx+1:fid_num));end
            Var_pred=Var_pred+(rho_prod_kp)^2*sigma_sq_list(add_idx);
        end
        Var_pred=Var_pred-dLR_r'*dLR_r+u_norm'*inv_FTcovF*u_norm;
        Var_pred=diag(Var_pred);Var_pred(Var_pred < eps)=0;

        % renormalize data
        Y_pred=Y_pred*stdD_Y+aver_Y;
    end

    function H_pred_norm=calRegFid(X_pred,aver_Y,stdD_Y,rho_list,fid_idx,reg_fcn_list,f_idx_list)
        % calculate regression value
        %
        [x_pred_num,~]=size(X_pred);
        H_pred_norm=zeros(x_pred_num,f_idx_list(end));

        % calculate each block of regression
        for fid_bdx=1:fid_idx-1
            f_bdx=f_idx_list(fid_bdx,:);
            reg_fcn_b=reg_fcn_list{fid_bdx};
            add_idx=fid_bdx;
            rho_prod_kp=1;
            if add_idx < fid_idx,rho_prod_kp=prod(rho_list(add_idx+1:fid_idx));end
            H_pred_norm(:,f_bdx(1):f_bdx(2))=rho_prod_kp*(reg_fcn_b(X_pred)-aver_Y)/stdD_Y;
        end
        fid_bdx=fid_idx;
        f_bdx=f_idx_list(fid_bdx,:);
        reg_fcn_b=reg_fcn_list{fid_bdx};
        H_pred_norm(:,f_bdx(1):f_bdx(2))=(reg_fcn_b(X_pred)-aver_Y)/stdD_Y;
    end

    function cov_pred=calCovFid(X_norm,X_pred_norm,fid_num,fid_l,vari_num,...
            x_idx_list,hyp_list,rho_list,sigma_sq_list)
        % predict covariance
        [x_pred_num,~]=size(X_pred_norm);
        cov_pred=zeros(x_idx_list(end),x_pred_num);

        % calculate each block of covariance
        for fid_bdx=1:fid_num
            x_bdx=x_idx_list(fid_bdx,:);
            cov_blockp=zeros(x_bdx(2)-x_bdx(1)+1,x_pred_num);

            % calculate each item of block
            fid_U=min(fid_bdx,fid_l);
            for fid_tdx=1:fid_U
                % calculate covariance
                theta_bip=exp(hyp_list{fid_tdx});
                cov_itp=zeros(x_bdx(2)-x_bdx(1)+1,x_pred_num);
                for vari_i=1:vari_num
                    cov_itp=cov_itp+(X_norm(x_bdx(1):x_bdx(2),vari_i)-X_pred_norm(:,vari_i)').^2*theta_bip(vari_i);
                end
                cov_itp=exp(-cov_itp/vari_num^2)*sigma_sq_list(fid_tdx);

                U_idx=fid_bdx;
                rho_prod_mp=1;
                if fid_tdx < U_idx,rho_prod_mp=prod(rho_list(fid_tdx+1:U_idx));end
                rho_prod_lp=1;
                if fid_tdx < fid_l,rho_prod_lp=prod(rho_list(fid_tdx+1:fid_l));end

                cov_blockp=cov_blockp+rho_prod_mp*rho_prod_lp*cov_itp;
            end
            cov_pred(x_bdx(1):x_bdx(2),:)=cov_blockp;
        end
    end

    function [Var_pred]=varFidelity(X_pred,X_norm,fid_l,fid_num,vari_num,...
            aver_X,stdD_X,aver_Y,stdD_Y,...
            x_idx_list,f_idx_list,hyp_list,rho_list,sigma_sq_list,...
            reg_fcn_list,L_cov,dLR_H,inv_FTcovF)
        % calculate Posterior distribution of fidelity l
        %

        % normalize data
        X_pred_norm=(X_pred-aver_X)./stdD_X;

        % predict regression
        FR_norm_l=calRegFid(X_pred,aver_Y,stdD_Y,rho_list,fid_l,reg_fcn_list,f_idx_list);
        FR_norm_m=calRegFid(X_pred,aver_Y,stdD_Y,rho_list,fid_num,reg_fcn_list,f_idx_list);

        % predict covaiance
        cov_pred_l=calCovFid(X_norm,X_pred_norm,fid_num,fid_l,vari_num,...
            x_idx_list,hyp_list,rho_list,sigma_sq_list);
        cov_pred_m=calCovFid(X_norm,X_pred_norm,fid_num,fid_num,vari_num,...
            x_idx_list,hyp_list,rho_list,sigma_sq_list);

        % predict variance
        dLR_r_l=L_cov\cov_pred_l;
        dLR_r_m=L_cov\cov_pred_m;
        u_norm_l=(dLR_H)'*dLR_r_l-FR_norm_l';
        u_norm_m=(dLR_H)'*dLR_r_m-FR_norm_m';

        Var_pred=0;
        for fid_bdx=1:fid_l
            add_idx=fid_bdx;
            rho_prod_lp=1;
            if add_idx < fid_num, rho_prod_lp=prod(rho_list(add_idx+1:fid_l));end
            rho_prod_mp=1;
            if add_idx < fid_num, rho_prod_mp=prod(rho_list(add_idx+1:fid_num));end
            Var_pred=Var_pred+rho_prod_lp*rho_prod_mp*sigma_sq_list(fid_bdx);
        end

        if fid_l == fid_num
            Var_pred=Var_pred-dLR_r_l'*dLR_r_m+u_norm_l'*inv_FTcovF*u_norm_m;
        else
            % to satisfy that corr(A,B) is in range [0,1]
            Var_pred=abs(Var_pred-dLR_r_l'*dLR_r_m);
        end
        Var_pred=diag(Var_pred);Var_pred(Var_pred < eps)=0;
    end
end
