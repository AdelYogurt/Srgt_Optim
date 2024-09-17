function srgt=srgtdfMmRBF(XLF,YLF,XHF,YHF,option)
% generate Modified Multi-Fildelity Surrogate model(MMFS)
% using adaptive scaling factor compare CoRBF
% input data will be normalize by average and standard deviation of data
%
% input:
% XLF (matrix): low fidelity trained X, x_num x vari_num
% YLF (vector): low fidelity trained Y, x_num x 1
% XHF (matrix): high fidelity trained X, x_num x vari_num
% YHF (vector): high fidelity trained Y, x_num x 1
% option (struct): optional input, construct option
%
% output:
% srgt(struct): a Modified Multi-Fildelity model
%
% reference:
% [1] Liu Y, Wang S, Zhou Q, et al. Modified Multifidelity Surrogate Model
% Based on Radial Basis Function with Adaptive Scale Factor[J]. Chinese
% Journal of Mechanical Engineering, 2022, 35: 77.
%
% Copyright 2023.2 Adel
%
if nargin < 5,option=struct();end

% MMFS option
if ~isfield(option,'basis_fcn_lf'), option.('basis_fcn_lf')=[];end
if ~isfield(option,'basis_fcn_hf'), option.('basis_fcn_hf')=[];end

% first step
% construct LF RBF model
basis_fcn_lf=option.basis_fcn_lf;
LF_model=srgtsfRBF(XLF,YLF,struct('basis_fcn',basis_fcn_lf));
pred_fcn_lf=@(x) LF_model.predict(x);

% second step
% construct MMFS model

% normalize data
[xhf_num,vari_num]=size(XHF);
aver_X=mean(XHF);
stdD_X=std(XHF);stdD_X(stdD_X == 0)=1;
aver_Y=mean(YHF);
stdD_Y=std(YHF);stdD_Y(stdD_Y == 0)=1;

XHF_norm=(XHF-aver_X)./stdD_X;
YHF_norm=(YHF-aver_Y)./stdD_Y;

% predict LF value at XHF point
YHF_pred=pred_fcn_lf(XHF);

% nomalizae
YHF_pred_norm=(YHF_pred-aver_Y)./stdD_Y;

basis_fcn_hf=option.basis_fcn_hf;
if isempty(basis_fcn_hf)
    basis_fcn_hf=@(r) r.^3;
end

% initialization distance of XHF_norm
rHF=zeros(xhf_num,xhf_num);
for vari_idx=1:vari_num
    rHF=rHF+(XHF_norm(:,vari_idx)-XHF_norm(:,vari_idx)').^2;
end
rHF=sqrt(rHF);

[omega,gram,gram_Mm]=calMmRBF...
    (rHF,YHF_norm,basis_fcn_hf,xhf_num,YHF_pred_norm);
alpha=omega(1:xhf_num);
beta=omega(xhf_num+1:end);

% initialization predict function
pred_fcn=@(X_predict) predictMmRBF...
    (X_predict,XHF_norm,aver_X,stdD_X,aver_Y,stdD_Y,...
    xhf_num,vari_num,omega,basis_fcn_hf,pred_fcn_lf);

srgt=option;
srgt.X={XLF,XHF};
srgt.Y={YLF,YHF};
srgt.LF=LF_model;
srgt.gram=gram;
srgt.gram_Mm=gram_Mm;
srgt.alpha=alpha;
srgt.beta=beta;
srgt.predict=pred_fcn;

    function [omega,RBF_matrix,MmRBF_matrix]=calMmRBF...
            (X_dis,Y,basis_fcn,x_num,YHF_pred_norm)
        % interp polynomial responed surface core function
        % calculation beta
        %
        % Copyright 2022 Adel
        %
        RBF_matrix=basis_fcn(X_dis);

        % add low fildelity value
        MmRBF_matrix=[RBF_matrix.*YHF_pred_norm,RBF_matrix];
        MmRBF_matrix_hess=(MmRBF_matrix*MmRBF_matrix');

        % stabilize matrix
        MmRBF_matrix_hess=MmRBF_matrix_hess+eye(x_num)*(1000+x_num)*eps;

        % solve omega
        omega=MmRBF_matrix'*(MmRBF_matrix_hess\Y);
        % omega=H\Y;
    end

    function [Y_pred]=predictMmRBF...
            (X_pred,X_norm,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,omega,basis_fcn,pred_fcn_lf)
        % radial basis function interpolation predict function
        %
        [x_pred_num,~]=size(X_pred);

        % normalize data
        X_pred_norm=(X_pred-aver_X)./stdD_X;

        % calculate distance
        X_dis_pred=zeros(x_pred_num,x_num);
        for vari_i=1:vari_num
            X_dis_pred=X_dis_pred+...
                (X_pred_norm(:,vari_i)-X_norm(:,vari_i)').^2;
        end
        X_dis_pred=sqrt(X_dis_pred);

        % predict low fildelity value
        Y_pred_lf=pred_fcn_lf(X_pred);

        % nomalizae
        Y_pred_lf_norm=(Y_pred_lf-aver_Y)./stdD_Y;

        % combine two matrix
        RBF_matrix_pred=basis_fcn(X_dis_pred);
        H=[RBF_matrix_pred.*Y_pred_lf_norm,RBF_matrix_pred];

        % predict variance
        Y_pred_norm=H*omega;

        % normalize data
        Y_pred=Y_pred_norm*stdD_Y+aver_Y;
    end
end
