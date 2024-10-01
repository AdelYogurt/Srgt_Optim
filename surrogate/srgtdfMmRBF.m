function srgt=srgtdfMmRBF(X_LF,Y_LF,X_HF,Y_HF,option)
% generate Modified Multi-Fildelity Surrogate model(MMFS)
% using adaptive scaling factor compare CoRBF
% input data will be normalize by average and standard deviation of data
%
% input:
% X_LF (matrix): low fidelity trained X, x_num x vari_num
% Y_LF (vector): low fidelity trained Y, x_num x 1
% X_HF (matrix): high fidelity trained X, x_num x vari_num
% Y_HF (vector): high fidelity trained Y, x_num x 1
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
if ~isfield(option,'basis_fcn_LF'), option.('basis_fcn_LF')=[];end
if ~isfield(option,'basis_fcn_HF'), option.('basis_fcn_HF')=[];end

% first step
% construct LF RBF model
basis_fcn_LF=option.basis_fcn_LF;
LF_model=srgtsfRBF(X_LF,Y_LF,struct('basis_fcn',basis_fcn_LF));
pred_fcn_LF=@(x) LF_model.predict(x);

% second step
% construct MMFS model

% normalize data
[x_HF_num,vari_num]=size(X_HF);
aver_X=mean(X_HF);
stdD_X=std(X_HF);stdD_X(stdD_X == 0)=1;
aver_Y=mean(Y_HF);
stdD_Y=std(Y_HF);stdD_Y(stdD_Y == 0)=1;

X_HF_norm=(X_HF-aver_X)./stdD_X;
Y_HF_norm=(Y_HF-aver_Y)./stdD_Y;

% predict LF value at X_HF point
Y_HF_pred=pred_fcn_LF(X_HF);

% nomalizae
Y_HF_pred_norm=(Y_HF_pred-aver_Y)./stdD_Y;

basis_fcn_HF=option.basis_fcn_HF;
if isempty(basis_fcn_HF)
    basis_fcn_HF=@(r) r.^3;
end

% initialization distance of X_HF_norm
rHF=zeros(x_HF_num,x_HF_num);
for vari_idx=1:vari_num
    rHF=rHF+(X_HF_norm(:,vari_idx)-X_HF_norm(:,vari_idx)').^2;
end
rHF=sqrt(rHF);

[omega,gram,gram_Mm]=calMmRBF...
    (rHF,Y_HF_norm,basis_fcn_HF,x_HF_num,Y_HF_pred_norm);
alpha=omega(1:x_HF_num);
beta=omega(x_HF_num+1:end);

% initialization predict function
pred_fcn=@(X_predict) predictMmRBF...
    (X_predict,X_HF_norm,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_HF_num,vari_num,omega,basis_fcn_HF,pred_fcn_LF);

srgt=option;
srgt.X={X_LF,X_HF};
srgt.Y={Y_LF,Y_HF};
srgt.LF=LF_model;
srgt.gram=gram;
srgt.gram_Mm=gram_Mm;
srgt.alpha=alpha;
srgt.beta=beta;
srgt.predict=pred_fcn;

    function [omega,RBF_matrix,MmRBF_matrix]=calMmRBF...
            (X_dis,Y,basis_fcn,x_num,Y_HF_pred_norm)
        % interp polynomial responed surface core function
        % calculation beta
        %
        % Copyright 2022 Adel
        %
        RBF_matrix=basis_fcn(X_dis);

        % add low fildelity value
        MmRBF_matrix=[RBF_matrix.*Y_HF_pred_norm,RBF_matrix];
        MmRBF_matrix_hess=(MmRBF_matrix*MmRBF_matrix');

        % stabilize matrix
        MmRBF_matrix_hess=MmRBF_matrix_hess+eye(x_num)*(1000+x_num)*eps;

        % solve omega
        omega=MmRBF_matrix'*(MmRBF_matrix_hess\Y);
        % omega=H\Y;
    end

    function [Y_pred]=predictMmRBF...
            (X_pred,X_norm,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,omega,basis_fcn,pred_fcn_LF)
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
        Y_pred_LF=pred_fcn_LF(X_pred);

        % nomalizae
        Y_pred_LF_norm=(Y_pred_LF-aver_Y)./stdD_Y;

        % combine two matrix
        RBF_matrix_pred=basis_fcn(X_dis_pred);
        H=[RBF_matrix_pred.*Y_pred_LF_norm,RBF_matrix_pred];

        % predict variance
        Y_pred_norm=H*omega;

        % normalize data
        Y_pred=Y_pred_norm*stdD_Y+aver_Y;
    end
end
