function model_MMFS=srgtMmRBF(X_LF,Y_LF,X_HF,Y_HF,model_option)
% generate Modified Multi-Fildelity Surrogate model(MMFS)
% adaptive scaling factor
% input data will be normalize by average and standard deviation of data
%
% input:
% X_LF(x_LF_num x vari_num matrix), Y_LF(x_LF_num x 1 matrix), ...
% X_HF(x_HF_num x vari_num matrix), Y_HF(x_HF_num x 1 matrix), ...
% model_option(optional, basis_fcn_HF, basis_fcn_LF)
%
% output:
% model_MmRBF
%
% abbreviation:
% num: number, pred: predict, vari: variable, hyp: hyper parameter
%
% reference: [1] LIU Y,WANG S,ZHOU Q,et al. Modified Multifidelity
% Surrogate Model Based on Radial Basis Function with Adaptive Scale Factor
% [J]. Chinese Journal of Mechanical Engineering,2022,35(1): 77.
%
% Copyright 2023 Adel
%
if nargin < 5,model_option=struct();end

% MMFS option
if ~isfield(model_option,'basis_fcn_LF'), model_option.('basis_fcn_LF')=[];end
if ~isfield(model_option,'basis_fcn_HF'), model_option.('basis_fcn_HF')=[];end

% first step
% construct LF RBF model
basis_fcn_LF=model_option.basis_fcn_LF;
LF_model=srgtRBF(X_LF,Y_LF,basis_fcn_LF);
pred_fcn_LF=@(x) LF_model.predict(x);

% second step
% construct MMFS model

% normalize data
[x_HF_num,vari_num]=size(X_HF);
aver_X=mean(X_HF);
stdD_X=std(X_HF);stdD_X(stdD_X == 0)=1;
aver_Y=mean(Y_HF);
stdD_Y=std(Y_HF);stdD_Y(stdD_Y == 0)=1;

XHF_nomlz=(X_HF-aver_X)./stdD_X;
YHF_nomlz=(Y_HF-aver_Y)./stdD_Y;

% predict LF value at XHF point
YHF_pred=pred_fcn_LF(X_HF);

% nomalizae
YHF_pred_nomlz=(YHF_pred-aver_Y)./stdD_Y;

basis_fcn_HF=model_option.basis_fcn_HF;
if isempty(basis_fcn_HF)
    basis_fcn_HF=@(r) r.^3;
end

% initialization distance of XHF_nomlz
XHF_dis=zeros(x_HF_num,x_HF_num);
for vari_idx=1:vari_num
    XHF_dis=XHF_dis+...
        (XHF_nomlz(:,vari_idx)-XHF_nomlz(:,vari_idx)').^2;
end
XHF_dis=sqrt(XHF_dis);

[omega,RBF_matrix_HF,MmRBF_matrix]=calMmRBF...
    (XHF_dis,YHF_nomlz,basis_fcn_HF,x_HF_num,YHF_pred_nomlz);
alpha=omega(1:x_HF_num);
beta=omega(x_HF_num+1:end);

% initialization predict function
pred_fcn=@(X_predict) predictMmRBF...
    (X_predict,XHF_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_HF_num,vari_num,omega,basis_fcn_HF,pred_fcn_LF);

model_MMFS=model_option;
model_MMFS.X={X_LF,X_HF};
model_MMFS.Y={Y_LF,Y_HF};
model_MMFS.LF=LF_model;

model_MMFS.RBF_matrix=RBF_matrix_HF;
model_MMFS.MmRBF_matrix=MmRBF_matrix;
model_MMFS.alpha=alpha;
model_MMFS.beta=beta;

model_MMFS.predict=pred_fcn;

    function [omega,RBF_matrix,MmRBF_matrix]=calMmRBF...
            (X_dis,Y,basis_fcn,x_num,YHF_pred_nomlz)
        % interp polynomial responed surface core function
        % calculation beta
        %
        % Copyright 2022 Adel
        %
        RBF_matrix=basis_fcn(X_dis);

        % add low fildelity value
        MmRBF_matrix=[RBF_matrix.*YHF_pred_nomlz,RBF_matrix];
        MmRBF_matrix_hess=(MmRBF_matrix*MmRBF_matrix');

        % stabilize matrix
        MmRBF_matrix_hess=MmRBF_matrix_hess+eye(x_num)*(1000+x_num)*eps;

        % solve omega
        omega=MmRBF_matrix'*(MmRBF_matrix_hess\Y);
        % omega=H\Y;
    end

    function [Y_pred]=predictMmRBF...
            (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,omega,basis_fcn,pred_fcn_LF)
        % radial basis function interpolation predict function
        %
        [x_pred_num,~]=size(X_pred);

        % normalize data
        X_pred_nomlz=(X_pred-aver_X)./stdD_X;

        % calculate distance
        X_dis_pred=zeros(x_pred_num,x_num);
        for vari_i=1:vari_num
            X_dis_pred=X_dis_pred+...
                (X_pred_nomlz(:,vari_i)-X_nomlz(:,vari_i)').^2;
        end
        X_dis_pred=sqrt(X_dis_pred);

        % predict low fildelity value
        Y_pred_LF=pred_fcn_LF(X_pred);

        % nomalizae
        Y_pred_LF_nomlz=(Y_pred_LF-aver_Y)./stdD_Y;

        % combine two matrix
        RBF_matrix_pred=basis_fcn(X_dis_pred);
        H=[RBF_matrix_pred.*Y_pred_LF_nomlz,RBF_matrix_pred];

        % predict variance
        Y_pred_nomlz=H*omega;

        % normalize data
        Y_pred=Y_pred_nomlz*stdD_Y+aver_Y;
    end
end
