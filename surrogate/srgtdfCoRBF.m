function srgt=srgtdfCoRBF(X_LF,Y_LF,X_HF,Y_HF,option)
% generate Co-RBF surrogate model
% optimize scaling factor
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
% srgt(struct): a Co-RBF surrogate model
%
% reference:
% [1] Durantin C, Rouxel J, Désidéri J-A, et al. Multifidelity surrogate
% modeling based on radial basis functions [J]. Structural and
% multidisciplinary optimization, 2017, 56(5): 1061-1075.
%
% Copyright 2023 Adel
%
if nargin < 5,option=struct();end

% Co-RBF option
if ~isfield(option,'optimize_rho'), option.('optimize_rho')=true;end
if ~isfield(option,'optimize_option')
    option.('optimize_option')=optimset('Display','none','TolFun',1e-6);
end

if ~isfield(option,'basis_fcn_LF'), option.('basis_fcn_LF')=[];end
if ~isfield(option,'basis_fcn_HF'), option.('basis_fcn_HF')=[];end
if ~isfield(option,'rho'), option.('rho')=[];end

% generate initial RBF
basis_fcn_LF=option.('basis_fcn_LF');
model_LF=srgtsfRBF(X_LF,Y_LF,struct('basis_fcn',basis_fcn_LF));
pred_fcn_LF=model_LF.predict;

% generate higher fidelity RBF
% construct hierarchical model
basis_fcn_HF=option.('basis_fcn_HF');

% scaling factor define and optimize
rho=option.('rho');
if isempty(rho),rho=mean(Y_HF)/mean(Y_LF);end
if option.optimize_rho % optimize rho by Rsq
    rho_low_bou=-10;
    rho_up_bou=10;
    obj_R2=@(rho) sum(srgtsfRBF(X_HF,Y_HF-rho*pred_fcn_LF(X_HF),struct('basis_fcn',basis_fcn_HF)).loo_err.^2);
    rho=fminbnd(obj_R2,rho_low_bou,rho_up_bou,option.('optimize_option'));
end

D=Y_HF-rho*pred_fcn_LF(X_HF);
model_bias=srgtsfRBF(X_HF,D,struct('basis_fcn',basis_fcn_HF));

% generate predict function
pred_fcn=@(X_pred) model_LF.predict(X_pred)*rho+model_bias.predict(X_pred);
srgt=option;

srgt.X={X_LF,X_HF};
srgt.Y={Y_LF,Y_HF};
srgt.LF=model_LF;
srgt.bias=model_bias;

srgt.predict=pred_fcn;
end
