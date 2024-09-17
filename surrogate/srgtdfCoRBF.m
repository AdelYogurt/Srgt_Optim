function srgt=srgtdfCoRBF(XLF,YLF,XHF,YHF,option)
% generate Co-RBF surrogate model
% optimize scaling factor
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

if ~isfield(option,'basis_fcn_lf'), option.('basis_fcn_lf')=[];end
if ~isfield(option,'basis_fcn_hf'), option.('basis_fcn_hf')=[];end
if ~isfield(option,'rho'), option.('rho')=[];end

% generate initial RBF
basis_fcn_lf=option.('basis_fcn_lf');
model_lf=srgtsfRBF(XLF,YLF,struct('basis_fcn',basis_fcn_lf));
pred_fcn_lf=model_lf.predict;

% generate higher fidelity RBF
% construct hierarchical model
basis_fcn_hf=option.('basis_fcn_hf');

% scaling factor define and optimize
rho=option.('rho');
if isempty(rho),rho=mean(YHF)/mean(YLF);end
if option.optimize_rho % optimize rho by Rsq
    rho_low_bou=-10;
    rho_up_bou=10;
    obj_R2=@(rho) sum(srgtsfRBF(XHF,YHF-rho*pred_fcn_lf(XHF),struct('basis_fcn',basis_fcn_hf)).loo_err.^2);
    rho=fminbnd(obj_R2,rho_low_bou,rho_up_bou,option.('optimize_option'));
end

D=YHF-rho*pred_fcn_lf(XHF);
model_bias=srgtsfRBF(XHF,D,struct('basis_fcn',basis_fcn_hf));

% generate predict function
pred_fcn=@(X_pred) model_lf.predict(X_pred)*rho+model_bias.predict(X_pred);
srgt=option;

srgt.X={XLF,XHF};
srgt.Y={YLF,YHF};
srgt.LF=model_lf;
srgt.bias=model_bias;

srgt.predict=pred_fcn;
end
