function model_CoRBF=srgtCoRBF(X_LF,Y_LF,X_HF,Y_HF,model_option)
% generate Co-RBF surrogate model
% optimize scaling factor
% input data will be normalize by average and standard deviation of data
%
% input:
% X_list (cell): x_num x vari_num matrix, the lower place, the higher fidelity
% Y_list (cell): x_num x 1 matrix, the lower place, the higher fidelity
% model_option (struct): optional input, include: basis_fcn_list, rho_list
%
% output:
% model_CoRBF
%
% abbreviation:
% num: number, pred: predict, vari: variable, hyp: hyper parameter
%
% reference: [1] Durantin C, Rouxel J, Désidéri J-A, et al. Multifidelity
% surrogate modeling based on radial basis functions [J]. Structural and
% multidisciplinary optimization, 2017, 56(5): 1061-75.
%
% Copyright 2023 Adel
%
if nargin < 5,model_option=struct();end

% Co-RBF option
if ~isfield(model_option,'model_option')
    model_option.('optimize_option')=optimoptions...
        ('fminunc','Display','none',...
        'OptimalityTolerance',1e-6,...
        'FiniteDifferenceStepSize',1e-5,...,
        'MaxIterations',20);
end
if ~isfield(model_option,'basis_fcn_LF'), model_option.('basis_fcn_LF')=[];end
if ~isfield(model_option,'basis_fcn_HF'), model_option.('basis_fcn_HF')=[];end
if ~isfield(model_option,'rho'), model_option.('rho')=[];end

% generate initial RBF
basis_fcn_LF=model_option.('basis_fcn_LF');
model_LF=srgtRBF(X_LF,Y_LF,basis_fcn_LF);

pred_fcn_LF=model_LF.predict;

% generate higher fidelity RBF
% construct hierarchical model
basis_fcn_HF=model_option.('basis_fcn_HF');
rho=model_option.('rho');
if isempty(rho)
    rho=mean(Y_HF)/mean(Y_LF);
end

% optimize rho by Rsq
rho_low_bou=-10;
rho_up_bou=10;
obj_Rsq=@(rho) norm(srgtRBF(X_HF,Y_HF-max(min(rho,rho_up_bou),rho_low_bou)*pred_fcn_LF(X_HF),basis_fcn_HF).Err());
rho=fminunc(obj_Rsq,rho,model_option.('optimize_option'));

D=Y_HF-rho*pred_fcn_LF(X_HF);
model_bias=srgtRBF(X_HF,D,basis_fcn_HF);

% generate predict function
pred_fcn=@(X_pred) model_LF.predict(X_pred)*rho+model_bias.predict(X_pred);
model_CoRBF=model_option;

model_CoRBF.X={X_LF,X_HF};
model_CoRBF.Y={Y_LF,Y_HF};
model_CoRBF.LF=model_LF;
model_CoRBF.bias=model_bias;

model_CoRBF.predict=pred_fcn;
end
