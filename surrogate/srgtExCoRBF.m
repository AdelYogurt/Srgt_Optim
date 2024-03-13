function model_CoRBF=srgtExCoRBF(X_list,Y_list,model_option)
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
if nargin < 3,model_option=struct();end

% Co-RBF option
if ~isfield(model_option,'model_option'), model_option.('optimize_option')=optimoptions...
        ('fminunc','Display','none',...
        'OptimalityTolerance',1e-6,...
        'FiniteDifferenceStepSize',1e-5,...,
        'MaxIterations',20);end
if ~isfield(model_option,'basis_fcn_list'), model_option.('basis_fcn_list')={};end
if ~isfield(model_option,'rho_list'), model_option.('rho_list')=[];end

if isnumeric(X_list),X_list={X_list};end;X_list=X_list(:);
if isnumeric(Y_list),Y_list={Y_list};end;Y_list=Y_list(:);

% load exist data
fid_num=length(X_list);
model_list=cell(fid_num,1);
basis_fcn_list=model_option.('basis_fcn_list');
basis_fcn_list=[basis_fcn_list;repmat({[]},fid_num-length(basis_fcn_list),1)];
rho_list=model_option.('rho_list');
if isempty(rho_list), rho_list=zeros(fid_num,1);end
rho_list=[rho_list;zeros(fid_num-length(rho_list),1)];

% generate initial RBF
fid_idx=1;
X=X_list{fid_idx};
Y=Y_list{fid_idx};
basis_fcn=basis_fcn_list{fid_idx};
model=srgtRBF(X,Y,basis_fcn);
model_list{fid_idx}=model;
pred_fcn=model.predict;

% generate higher fidelity RBF
% construct hierarchical model
while fid_idx < fid_num
    fid_idx=fid_idx+1;
    X=X_list{fid_idx};
    Y=Y_list{fid_idx};
    basis_fcn=basis_fcn_list{fid_idx};
    rho=rho_list(fid_idx);
    
    % optimize rho by Rsq
    rho_low_bou=-10;
    rho_up_bou=10;
    obj_Rsq=@(rho) norm(srgtRBF(X,Y-max(min(rho,rho_up_bou),rho_low_bou)*pred_fcn(X),basis_fcn).Rsq());
    rho=fminunc(obj_Rsq,rho,model_option.('optimize_option'));

    rho_list(fid_idx)=rho;
    D=Y-rho*pred_fcn(X);
    model=srgtRBF(X,D,basis_fcn);
    model_list{fid_idx}=model;

    pred_fcn=@(X_pred) predictCoRBF(X_pred,model_list(1:fid_idx),rho_list(1:fid_idx));
end

model_CoRBF=model_option;

model_CoRBF.X=X_list;
model_CoRBF.Y=Y_list;
model_CoRBF.basis_fcn_list=basis_fcn_list;
model_CoRBF.rho_list=rho_list;
model_CoRBF.model_list=model_list;

model_CoRBF.predict=pred_fcn;

    function [Y_pred]=predictCoRBF(X_pred,model_list,rho_list)
        % radial basis function interpolation predict function
        %
        Y_pred=model_list{1}.predict(X_pred);

        for fid_i=2:length(model_list)
            Y_pred=Y_pred*rho_list(fid_i)+model_list{fid_i}.predict(X_pred);
        end
    end
end
