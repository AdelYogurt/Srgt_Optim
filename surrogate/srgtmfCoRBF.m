function srgt=srgtmfCoRBF(X_list,Y_list,option)
% generate Multi-Level Co-RBF surrogate model
% optimize scaling factor
% input data will be normalize by average and standard deviation of data
%
% input:
% X_list (cell): x_num x vari_num matrix, the lower place, the higher fidelity
% Y_list (cell): x_num x 1 vector, the lower place, the higher fidelity
% model_option (struct): optional input, include: basis_fcn_list, rho_list
%
% output:
% srgt (struct): a Multi-Level Co-RBF model
%
% reference:
% [1] Durantin C, Rouxel J, Désidéri J-A, et al. Multifidelity surrogate
% modeling based on radial basis functions [J]. Structural and
% multidisciplinary optimization, 2017, 56(5): 1061-1075.
%
% Copyright 2023.2 Adel
%
if nargin < 3,option=struct();end

% Co-RBF option
if ~isfield(option,'optimize_rho'), option.('optimize_rho')=true;end
if ~isfield(option,'optimize_option')
    option.('optimize_option')=optimset('Display','none','TolFun',1e-6);
end

if ~isfield(option,'basis_fcn_list'), option.('basis_fcn_list')={};end
if ~isfield(option,'rho_list'), option.('rho_list')=[];end

if isnumeric(X_list),X_list={X_list};end;X_list=X_list(:);
if isnumeric(Y_list),Y_list={Y_list};end;Y_list=Y_list(:);

% load exist data
fid_num=length(X_list);
mdl_list=cell(fid_num,1);
basis_fcn_list=option.('basis_fcn_list');
basis_fcn_list=[basis_fcn_list;repmat({[]},fid_num-length(basis_fcn_list),1)];
rho_list=option.('rho_list');
rho_list=[rho_list;repmat({[]},fid_num-length(rho_list),1)];

% generate initial RBF
fid_idx=1;
X=X_list{fid_idx};
Y=Y_list{fid_idx};
basis_fcn=basis_fcn_list{fid_idx};
mdl=srgtsfRBF(X,Y,struct('basis_fcn',basis_fcn));
mdl_list{fid_idx}=mdl;
pred_fcn=mdl.predict;

% generate higher fidelity RBF
% construct hierarchical model
while fid_idx < fid_num
    fid_idx=fid_idx+1;
    X=X_list{fid_idx};
    Y=Y_list{fid_idx};
    basis_fcn=basis_fcn_list{fid_idx};
    rho=rho_list{fid_idx};

    if isempty(rho),rho=mean(X)/mean(X_list{fid_idx-1});end
    if option.optimize_rho % optimize rho by Rsq
        rho_low_bou=-10;
        rho_up_bou=10;
        obj_R2=@(rho) sum(srgtsfRBF(X,Y-rho*pred_fcn(X),basis_fcn).loo_err.^2);
        rho=fminbnd(obj_R2,rho_low_bou,rho_up_bou,option.('optimize_option'));
    end

    rho_list{fid_idx}=rho;
    D=Y-rho*pred_fcn(X);
    mdl=srgtsfRBF(X,D,basis_fcn);
    mdl_list{fid_idx}=mdl;

    pred_fcn=@(X_pred) predictCoRBF(X_pred,mdl_list(1:fid_idx),rho_list(1:fid_idx));
end

srgt=option;
srgt.X=X_list;
srgt.Y=Y_list;
srgt.basis_fcn_list=basis_fcn_list;
srgt.rho_list=rho_list;
srgt.model_list=mdl_list;
srgt.predict=pred_fcn;

    function [Y_pred]=predictCoRBF(X_pred,mdl_list,rho_list)
        % radial basis function interpolation predict function
        %
        Y_pred=mdl_list{1}.predict(X_pred);

        for fid_i=2:length(mdl_list)
            Y_pred=Y_pred*rho_list{fid_i}+mdl_list{fid_i}.predict(X_pred);
        end
    end
end
