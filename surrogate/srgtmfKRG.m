function srgt=srgtmfKRG(X_list,Y_list,option)
% generate Multi-Level Kriging surrogate model by fitting bias
% input data will be normalize by average and standard deviation of data
%
% input:
% X_list (cell): x_num x vari_num matrix, the lower, the higher fidelity
% Y_list (cell): x_num x 1 vector, the lower, the higher fidelity
% model_option (struct): optional, include: optimize_hyp, simplify_hyp, hyp, optimize_option
%
% output:
% srgt (struct): a Multi-Level Kriging model
%
% Copyright 2023.2 Adel
%
if nargin < 3,option=struct();end

% Kriging option
if ~isfield(option,'optimize_hyp'), option.('optimize_hyp')=true;end
if ~isfield(option,'simplify_hyp'), option.('simplify_hyp')=true;end
if option.('simplify_hyp'),FLAG_GRAD=false;else,FLAG_GRAD=true;end
if ~isfield(option,'optimize_option')
    option.('optimize_option')=optimoptions...
        ('fminunc','Display','none',...
        'OptimalityTolerance',1e-6,...
        'FiniteDifferenceStepSize',1e-5,...
        'MaxIterations',20,'SpecifyObjectiveGradient',FLAG_GRAD);
end

% Multi-Level Kriging option
if ~isfield(option,'model_list'), option.('model_list')={};end
if ~isfield(option,'hyp_list'), option.('hyp_list')={};end
if ~isfield(option,'reg_fcn_list'), option.('reg_fcn_list')={};end

if isnumeric(X_list),X_list={X_list};end;X_list=X_list(:);
if isnumeric(Y_list),Y_list={Y_list};end;Y_list=Y_list(:);

% load exist data
fid_num=length(X_list);
mdl_list=option.('model_list');
mdl_list=[mdl_list;repmat({[]},fid_num-length(mdl_list),1)];
hyp_list=option.('hyp_list');
hyp_list=[hyp_list;repmat({[]},fid_num-length(hyp_list),1)];
reg_fcn_list=option.('reg_fcn_list');
reg_fcn_list=[reg_fcn_list;repmat({[]},fid_num-length(reg_fcn_list),1)];

pred_fcn=@(X) zeros(size(X,1),1);
% construct total model
for fid_idx=1:fid_num
    X=X_list{fid_idx};
    Y=Y_list{fid_idx};
    mdl=mdl_list{fid_idx};
    hyp=hyp_list{fid_idx};
    reg_fcn=reg_fcn_list{fid_idx};

    if isempty(mdl)
        % generate initial model option
        mdl=struct();
        mdl.('optimize_hyp')=option.('optimize_hyp');
        mdl.('simplify_hyp')=option.('simplify_hyp');
        mdl.('optimize_option')=option.('optimize_option');
        mdl.('hyp')=hyp;
        mdl.('reg_fcn')=reg_fcn;
    
        % train bias Kriging
        D=Y-pred_fcn(X);
        mdl=srgtsfKRG(X,D,mdl);
        hyp_list{fid_idx}=mdl.('hyp');
        mdl_list{fid_idx}=mdl;
    end

    pred_fcn=mdl.predict;
end

% initialization predict function
pred_fcn=@(X_pred) predictMtKRG(X_pred,fid_num,mdl_list);

srgt=option;
srgt.X=X;
srgt.Y=Y;
srgt.hyp_list=hyp_list;
srgt.model_list=mdl_list;
srgt.predict=pred_fcn;

    function Y_pred=predictMtKRG(X_pred,fid_num,model_list)
        % multi-level kriging predict function
        %
        Y_pred=zeros(size(X_pred,1),1);
        for fidelity_i=1:fid_num
            Y_pred=Y_pred+model_list{fidelity_i}.predict(X_pred);
        end
    end
end
