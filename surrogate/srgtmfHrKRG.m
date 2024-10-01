function srgt=srgtmfHrKRG(X_list,Y_list,option)
% generate Multi-Level Hierarchical Kriging surrogate model
% input data will be normalize by average and standard deviation of data
%
% input:
% X_list (cell): x_num x vari_num matrix, the lower place, the higher fidelity
% Y_list (cell): x_num x 1 matrix, the lower place, the higher fidelity
% model_option (struct): optional input, include: optimize_hyp, simplify_hyp, hyp, optimize_option
%
% output:
% srgt (struct): a Multi-Level Hierarchical Kriging model
%
% reference:
% [1] HAN Z-H, GöRTZ S. Hierarchical Kriging Model for Variable-Fidelity
% Surrogate Modeling [J]. AIAA Journal,2012,50(9): 1885-1896.
%
% Copyright 2023.2 Adel
%
if nargin < 3,option=struct();end

% Kriging option
if isempty(option), option=struct();end
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

% Hierarchical Kriging option
if ~isfield(option,'model_list'), option.('model_list')={};end
if ~isfield(option,'hyp_list'), option.('hyp_list')={};end
if ~isfield(option,'reg_fcn'), option.('reg_fcn')=[];end

if isnumeric(X_list),X_list={X_list};end;X_list=X_list(:);
if isnumeric(Y_list),Y_list={Y_list};end;Y_list=Y_list(:);

% load exist data
fid_num=length(X_list);
model_list=option.('model_list');
model_list=[model_list;repmat({[]},fid_num-length(model_list),1)];
hyp_list=option.('hyp_list');
hyp_list=[hyp_list;repmat({[]},fid_num-length(hyp_list),1)];
reg_fcn=option.('reg_fcn');

% generate initial Kriging
fid_idx=1;
X=X_list{fid_idx};
Y=Y_list{fid_idx};
mdl=model_list{fid_idx};
hyp=hyp_list{fid_idx};

if isempty(mdl)
    % generate initial model option
    mdl=struct();
    mdl.('optimize_hyp')=option.('optimize_hyp');
    mdl.('simplify_hyp')=option.('simplify_hyp');
    mdl.('optimize_option')=option.('optimize_option');
    mdl.('hyp')=hyp;
    mdl.('reg_fcn')=reg_fcn;

    % train Kriging
    mdl=srgtsfKRG(X,Y,mdl);
    hyp_list{fid_idx}=mdl.('hyp');
    model_list{fid_idx}=mdl;
end

pred_fcn=mdl.predict;

% generate higher fidelity Kriging
% construct hierarchical model
while fid_idx < fid_num
    fid_idx=fid_idx+1;
    X=X_list{fid_idx};
    Y=Y_list{fid_idx};
    mdl=model_list{fid_idx};
    hyp=hyp_list{fid_idx};

    if isempty(mdl)
        % generate initial model option
        mdl=struct();
        mdl.('optimize_hyp')=option.('optimize_hyp');
        mdl.('simplify_hyp')=option.('simplify_hyp');
        mdl.('optimize_option')=option.('optimize_option');
        mdl.('hyp')=hyp;

        % regression function define
        reg_fcn=pred_fcn;
        mdl.('reg_fcn')=reg_fcn;

        % train Kriging
        mdl=srgtsfKRG(X,Y,mdl);
        hyp_list{fid_idx}=mdl.('hyp');
        model_list{fid_idx}=mdl;
    end

    pred_fcn=mdl.predict;
end

srgt=option;
srgt.X=X;
srgt.Y=Y;
srgt.hyp_list=hyp_list;
srgt.model_list=model_list;
srgt.predict=pred_fcn;
end
