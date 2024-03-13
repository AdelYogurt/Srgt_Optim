function model_HrKRG=srgtHrKRG(X_list,Y_list,model_option)
% generate Hierarchical Kriging surrogate model
% support multi level fidelity input
% input data will be normalize by average and standard deviation of data
% hyp_list contain all level hyp
% notice: theta=exp(hyp)
%
% input:
% X_list (cell): x_num x vari_num matrix, the lower place, the higher fidelity
% Y_list (cell): x_num x 1 matrix, the lower place, the higher fidelity
% model_option (struct): optional input, include: optimize_hyp, simplify_hyp, hyp, optimize_option
%
% output:
% model_HrKRG (struct): a Hierarchical Kriging model
%
% abbreviation:
% num: number, pred: predict, vari: variable, hyp: hyper parameter
% NLL: negative log likelihood
%
% reference: [1] HAN Z-H, GöRTZ S. Hierarchical Kriging Model for
% Variable-Fidelity Surrogate Modeling [J]. AIAA Journal,2012,50(9):
% 1885-96.
%
% Copyright 2023.2 Adel
%
if nargin < 3,model_option=struct();end

% Kriging option
if isempty(model_option), model_option=struct();end
if ~isfield(model_option,'optimize_hyp'), model_option.('optimize_hyp')=true;end
if ~isfield(model_option,'simplify_hyp'), model_option.('simplify_hyp')=true;end
if model_option.('simplify_hyp'),FLAG_GRAD=false;else,FLAG_GRAD=true;end
if ~isfield(model_option,'optimize_option')
    model_option.('optimize_option')=optimoptions...
        ('fminunc','Display','none',...
        'OptimalityTolerance',1e-6,...
        'FiniteDifferenceStepSize',1e-5,...,
        'MaxIterations',20,'SpecifyObjectiveGradient',FLAG_GRAD);
end

% Hierarchical Kriging option
if ~isfield(model_option,'model_list'), model_option.('model_list')={};end
if ~isfield(model_option,'hyp_list'), model_option.('hyp_list')={};end
if ~isfield(model_option,'reg_fcn'), model_option.('reg_fcn')=[];end

if isnumeric(X_list),X_list={X_list};end;X_list=X_list(:);
if isnumeric(Y_list),Y_list={Y_list};end;Y_list=Y_list(:);

% load exist data
fid_num=length(X_list);
model_list=model_option.('model_list');
model_list=[model_list;repmat({[]},fid_num-length(model_list),1)];
hyp_list=model_option.('hyp_list');
hyp_list=[hyp_list;repmat({[]},fid_num-length(hyp_list),1)];
reg_fcn=model_option.('reg_fcn');

% generate initial Kriging
fid_idx=1;
X=X_list{fid_idx};
Y=Y_list{fid_idx};
model=model_list{fid_idx};
hyp=hyp_list{fid_idx};

if isempty(model)
    % generate initial model option
    model=struct();
    model.('optimize_hyp')=model_option.('optimize_hyp');
    model.('simplify_hyp')=model_option.('simplify_hyp');
    model.('optimize_option')=model_option.('optimize_option');
    model.('hyp')=hyp;
    model.('reg_fcn')=reg_fcn;

    % train Kriging
    model=srgtKRG(X,Y,model);
    hyp_list{fid_idx}=model.('hyp');
    model_list{fid_idx}=model;
end

pred_fcn=model.predict;

% generate higher fidelity Kriging
% construct hierarchical model
while fid_idx < fid_num
    fid_idx=fid_idx+1;
    X=X_list{fid_idx};
    Y=Y_list{fid_idx};
    model=model_list{fid_idx};
    hyp=hyp_list{fid_idx};

    if isempty(model)
        % generate initial model option
        model=struct();
        model.('optimize_hyp')=model_option.('optimize_hyp');
        model.('simplify_hyp')=model_option.('simplify_hyp');
        model.('optimize_option')=model_option.('optimize_option');
        model.('hyp')=hyp;

        % regression function define
        reg_fcn=pred_fcn;
        model.('reg_fcn')=reg_fcn;

        % train Kriging
        model=srgtKRG(X,Y,model);
        hyp_list{fid_idx}=model.('hyp');
        model_list{fid_idx}=model;
    end

    pred_fcn=model.predict;
end

model_HrKRG=model_option;

model_HrKRG.X=X;
model_HrKRG.Y=Y;
model_HrKRG.hyp_list=hyp_list;
model_HrKRG.model_list=model_list;

model_HrKRG.predict=pred_fcn;
end
