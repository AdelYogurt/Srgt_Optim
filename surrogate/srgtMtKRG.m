function model_MtKRG=srgtMtKRG(X_list,Y_list,model_option)
% generate Multi-Level Kriging surrogate model by fitting bias
% input data will be normalize by average and standard deviation of data
%
% input:
% X_list (cell): x_HF_num x vari_num matrix, the lower, the higher fidelity
% Y_list (cell): x_HF_num x 1 matrix, the lower, the higher fidelity
% model_option (struct): optional, include: optimize_hyp, simplify_hyp, hyp, optimize_option
%
% output:
% model_HrKRG (struct): a Hierarchical Kriging model
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
if ~isfield(model_option,'hyp_list'), model_option.('hyp_list')={};end
if ~isfield(model_option,'model_list'), model_option.('model_list')={};end
if ~isfield(model_option,'reg_fcn_list'), model_option.('reg_fcn_list')={};end

if isnumeric(X_list),X_list={X_list};end;X_list=X_list(:);
if isnumeric(Y_list),Y_list={Y_list};end;Y_list=Y_list(:);

% load exist data
fidelity_num=length(X_list);
model_list=model_option.('model_list');
model_list=[model_list;repmat({[]},fidelity_num-length(model_list),1)];
hyp_list=model_option.('hyp_list');
hyp_list=[hyp_list;repmat({[]},fidelity_num-length(hyp_list),1)];
reg_fcn_list=model_option.('reg_fcn_list');
reg_fcn_list=[reg_fcn_list;repmat({[]},fidelity_num-length(reg_fcn_list),1)];

pred_fcn=@(X) zeros(size(X));
% construct total model
for fidelity_idx=1:fidelity_num
    X=X_list{fidelity_idx};
    Y=Y_list{fidelity_idx};
    hyp=hyp_list{fidelity_idx};
    model=model_list{fidelity_idx};
    reg_fcn=reg_fcn_list{fidelity_idx};

    if isempty(model)
        % generate initial model option
        model=struct();
        model.('optimize_hyp')=model_option.('optimize_hyp');
        model.('simplify_hyp')=model_option.('simplify_hyp');
        model.('optimize_option')=model_option.('optimize_option');

        [~,vari_num]=size(X);
        % kernal function is exp(-X_sq/vari_num^2*exp(hyp))
        if isempty(hyp), hyp=ones(1,vari_num);end
        % if isempty(hyp), hyp=log(x_num^(1/vari_num)*vari_num)*ones(1,vari_num);end
        model.('hyp')=hyp;

        % regression function define
        if isempty(reg_fcn)
            stdD_X=std(X);stdD_X(stdD_X == 0)=1;
            % reg_fcn=@(X) ones(size(X,1),1); % zero
            reg_fcn=@(X) [ones(size(X,1),1),X-stdD_X]; % linear
        end
        model.('reg_fcn')=reg_fcn;
    
        % train bias Kriging
        Y_bias=Y-pred_fcn(X);
        model=srgtKRG(X,Y_bias,model);
        hyp_list{fidelity_idx}=model.('hyp');
        model_list{fidelity_idx}=model;
    end

    pred_fcn=model.predict;
end

% initialization predict function
pred_fcn=@(X_pred) predictMtKRG(X_pred,fidelity_num,model_list);

model_MtKRG=model_option;

model_MtKRG.X=X;
model_MtKRG.Y=Y;
model_MtKRG.hyp_list=hyp_list;
model_MtKRG.model_list=model_list;

model_MtKRG.predict=pred_fcn;

    function Y_pred=predictMtKRG(X_pred,fidelity_num,model_list)
        % multi-level kriging predict function
        %
        Y_pred=zeros(size(X_pred,2,1));
        for fidelity_i=1:fidelity_num
            Y_pred=Y_pred+model_list{fidelity_i}.predict(X_pred);
        end
    end
end
