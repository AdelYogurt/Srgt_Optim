function srgt=srgtdfKRGQdPg(X_LF_list,Y_LF_list,X_HF,Y_HF,option)
% construct Nonhierarchical Multi-Model Fusion Method Using Multi-Level 
% Kriging and Quadratic Programming surrogate model
%
% input:
% X_LF_list(cell): low fidelity trained X, list of x_LF_num x vari_num matrix
% Y_LF_list(cell): low fidelity trained Y, list of x_LF_num x 1 vector
% X_HF (matrix): high fidelity trained X, x_num x vari_num
% Y_HF (vector): high fidelity trained Y, x_num x 1
% option (struct): optional input, construct option
% model_option(optional, include: optimize_hyp, simplify_hyp,...
% hyp_bias, hyp_LF_list, optimize_option)
%
% output:
% srgt(a NMF-MKQP model)
%
% reference:
% [1] Yufei WU, Teng LONG, Renhe SHI, Yao ZHANG. Non⁃hierarchical
% multi⁃model fusion order reduction based on aerodynamic and
% aerothermodynamic characteristics for cross⁃domain morphing aircraft[J].
% Acta Aeronautica et Astronautica Sinica, 2023, 44(21): 528259-528259.
%
% Copyright 2023.2 Adel
%
if nargin < 5,option=struct();end
if ~isfield(option,'optimize_hyp'), option.('optimize_hyp')=true;end
if ~isfield(option,'simplify_hyp'), option.('simplify_hyp')=true;end
if option.('simplify_hyp'),FLAG_GRAD=false;else,FLAG_GRAD=true;end
if ~isfield(option,'optimize_option'), option.('optimize_option')=optimoptions...
        ('fminunc','Display','none',...
        'OptimalityTolerance',1e-6,...
        'FiniteDifferenceStepSize',1e-5,...
        'MaxIterations',20,'SpecifyObjectiveGradient',FLAG_GRAD);
end

% MMKQP option
if ~isfield(option,'hyp_bias'), option.('hyp_bias')=[];end
if ~isfield(option,'hyp_LF_list'), option.('hyp_LF_list')=[];end

[x_HF_num,~]=size(X_HF);

% check input
if isnumeric(X_LF_list),X_LF_list={X_LF_list};end
if isnumeric(Y_LF_list),Y_LF_list={Y_LF_list};end
if length(X_LF_list) ~= length(Y_LF_list)
    error('srgtdfKRGQdPg: low fidelity data number is inequal');
end
LF_num=length(X_LF_list);

% construct each low fidelity KRG model
hyp_LF_list=option.hyp_LF_list;
if isempty(hyp_LF_list),hyp_LF_list=cell(1,LF_num);end
mdl_LF_list=cell(1,LF_num);
for LF_idx=1:LF_num
    option_LF.optimize_hyp=option.optimize_hyp;
    option_LF.simplify_hyp=option.simplify_hyp;
    option_LF.optimize_option=option.optimize_option;
    option_LF.hyp=hyp_LF_list{LF_idx};

    mdl_LF_list{LF_idx}=srgtsfKRG(X_LF_list{LF_idx},Y_LF_list{LF_idx},option_LF);
    hyp=mdl_LF_list{LF_idx}.hyp;
    hyp_LF_list{LF_idx}=hyp;
end

% evaluate bias in high fidelity point
Y_pred_LF=zeros(x_HF_num,LF_num); % 
for LF_idx=1:LF_num
    Y_pred_LF(:,LF_idx)=mdl_LF_list{LF_idx}.predict(X_HF);
end
Y_bias_mat=Y_pred_LF-Y_HF;

% calculate weight of each model
C=Y_bias_mat'*Y_bias_mat;
% eta=trace(C)/x_HF_number;
eta=1000*eps;
weight=(C+eta*eye(LF_num))\ones(LF_num,1)/...
    (ones(1,LF_num)/(C+eta*eye(LF_num))*ones(LF_num,1));
% disp(['no check min w: ',num2str(min(w))])
while min(weight) < -0.05
    eta=eta*10;
    weight=(C+eta*eye(LF_num))\ones(LF_num,1)/...
        (ones(1,LF_num)/(C+eta*eye(LF_num))*ones(LF_num,1));
end

% construct bias kriging model
Y_bias=Y_HF-Y_pred_LF*weight;
option_bias.optimize_hyp=option.optimize_hyp;
option_bias.simplify_hyp=option.simplify_hyp;
option_bias.optimize_option=option.optimize_option;
option_bias.hyp=option.hyp_bias;
mdl_bias=srgtsfKRG(X_HF,Y_bias,option_bias);

% initialization predict function
pred_fcn=@(X_pred) predictMtKRGQdPg(X_pred,mdl_LF_list,mdl_bias,weight,LF_num);

hyp_bias=mdl_bias.hyp;
hyp_LF_list=cell(1,LF_num);
for LF_idx=1:LF_num,hyp_LF_list{LF_idx}=mdl_LF_list{LF_idx}.hyp;end

srgt=option;
srgt.X=X_HF;
srgt.Y=Y_HF;
srgt.X_LF_list=X_LF_list;
srgt.Y_LF_list=Y_LF_list;
% srgt.model_LF_list=mdl_LF_list;
% srgt.model_bias=mdl_bias;
srgt.hyp_bias=hyp_bias;
srgt.hyp_LF_list=hyp_LF_list;
srgt.weight=weight;
srgt.predict=pred_fcn;


    function Y_pred=predictMtKRGQdPg(X_pred,mdl_LF_list,mdl_bias,weight,LF_num)
        % Multi-Level Kriging and Quadratic Programming kriging predict function
        %
        x_pred_num=size(X_pred,1);
        Y_pred_LF=zeros(x_pred_num,LF_num);
        for LF_idx=1:LF_num
            Y_pred_LF(:,LF_idx)=mdl_LF_list{LF_idx}.predict(X_pred);
        end
        
        Y_pred=Y_pred_LF*weight+mdl_bias.predict(X_pred);
    end
end
