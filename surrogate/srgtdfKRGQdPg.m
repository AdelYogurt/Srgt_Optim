function srgt=srgtdfKRGQdPg(XLF_list,YLF_list,XHF,YHF,option)
% construct Nonhierarchical Multi-Model Fusion Method Using Multi-Level 
% Kriging and Quadratic Programming surrogate model
%
% input:
% XLF_list(cell): low fidelity trained X, list of xlf_num x vari_num matrix
% YLF_list(cell): low fidelity trained Y, list of xlf_num x 1 vector
% XHF (matrix): high fidelity trained X, x_num x vari_num
% YHF (vector): high fidelity trained Y, x_num x 1
% option (struct): optional input, construct option
% model_option(optional, include: optimize_hyp, simplify_hyp,...
% hyp_bias, hyp_lf_list, optimize_option)
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
if ~isfield(option,'model_option'), option.('optimize_option')=optimoptions...
        ('fminunc','Display','none',...
        'OptimalityTolerance',1e-6,...
        'FiniteDifferenceStepSize',1e-5,...
        'MaxIterations',20,'SpecifyObjectiveGradient',FLAG_GRAD);
end

% MMKQP option
if ~isfield(option,'hyp_bias'), option.('hyp_bias')=[];end
if ~isfield(option,'hyp_lf_list'), option.('hyp_lf_list')=[];end

[xhf_num,~]=size(XHF);

% check input
if isnumeric(XLF_list),XLF_list={XLF_list};end
if isnumeric(YLF_list),YLF_list={YLF_list};end
if length(XLF_list) ~= length(YLF_list)
    error('srgtdfKRGQdPg: low fidelity data number is inequal');
end
lf_num=length(XLF_list);

% construct each low fidelity KRG model
hyp_lf_list=option.hyp_lf_list;
if isempty(hyp_lf_list),hyp_lf_list=cell(1,lf_num);end
option_lf=option;
mdl_lf_list=cell(1,lf_num);
for lf_idx=1:lf_num
    option_lf.hyp=hyp_lf_list{lf_idx};
    mdl_lf_list{lf_idx}=srgtsfKRG(XLF_list{lf_idx},YLF_list{lf_idx},option_lf);
    hyp=mdl_lf_list{lf_idx}.hyp;
    hyp_lf_list{lf_idx}=hyp;
end

% evaluate bias in high fidelity point
Y_pred_lf=zeros(xhf_num,lf_num); % 
for lf_idx=1:lf_num
    Y_pred_lf(:,lf_idx)=mdl_lf_list{lf_idx}.predict(XHF);
end
Y_bias_mat=Y_pred_lf-YHF;

% calculate weight of each model
C=Y_bias_mat'*Y_bias_mat;
% eta=trace(C)/xhf_number;
eta=1000*eps;
weight=(C+eta*eye(lf_num))\ones(lf_num,1)/...
    (ones(1,lf_num)/(C+eta*eye(lf_num))*ones(lf_num,1));
% disp(['no check min w: ',num2str(min(w))])
while min(weight) < -0.05
    eta=eta*10;
    weight=(C+eta*eye(lf_num))\ones(lf_num,1)/...
        (ones(1,lf_num)/(C+eta*eye(lf_num))*ones(lf_num,1));
end

% construct bias kriging model
Y_bias=YHF-Y_pred_lf*weight;
option_bias=option;
option_bias.hyp=option.hyp_bias;
mdl_bias=srgtsfKRG(XHF,Y_bias,option_bias);

% initialization predict function
pred_fcn=@(X_pred) predictMtKRGQdPg(X_pred,mdl_lf_list,mdl_bias,weight,lf_num);

srgt=option;
srgt.X=XHF;
srgt.Y=YHF;
srgt.XLF_list=XLF_list;
srgt.YLF_list=YLF_list;
srgt.model_lf_list=mdl_lf_list;
srgt.model_bias=mdl_bias;
srgt.weight=weight;
srgt.predict=pred_fcn;

    function Y_pred=predictMtKRGQdPg(X_pred,mdl_lf_list,mdl_bias,weight,lf_num)
        % Multi-Level Kriging and Quadratic Programming kriging predict function
        %
        x_pred_num=size(X_pred,1);
        Y_pred_lf=zeros(x_pred_num,lf_num);
        for LF_idx=1:lf_num
            Y_pred_lf(:,LF_idx)=mdl_lf_list{LF_idx}.predict(X_pred);
        end
        
        Y_pred=Y_pred_lf*weight+mdl_bias.predict(X_pred);
    end
end
