function model_MKQP=srgtMtKRGQdPg(X_HF,Y_HF,X_LF_list,Y_LF_list,model_option)
% construct Nonhierarchical Multi-Model Fusion Method Using Multi-Level 
% Kriging and Quadratic Programming surrogate model
%
% input:
% X_HF(x_HF_num x vari_num matrix), Y_HF(x_HF_num x 1 matrix),...
% X_LF_list(cell list of x_LF_num x vari_num matrix),...
% Y_LF_list(cell list of x_LF_num x 1 matrix),...
% model_option(optional, include: optimize_hyp, simplify_hyp,...
% hyp_bias, hyp_LF_list, optimize_option)
%
% output:
% model_MKQP(a NMF-MKQP model)
%
% abbreviation:
% num: number, pred: predict, vari: variable, hyp: hyper parameter
% NLL: negative log likelihood
%
% reference: [1] 武宇飞, 龙腾, 史人赫, 等. 跨域变体飞行器气动力热非层次
% 多模型融合降阶方法 [J/OL]. 航空学报.
%
% Copyright 2023.2 Adel
%
if nargin < 5,model_option=struct();end
if ~isfield(model_option,'optimize_hyp'), model_option.('optimize_hyp')=true(1);end
if ~isfield(model_option,'simplify_hyp'), model_option.('simplify_hyp')=false(1);end
if model_option.('simplify_hyp'),FLAG_GRAD=false;else,FLAG_GRAD=true;end
if ~isfield(model_option,'optimize_option')
    model_option.('optimize_option')=optimoptions...
        ('fminunc','Display','none',...
        'OptimalityTolerance',1e-6,...
        'FiniteDifferenceStepSize',1e-5,...,
        'MaxIterations',20,'SpecifyObjectiveGradient',FLAG_GRAD);
end

% MMKQP option
if ~isfield(model_option,'hyp_bias'), model_option.('hyp_bias')=[];end
if ~isfield(model_option,'hyp_LF_list'), model_option.('hyp_LF_list')=[];end

[x_HF_num,~]=size(X_HF);

% check input
if length(X_LF_list) ~= length(Y_LF_list)
    error('srgtMtKRGQdPg: low fidelity data number is inequal');
end
LF_num=length(X_LF_list);

% construct each low fidelity KRG model
hyp_LF_list=model_option.hyp_LF_list;
if isempty(hyp_LF_list)
    hyp_LF_list=cell(1,LF_num);
end
model_option_LF=model_option;
Model_KRG_LF=cell(1,LF_num);
for LF_index=1:LF_num
    model_option_LF.hyp=hyp_LF_list{LF_index};
    Model_KRG_LF{LF_index}=srgtKRG(X_LF_list{LF_index},Y_LF_list{LF_index},model_option_LF);
    hyp=Model_KRG_LF{LF_index}.hyp;
    hyp_LF_list{LF_index}=hyp;
end

% evaluate bias in high fidelity point
Y_pred_LF=zeros(x_HF_num,LF_num); % 
for LF_index=1:LF_num
    Y_pred_LF(:,LF_index)=Model_KRG_LF{LF_index}.predict(X_HF);
end
Y_bias_mat=Y_pred_LF-Y_HF;

% calculate weight of each model
C=Y_bias_mat'*Y_bias_mat;
% eta=trace(C)/x_HF_number;
eta=1000*eps;
w=(C+eta*eye(LF_num))\ones(LF_num,1)/...
    (ones(1,LF_num)/(C+eta*eye(LF_num))*ones(LF_num,1));
% disp(['no check min w: ',num2str(min(w))])
while min(w) < -0.05
    eta=eta*10;
    w=(C+eta*eye(LF_num))\ones(LF_num,1)/...
        (ones(1,LF_num)/(C+eta*eye(LF_num))*ones(LF_num,1));
end

% construct bias kriging model
Y_bias=Y_HF-Y_pred_LF*w;
model_option_bias=model_option;
model_option_bias.hyp=model_option.hyp_bias;
model_KRG_bias=srgtKRG(X_HF,Y_bias,model_option_bias);

% initialization predict function
pred_fcn=@(X_predict) predictMtKRGQdPg(X_predict,Model_KRG_LF,model_KRG_bias,w,LF_num);

model_MKQP.X=X_HF;
model_MKQP.Y=Y_HF;
model_MKQP.X_LF_list=X_LF_list;
model_MKQP.Y_LF_list=Y_LF_list;

model_MKQP.Model_KRG_LF=Model_KRG_LF;
model_MKQP.model_KRG_bias=model_KRG_bias;

model_MKQP.w=w;

model_MKQP.predict=pred_fcn;

    function Y_pred=predictMtKRGQdPg(X_pred,Model_KRG_LF,model_KRG_bias,w,LF_num)
        % Multi-Level Kriging and Quadratic Programming kriging predict function
        %
        x_pred_num=size(X_pred,1);
        Fval_pred_LF=zeros(x_pred_num,LF_num);
        for LF_idx=1:LF_num
            Fval_pred_LF(:,LF_idx)=Model_KRG_LF{LF_idx}.predict(X_pred);
        end
        
        Y_pred=Fval_pred_LF*w+model_KRG_bias.predict(X_pred);
    end
end
