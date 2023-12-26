clc;
clear;
close all hidden;

load('Morphed.mat');

matCL2CD = zeros(size(DataStruct,2),4);
vecNumMorph = zeros(size(DataStruct,2),1);
matMorph = [];
matMaAoAH = [];
matCl = [];
matCd = [];
matSref = [];
for i = 1:size(DataStruct,2)
    CLCD = DataStruct(i).cl./DataStruct(i).cd;
    
    [a,b] = max(CLCD);
    [c,d] = min(CLCD);
    matCL2CD(i,:) = [a,b,c,d];
%     num = size(DataStruct(i).morph,1);
    vecNumMorph(i) = size(DataStruct(i).morph,1);
    Index = randperm(vecNumMorph(i),15);
    matMorph = [ matMorph; DataStruct(i).morph(Index,:)];
    matMaAoAH = [matMaAoAH;  repmat([DataStruct(i).Ma, DataStruct(i).AoA, DataStruct(i).H],15,1)];
    matCl = [ matCl; DataStruct(i).cl(Index,:)];
    matCd = [ matCd; DataStruct(i).cd(Index,:)];
    matSref = [matSref; DataStruct(i).sref(Index,:)];
end

X  = [matMaAoAH,matMorph];
Y = matCd;

X(:,1) = ( X(:,1) - 7 )/10.5;
X(:,2) = ( X(:,2) - 10 )/10;
X(:,3) = ( X(:,3) - 30 )/30;
X(:,4) = ( X(:,4) )/(pi/6);
X(:,5) = X(:,5);
X(:,6) = X(:,6);

index = 1:360;
select = randperm(360,36);
index(select)=[];

X_test = X(select,:);
Y_test = Y(select,:);
X = X(index,:);
Y = Y(index,:);

variable_number = size(X,2);

[predict_function,kriging_model] = interpKrigingPreModel(X,Y);

Y_test_pred = predict_function(X_test);
R2=1-sum((Y_test-Y_test_pred).^2)/sum((mean(Y_test)-Y_test).^2)

function [predict_function,kriging_model] = interpKrigingPreModel(X,Y,hyp)
% nomalization method is grassian
% add multi x_predict input support
% prepare model,optimal theta and calculation parameter
% X,Y are x_number x variable_number matrix
% aver_X,stdD_X is 1 x x_number matrix
% theta beta gama sigma_sq is normalizede,so predict y is normalize
% theta = exp(hyp)
%
% input initial data X,Y,which are real data
%
% output is a kriging model,include predict_function...
% X,Y,base_function_list
%
% Copyright 2023.2 Adel
%
[x_number,variable_number] = size(X);
if nargin < 3
    hyp = 0;
end

% normalize data
aver_X = mean(X);
stdD_X = std(X);
aver_Y = mean(Y);
stdD_Y = std(Y);
index__ = find(stdD_X == 0);
if  ~isempty(index__),stdD_X(index__) = 1; end
index__ = find(stdD_Y == 0);
if  ~isempty(index__),stdD_Y(index__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;
Y_nomlz = (Y-aver_Y)./stdD_Y;

% initial X_dis_sq
X_dis_sq = zeros(x_number,x_number,variable_number);
for variable_index = 1:variable_number
    X_dis_sq(:,:,variable_index) = ...
        (X_nomlz(:,variable_index)-X_nomlz(:,variable_index)').^2;
end

% regression function define
% notice reg_function process no normalization data
% reg_function = @(X) regZero(X);
reg_function = @(X) regLinear(X);

% calculate reg
fval_reg_nomlz = (reg_function(X)-aver_Y)./stdD_Y;

% optimal to get hyperparameter
fmincon_option = optimoptions('fmincon','Display','none',...
    'OptimalityTolerance',1e-2,...
    'FiniteDifferenceStepSize',1e-5,...,
    'MaxIterations',10,'SpecifyObjectiveGradient',false);
low_bou_hyp = -3;
up_bou_hyp = 3;
prob_NLL_function = @(hyp) probNLLKriging...
    (X_dis_sq,Y_nomlz,x_number,variable_number,hyp,fval_reg_nomlz);

% [fval,gradient] = prob_NLL_function(hyp)
% [~,gradient_differ] = differ(prob_NLL_function,hyp)

% drawFunction(prob_NLL_function,low_bou_hyp,up_bou_hyp);

hyp = fmincon...
    (prob_NLL_function,hyp,[],[],[],[],low_bou_hyp,up_bou_hyp,[],fmincon_option);

% get parameter
[covariance,inv_covariance,~,beta,sigma_sq] = interpKriging...
    (X_dis_sq,Y_nomlz,x_number,variable_number,exp(hyp),fval_reg_nomlz);
gama = inv_covariance*(Y_nomlz-fval_reg_nomlz*beta);
FTRF = fval_reg_nomlz'*inv_covariance*fval_reg_nomlz;

% initialization predict function
predict_function = @(X_predict) interpKrigingPredictor...
    (X_predict,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_number,variable_number,exp(hyp),beta,gama,sigma_sq,...
    inv_covariance,fval_reg_nomlz,FTRF,reg_function);

kriging_model.X = X;
kriging_model.Y = Y;
kriging_model.fval_regression = fval_reg_nomlz;
kriging_model.covariance = covariance;
kriging_model.inv_covariance = inv_covariance;

kriging_model.hyp = hyp;
kriging_model.beta = beta;
kriging_model.gama = gama;
kriging_model.sigma_sq = sigma_sq;
kriging_model.aver_X = aver_X;
kriging_model.stdD_X = stdD_X;
kriging_model.aver_Y = aver_Y;
kriging_model.stdD_Y = stdD_Y;

kriging_model.predict_function = predict_function;

% abbreviation:
% num: number,pred: predict,vari: variable,hyp: hyper parameter
% NLL: negative log likelihood
    function [fval,gradient] = probNLLKriging...
            (X_dis_sq,Y,x_num,vari_num,hyp,F_reg)
        % function to minimize sigma_sq
        %
        theta = exp(hyp);
        [cov,inv_cov,L,~,sigma2,inv_FTRF,Y_Fmiu] = interpKriging...
            (X_dis_sq,Y,x_num,vari_num,theta,F_reg);

        % calculation negative log likelihood
        fval = x_num/2*log(sigma2)+sum(log(diag(L)));

        % calculate gradient
        if nargout > 1
            % gradient
            dcov_dtheta = zeros(x_num,x_num);
            for vari_index = 1:vari_num
                dcov_dtheta = dcov_dtheta + X_dis_sq(:,:,vari_index);
            end
            dcov_dtheta = -dcov_dtheta.*cov*theta/vari_num;

            dinv_cov_dtheta = ...
                -inv_cov*dcov_dtheta*inv_cov;

            dinv_FTRF_dtheta = -inv_FTRF*...
                (F_reg'*dinv_cov_dtheta*F_reg)*...
                inv_FTRF;

            dmiu_dtheta = dinv_FTRF_dtheta*(F_reg'*inv_cov*Y)+...
                inv_FTRF*(F_reg'*dinv_cov_dtheta*Y);

            dY_Fmiu_dtheta = -F_reg*dmiu_dtheta;

            dsigma2_dtheta = (dY_Fmiu_dtheta'*inv_cov*Y_Fmiu+...
                Y_Fmiu'*dinv_cov_dtheta*Y_Fmiu+...
                Y_Fmiu'*inv_cov*dY_Fmiu_dtheta)/x_num;

            dlnsigma2_dtheta = 1/sigma2*dsigma2_dtheta;

            dlndetR = trace(inv_cov*dcov_dtheta);

            gradient = x_num/2*dlnsigma2_dtheta+0.5*dlndetR;

        end
    end

    function [cov,inv_cov,L,beta,sigma_sq,inv_FTRF,Y_Fmiu] = interpKriging...
            (X_dis_sq,Y,x_num,vari_num,theta,F_reg)
        % kriging interpolation kernel function
        % Y(x) = beta+Z(x)
        %
        cov = zeros(x_num,x_num);
        for vari_index = 1:vari_num
            cov = cov+X_dis_sq(:,:,vari_index)*theta;
        end
        cov = exp(-cov/vari_num)+eye(x_num)*1e-6;

        % coefficient calculation
        L = chol(cov)';
        inv_L = L\eye(x_num);
        inv_cov = inv_L'*inv_L;
        inv_FTRF = (F_reg'*inv_cov*F_reg)\eye(size(F_reg,2));

        % analytical solve sigma_sq
        beta = inv_FTRF*(F_reg'*inv_cov*Y);
        Y_Fmiu = Y-F_reg*beta;
        sigma_sq = (Y_Fmiu'*inv_cov*Y_Fmiu)/x_num;

    end

    function [Y_pred,Var_pred] = interpKrigingPredictor...
            (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,theta,beta,gama,sigma_sq,...
            inv_cov,fval_reg_nomlz,FTRF,reg_function)
        % kriging interpolation predict function
        % output the predict value and predict variance
        %
        [x_pred_num,~] = size(X_pred);
        fval_reg_pred = reg_function(X_pred);

        % normalize data
        X_pred_nomlz = (X_pred-aver_X)./stdD_X;
        fval_reg_pred_nomlz = (fval_reg_pred-aver_Y)./stdD_Y;

        % predict covariance
        cov_pred = zeros(x_num,x_pred_num);
        for vari_index = 1:vari_num
            cov_pred = cov_pred+...
                (X_nomlz(:,vari_index)-X_pred_nomlz(:,vari_index)').^2*theta;
        end
        cov_pred = exp(-cov_pred/vari_num);

        % predict base fval

        Y_pred = fval_reg_pred_nomlz*beta+cov_pred'*gama;

        % predict variance
        u__ = fval_reg_nomlz'*inv_cov*cov_pred-fval_reg_pred_nomlz';
        Var_pred = sigma_sq*...
            (1+u__'/FTRF*u__+...
            -cov_pred'*inv_cov*cov_pred);

        % normalize data
        Y_pred = Y_pred*stdD_Y+aver_Y;
        Var_pred = diag(Var_pred)*stdD_Y*stdD_Y;
    end

    function F_reg = regZero(X)
        % zero order base funcion
        %
        F_reg = ones(size(X,1),1); % zero
    end

    function F_reg = regLinear(X)
        % first order base funcion
        %
        F_reg = [ones(size(X,1),1),X]; % linear
    end
end

