clc;
clear;
close all hidden;

% load('MFK.mat')
%
% K_fold = 5;
%
% HF_select = randi(100,1,size(matPntHF,1)/K_fold*(K_fold-1));
% HF_check = 1:100;
% HF_check(HF_select) = [];
%
% X = matPntHF(HF_select,:);
% Y = matObjHF(HF_select,1);
%
% [predict_function,kriging_model] = interpGaussPreModel(X,Y);
% check_error_list = zeros(size(matPntHF,1)/K_fold,1);
% for check_index = 1:(size(matPntHF,1)/K_fold)
%     x_index = HF_check(check_index);
%     check_error_list(check_index) = (predict_function(matPntHF(x_index,:))-matObjHF(x_index,1));
% end
% disp(['max error: ',num2str(max(check_error_list))]);

load('PK.mat')

[predict_function,kriging_model] = interpGaussianPreModel(X,Y);
figure_handle = figure(1);
interpVisualize(kriging_model,low_bou,up_bou,[],[],[],figure_handle)

function [predict_function,GPR_model] = interpGaussianPreModel...
    (X,Y,hyp,sigma_obv)
% generate gaussian process regression model,version 0
% X,Y are x_number x variable_number matrix
% aver_X,stdD_X is 1 x x_number matrix
% theta beta gama sigma_sq is normalizede,so predict y is normalize
%
% input:
% X,Y (initial data,which are real data)
% theta (hyperparameter,len,eta)
%
% output:
% predict_function,GPR_model (a gauss process regression model)
%
% reference: [1] RASMUSSEN C E, WILLIAMS C K I. Gaussian Processes for
% Machine Learning [M/OL]. 2005
% [https://doi.org/10.7551/mitpress/3206.001.0001].
%
% Copyright 2023.2 Adel
%
[x_number,variable_number] = size(X);
if nargin < 4 || isempty(sigma_obv)
    sigma_obv = 0;
    if nargin < 3 || isempty(hyp)
        hyp = [ones(1,variable_number)*0.5,10];
    end
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
X_nomlz = (X-repmat(aver_X,x_number,1))./repmat(stdD_X,x_number,1);
Y_nomlz = (Y-repmat(aver_Y,x_number,1))./repmat(stdD_Y,x_number,1);

% initial X_dis_sq
X_dis_sq = zeros(x_number,x_number,variable_number);
for rank = 1:x_number
    for colume = 1:rank-1
        X_dis_sq(rank,colume,:) = X_dis_sq(colume,rank,:);
    end
    for colume = rank:x_number
        X_dis_sq(rank,colume,:) = (X_nomlz(rank,:)-X_nomlz(colume,:)).^2;
    end
end

% optimal to get hyperparameter
fmincon_option = optimoptions('fmincon','Display','none',...
    'OptimalityTolerance',1e-2,...
    'FiniteDifferenceStepSize',1e-5,...,
    'MaxIterations',10,'SpecifyObjectiveGradient',false);
low_bou_hyp = -3*ones(1,variable_number+1);
up_bou_hyp = 3*ones(1,variable_number+1);
pro_NLL_function = @(hyp) probNLLGaussian...
    (X_dis_sq,Y,x_number,variable_number,hyp,sigma_obv);

% [fval,gradient] = pro_NLL_function(hyp)
% [~,gradient_differ] = differ(pro_NLL_function,hyp)

[hyp,~,~,~] = fmincon...
    (pro_NLL_function,hyp,[],[],[],[],low_bou_hyp,up_bou_hyp,[],fmincon_option);

% obtian covariance matrix
[covariance,inv_covariance,~,~] = getCovariance...
    (X_dis_sq,x_number,variable_number,exp(hyp(1:variable_number)),exp(hyp(variable_number+1)),sigma_obv);

% initialization predict function
predict_function = @(X_pred) interpGaussianPredictor...
    (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_number,variable_number,Y_nomlz,exp(hyp(1:variable_number)),exp(hyp(variable_number+1)),sigma_obv,...
    inv_covariance);

GPR_model.X = X;
GPR_model.Y = Y;
GPR_model.covariance = covariance;
GPR_model.inv_covariance = inv_covariance;

GPR_model.hyp = hyp;
GPR_model.aver_X = aver_X;
GPR_model.stdD_X = stdD_X;
GPR_model.aver_Y = aver_Y;
GPR_model.stdD_Y = stdD_Y;

GPR_model.predict_function = predict_function;

% abbreviation:
% num: number,pred: predict,vari: variable,hyp: hyper parameter
% NLL: negative log likelihood,hyp: hyperparameter
% obv: observe
    function [fval,gradient] = probNLLGaussian...
            (X_dis_sq,Y,x_num,vari_num,hyp,sigma_obv)
        % hyperparameter is [len,sigma_sq]
        % notice to normalize X_dis for difference variable number
        % X_dis_sq will be divide by vari_num
        %
        len = exp(hyp(1:vari_num));
        sigma_sq = exp(hyp(vari_num+1));

        % obtian covariance matrix
        [cov,inv_cov,L,~] = getCovariance...
            (X_dis_sq,x_num,vari_num,len,sigma_sq,sigma_obv);

        % calculation negative log likelihood
        fval = 0.5*Y'*inv_cov*Y+sum(log(diag(L)))+x_num/2*log(2*pi);

        if nargout > 1
            % get gradient
            gradient = zeros(vari_num+1,1);

            % var: len_1,len_2,...,len_n,sigma_sq
%             dcov_dvar = zeros(x_number,x_number,variable_number+1);
%             dinv_K_dvar = zeros(x_number,x_number,variable_number+1);
            % len
            for vari_index = 1:vari_num
                dcov_dlen = -cov.*X_dis_sq(:,:,vari_index)*len(vari_index)/vari_num;
                dinv_cov_dlen = -inv_cov*dcov_dlen*inv_cov;

%                 dcov_dvar(:,:,len_index__) = dcov_dlen;
%                 dinv_K_dvar(:,:,len_index__) = dinv_K_dlen;

                gradient(vari_index) = 0.5*Y'*dinv_cov_dlen*Y+...
                    0.5*trace(inv_cov*dcov_dlen);
            end
            % sigma_sq
            dcov_dsigma_sq = cov;
            dinv_cov_dsigma_sq = -inv_cov*dcov_dsigma_sq*inv_cov;

%             dcov_dvar(:,:,end) = dcov_dsigma_sq;
%             dinv_K_dvar(:,:,end) = dinv_cov_deta;

            gradient(vari_num+1) = 0.5*Y'*dinv_cov_dsigma_sq*Y+...
                0.5*trace(inv_cov*dcov_dsigma_sq);
        end

%         if nargout > 2
%             % get hessian
%             % var: len1,len2,... eta
%             hessian = zeros(variable_number+1);
%             for len_i = 1:variable_number
%                 for len_j = 1:len_i-1
%                     hessian(len_i,len_j) = hessian(len_j,len_i);
%                 end
%                 len_j = len_i;
%                 ddK_dii = dK_dvar(:,:,len_i).*X_dis_sq(:,:,len_j)/len__(len_j)^3+...
%                     eta_exp_dis__.*(-3*X_dis_sq(:,:,len_j)/len__(len_j)^4);
%                 ddinv_K_dij = -dinv_K_dvar(:,:,len_j)*dK_dvar(:,:,len_i)*inv_cov+...
%                     -inv_cov*ddK_dii*inv_cov+...
%                     -inv_cov*dK_dvar(:,:,len_i)*dinv_K_dvar(:,:,len_j);
%                 dddetK_dij = ddetK_dvar(:,:,len_j)*trace(inv_cov*dK_dvar(:,:,len_i))+...
%                     detK*trace(dinv_K_dvar(:,:,len_j)*dK_dvar(:,:,len_i)+inv_cov*ddK_dii);
%                 hessian(len_i,len_j) = -0.5*Y'*ddinv_K_dij*Y-0.5/detK*dddetK_dij+...
%                     0.5/detK^2*ddetK_dvar(:,:,len_j)*ddetK_dvar(:,:,len_i);
% 
%                 for len_j = len_i+1:variable_number
%                     ddK_dij = dK_dvar(:,:,len_i).*X_dis_sq(:,:,len_j)/len__(len_j)^3;
%                     ddinv_K_dij = -dinv_K_dvar(:,:,len_j)*dK_dvar(:,:,len_i)*inv_cov+...
%                         -inv_cov*ddK_dij*inv_cov+...
%                         -inv_cov*dK_dvar(:,:,len_i)*dinv_K_dvar(:,:,len_j);
%                     dddetK_dij = ddetK_dvar(:,:,len_j)*trace(inv_cov*dK_dvar(:,:,len_i))+...
%                         detK*trace(dinv_K_dvar(:,:,len_j)*dK_dvar(:,:,len_i)+inv_cov*ddK_dij);
%                     hessian(len_i,len_j) = -0.5*Y'*ddinv_K_dij*Y-0.5/detK*dddetK_dij+...
%                         0.5/detK^2*ddetK_dvar(:,:,len_j)*ddetK_dvar(:,:,len_i);
%                 end
% 
%                 len_j = variable_number+1; % eta
%                 ddK_dieta = exp_dis__.*X_dis_sq(:,:,len_i)/len__(len_i)^3;
%                 ddinv_K_dij = -dinv_K_dvar(:,:,len_j)*dK_dvar(:,:,len_i)*inv_cov+...
%                     -inv_cov*ddK_dieta*inv_cov+...
%                     -inv_cov*dK_dvar(:,:,len_i)*dinv_K_dvar(:,:,len_j);
%                 dddetK_dij = ddetK_dvar(:,:,len_j)*trace(inv_cov*dK_dvar(:,:,len_i))+...
%                     detK*trace(dinv_K_dvar(:,:,len_j)*dK_dvar(:,:,len_i)+inv_cov*ddK_dieta);
%                 hessian(len_i,len_j) = -0.5*Y'*ddinv_K_dij*Y-0.5/detK*dddetK_dij+...
%                     0.5/detK^2*ddetK_dvar(:,:,len_j)*ddetK_dvar(:,:,len_i);
%             end
% 
%             % eta eta
%             len_i = variable_number+1;
%             for len_j = 1:len_i-1
%                 hessian(len_i,len_j) = hessian(len_j,len_i);
%             end
%             len_j = len_i;
%             ddK_detaeta = 0;
%             ddinv_K_dij = -dinv_K_dvar(:,:,len_j)*dK_dvar(:,:,len_i)*inv_cov+...
%                 -inv_cov*ddK_detaeta*inv_cov+...
%                 -inv_cov*dK_dvar(:,:,len_i)*dinv_K_dvar(:,:,len_j);
%             dddetK_dij = ddetK_dvar(:,:,len_j)*trace(inv_cov*dK_dvar(:,:,len_i))+...
%                 detK*trace(dinv_K_dvar(:,:,len_j)*dK_dvar(:,:,len_i)+inv_cov*ddK_detaeta);
%             hessian(len_i,len_j) = -0.5*Y'*ddinv_K_dij*Y-0.5/detK*dddetK_dij+...
%                 0.5/detK^2*ddetK_dvar(:,:,len_j)*ddetK_dvar(:,:,len_i);
% 
%         end

    end

    function [cov,inv_cov,L,exp_dis] = getCovariance...
            (X_dis_sq,x_num,vari_num,len,sigma_sq,sigma_obv)
        % obtain covariance of x
        % notice to normalize X_dis for difference variable number
        % X_dis_sq will be divide by vari_num
        % cov = K+sigma_obv*I
        %
        exp_dis = zeros(x_num,x_num);
        for vari_index = 1:vari_num
            exp_dis = exp_dis+X_dis_sq(:,:,vari_index)*len(vari_index);
        end
        exp_dis = exp(-exp_dis/vari_num)+eye(x_num)*1e-6;

        cov = sigma_sq*exp_dis+sigma_obv*eye(x_num);
        L = chol(cov)';
        inv_L = L\eye(x_num);
        inv_cov = inv_L'*inv_L;
    end

    function [Y_pred,Var_pred] = interpGaussianPredictor...
            (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,Y,len,sigma_sq,sigma_obv,...
            inv_cov)
        % gaussian process regression predict function
        % output the predict value and predict variance
        %
        [x_pred_num,~] = size(X_pred);
        
        % normalize data
        X_pred_nomlz = (X_pred-aver_X)./stdD_X;
        
        % predict covariance
        cov_pred = zeros(x_num,x_pred_num);
        for vari_index = 1:vari_num
            cov_pred = cov_pred+...
                (X_nomlz(:,vari_index)-X_pred_nomlz(:,vari_index)').^2*len(vari_index);
        end
        cov_pred = sigma_sq*exp(-cov_pred/vari_num);

        % get miu and variance of predict x
        Y_pred = cov_pred'*inv_cov*Y;
        Var_pred = (sigma_sq+sigma_obv)*eye(x_pred_num)-...
            cov_pred'*inv_cov*cov_pred;

        % normalize data
        Y_pred = Y_pred*stdD_Y+aver_Y;
        Var_pred = diag(Var_pred)*stdD_Y*stdD_Y;
    end
end
