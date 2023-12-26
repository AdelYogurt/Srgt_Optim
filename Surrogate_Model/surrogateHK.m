clc;
clear;
close all hidden;

% load('MFK.mat')
% 
% K_fold = 5;
% 
% HF_select = randi(100, 1, size(matPntHF, 1)/K_fold*(K_fold-1));
% HF_check = 1:100;
% HF_check(HF_select) = [];
% LF_select = randi(400, 1, size(matPntLF1, 1)/K_fold*(K_fold-1));
% 
% XHF = matPntHF(HF_select, :);
% YHF = matObjHF(HF_select, 1);
% 
% XLF = matPntLF1(LF_select, :);
% YLF = matObjLF1(LF_select, 1);
% 
% [predict_function, MK_model] = interpHieraKrigingPreModel...
%     (XHF, YHF, XLF, YLF);
% 
% check_error_list = zeros(size(matPntHF, 1)/K_fold, 1);
% for check_index = 1:(size(matPntHF, 1)/K_fold)
%     x_index = HF_check(check_index);
%     check_error_list(check_index) = (predict_function(matPntHF(x_index, :))-matObjHF(x_index, 1));
% end
% disp(['max error: ', num2str(max(check_error_list))]);

load('Forrester.mat')
[predict_function_HK, HK_model] = interpHieraKrigingPreModel...
    (XHF, YHF, [], XLF, YLF, []);
[predict_function, K_model] = interpKrigingPreModel(XHF, YHF);
[y_HK] = predict_function_HK(x);
[y] = predict_function(x);
line(x, y_real, 'Color', 'k', 'LineStyle', '-')
line(x, y_real_low, 'Color', 'k', 'LineStyle', '--');
line(x, y_HK, 'Color', 'b', 'LineStyle', '-')
line(x, y, 'Color', 'b', 'LineStyle', '--');
legend({'singleForresterObject', 'singleForresterObjectLow', 'HK', 'K'})

% load('2DMF.mat');
% [predict_function_HK, HK_model] = interpHieraKrigingPreModel(XHF, YHF, [], XLF, YLF, []);
% interpVisualize(HK_model, low_bou, up_bou);

% function [predict_function, HK_model] = interpHieraKrigingPreModel...
%     (XHF, YHF, varargin)
% % construct Hierarchical Kriging model
% % XHF, YHF are x_HF_number x variable_number matrix
% % XLF, YLF are x_LF_number x variable_number matrix
% % aver_X, stdD_X is 1 x x_HF_number matrix
% % theta beta gama sigma_sq is normalizede, so predict y is normalize
% % hyp: hyp_HF, hyp_LF
% % notice theta = exp(hyp)
% %
% % input:
% % XHF, YHF, hyp_HF(can be []), XLF, YLF, hyp_LF(can be [])
% % XHF, YHF, hyp_HF(can be []), LF_model
% %
% % output:
% % predict_function, HK_model
% %
% % reference: [1] HAN Z-H, GöRTZ S. Hierarchical Kriging Model for
% % Variable-Fidelity Surrogate Modeling [J]. AIAA Journal, 2012, 50(9):
% % 1885-96.
% %
% % Copyright 2023.2 Adel
% %
% [x_HF_number, variable_number] = size(XHF);
% switch nargin
%     case 4
%         hyp_HF = varargin{1};
%         LF_model = varargin{2};
% 
%         % check whether LF model exist predict_function
%         if ~isfield(LF_model, 'predict_function')
%             error('interpHieraKrigingPreModel: low fidelity lack predict function');
%         end
%     case 6
%         hyp_HF = varargin{1};
%         XLF = varargin{2};
%         YLF = varargin{3};
%         hyp_LF = varargin{4};
% 
%         [x_LF_number, variable_number] = size(XLF);
% 
%         if isempty(hyp_LF)
%             hyp_LF = ones(1, variable_number);
%         end
% 
%         % first step
%         % construct low fidelity model
% 
%         % normalize data
%         aver_X = mean(XLF);
%         stdD_X = std(XLF);
%         aver_Y = mean(YLF);
%         stdD_Y = std(YLF);
%         index__ = find(stdD_X == 0);
%         if  ~isempty(index__), stdD_X(index__) = 1; end
%         index__ = find(stdD_Y == 0);
%         if  ~isempty(index__), stdD_Y(index__) = 1; end
% 
%         XLF_nomlz = (XLF-aver_X)./stdD_X;
%         YLF_nomlz = (YLF-aver_Y)./stdD_Y;
% 
%         % initial X_dis_sq
%         XLF_dis_sq = zeros(x_LF_number, x_LF_number, variable_number);
%         for variable_index = 1:variable_number
%             XLF_dis_sq(:, :, variable_index) = ...
%                 (XLF_nomlz(:, variable_index)-XLF_nomlz(:, variable_index)').^2;
%         end
% 
%         % regression function define
%         % notice reg_function process no normalization data
%         % reg_function = @(X) regZero(X);
%         reg_function = @(X) regLinear(X);
% 
%         % calculate reg
%         fval_reg_LF = reg_function(XLF);
%         fval_reg_nomlz_LF = (fval_reg_LF-aver_Y)./stdD_Y;
% 
%         % optimal to get hyperparameter
%         low_bou_hyp = -3*ones(1, variable_number);
%         up_bou_hyp = 3*ones(1, variable_number);
%         fmincon_option = optimoptions('fmincon', 'Display', 'none', ...
%             'OptimalityTolerance', 1e-2, ...
%             'FiniteDifferenceStepSize', 1e-5, ..., 
%             'MaxIterations', 10, 'SpecifyObjectiveGradient', true);
%         object_function_hyp = @(hyp) probNLLKriging...
%             (XLF_dis_sq, YLF_nomlz, x_LF_number, variable_number, hyp, fval_reg_nomlz_LF);
% 
%         % [fval, gradient] = object_function_hyp(hyp_LF)
%         % [~, gradient_differ] = differ(object_function_hyp, hyp_LF)
% 
%         hyp_LF = fmincon...
%             (object_function_hyp, hyp_LF, [], [], [], [], low_bou_hyp, up_bou_hyp, [], fmincon_option);
% 
%         % get parameter
%         [cov_LF, inv_cov_LF, beta_LF, sigma_sq_LF] = interpKriging...
%             (XLF_dis_sq, YLF_nomlz, x_LF_number, variable_number, exp(hyp_LF), fval_reg_nomlz_LF);
%         gama_LF = inv_cov_LF*(YLF_nomlz-fval_reg_nomlz_LF*beta_LF);
%         FTRF_LF = fval_reg_nomlz_LF'*inv_cov_LF*fval_reg_nomlz_LF;
% 
%         % initialization predict function
%         predict_function_LF = @(X_predict) interpKrigingPredictor...
%             (X_predict, XLF_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
%             x_LF_number, variable_number, exp(hyp_LF), beta_LF, gama_LF, sigma_sq_LF, ...
%             inv_cov_LF, fval_reg_nomlz_LF, FTRF_LF, reg_function);
% 
%         LF_model.X = XLF;
%         LF_model.Y = YLF;
%         LF_model.fval_regression = fval_reg_LF;
%         LF_model.covariance = cov_LF;
%         LF_model.inv_covariance = inv_cov_LF;
% 
%         LF_model.hyp = hyp_LF;
%         LF_model.beta = beta_LF;
%         LF_model.gama = gama_LF;
%         LF_model.sigma_sq = sigma_sq_LF;
%         LF_model.aver_X = aver_X;
%         LF_model.stdD_X = stdD_X;
%         LF_model.aver_Y = aver_Y;
%         LF_model.stdD_Y = stdD_Y;
% 
%         LF_model.predict_function = predict_function_LF;
%     otherwise
%         error('interpHieraKrigingPreModel: error input');
% end
% HK_model.LF_model = LF_model;
% 
% % second step
% % construct hierarchical model
% if isempty(hyp_HF)
%     hyp_HF = ones(1, variable_number);
% end
% 
% predict_function_LF = LF_model.predict_function;
% 
% % normalize data
% aver_X = mean(XHF);
% stdD_X = std(XHF);
% aver_Y = mean(YHF);
% stdD_Y = std(YHF);
% index__ = find(stdD_X == 0);
% if  ~isempty(index__), stdD_X(index__) = 1; end
% index__ = find(stdD_Y == 0);
% if  ~isempty(index__), stdD_Y(index__) = 1; end
% XHF_nomlz = (XHF-aver_X)./stdD_X;
% YHF_nomlz = (YHF-aver_Y)./stdD_Y;
% 
% % initial X_dis_sq
% XHF_dis_sq = zeros(x_HF_number, x_HF_number, variable_number);
% for variable_index = 1:variable_number
%     XHF_dis_sq(:, :, variable_index) = ...
%         (XHF_nomlz(:, variable_index)-XHF_nomlz(:, variable_index)').^2;
% end
% 
% % evaluate low fidelty predict value in high fidelity point as base fval
% reg_function = @(X) predict_function_LF(X);
% 
% % calculate reg
% fval_reg_HF = reg_function(XHF);
% fval_reg_nomlz_HF = (fval_reg_HF-aver_Y)./stdD_Y;
% 
% % optimal to get hyperparameter
% object_function_hyp = @(hyp) probNLLKriging...
%     (XHF_dis_sq, YHF_nomlz, x_HF_number, variable_number, hyp, fval_reg_nomlz_HF);
% 
% % [fval, gradient] = object_function_hyp(hyp_HF)
% % [~, gradient_differ] = differ(object_function_hyp, hyp_HF)
% 
% % drawFunction(object_function_hyp, low_bou_hyp, up_bou_hyp);
% 
% hyp_HF = fmincon...
%     (object_function_hyp, hyp_HF, [], [], [], [], low_bou_hyp, up_bou_hyp, [], fmincon_option);
% 
% % calculate covariance and other parameter
% [cov_HF, inv_cov_HF, beta_HF, sigma_sq_HF] = interpKriging...
%     (XHF_dis_sq, YHF_nomlz, x_HF_number, variable_number, exp(hyp_HF), fval_reg_nomlz_HF);
% gama_HF = inv_cov_HF*(YHF_nomlz-fval_reg_nomlz_HF*beta_HF);
% FTRF_HF = fval_reg_nomlz_HF'*inv_cov_HF*fval_reg_nomlz_HF;
% 
% % initialization predict function
% predict_function = @(X_predict) interpKrigingPredictor...
%     (X_predict, XHF_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
%     x_HF_number, variable_number, exp(hyp_HF), beta_HF, gama_HF, sigma_sq_HF, ...
%     inv_cov_HF, fval_reg_nomlz_HF, FTRF_HF, reg_function);
% 
% HK_model.X = {XHF, XLF};
% HK_model.Y = {YHF, YLF};
% HK_model.fval_regression = fval_reg_HF;
% HK_model.covariance = cov_HF;
% HK_model.inv_covariance = inv_cov_HF;
% 
% HK_model.hyp = hyp_HF;
% HK_model.beta = beta_HF;
% HK_model.gama = gama_HF;
% HK_model.sigma_sq = sigma_sq_HF;
% 
% HK_model.aver_X = aver_X;
% HK_model.stdD_X = stdD_X;
% HK_model.aver_Y = aver_Y;
% HK_model.stdD_Y = stdD_Y;
% 
% HK_model.predict_function = predict_function;
% 
% % abbreviation:
% % num: number, pred: predict, vari: variable, hyp: hyper parameter
% % NLL: negative log likelihood
%     function [fval, gradient] = probNLLKriging...
%             (X_dis_sq, Y, x_num, vari_num, hyp, F_reg)
%         % function to minimize sigma_sq
%         %
%         theta = exp(hyp);
%         [cov, inv_cov, ~, sigma2, inv_FTRF, Y_Fmiu] = interpKriging...
%             (X_dis_sq, Y, x_num, vari_num, theta, F_reg);
%         
%         % calculation negative log likelihood
%         L = chol(cov)';
%         fval = x_num/2*log(sigma2)+sum(log(diag(L)));
% 
%         % calculate gradient
%         if nargout > 1
%             % gradient
%             gradient = zeros(vari_num, 1);
%             for vari_index = 1:vari_num
%                 dcov_dtheta = -(X_dis_sq(:, :, vari_index).*cov)*theta(vari_index)/vari_num;
% 
%                 dinv_cov_dtheta = ...
%                     -inv_cov*dcov_dtheta*inv_cov;
% 
%                 dinv_FTRF_dtheta = -inv_FTRF*...
%                     (F_reg'*dinv_cov_dtheta*F_reg)*...
%                     inv_FTRF;
%                 
%                 dmiu_dtheta = dinv_FTRF_dtheta*(F_reg'*inv_cov*Y)+...
%                     inv_FTRF*(F_reg'*dinv_cov_dtheta*Y);
%                 
%                 dY_Fmiu_dtheta = -F_reg*dmiu_dtheta;
% 
%                 dsigma2_dtheta = (dY_Fmiu_dtheta'*inv_cov*Y_Fmiu+...
%                     Y_Fmiu'*dinv_cov_dtheta*Y_Fmiu+...
%                     Y_Fmiu'*inv_cov*dY_Fmiu_dtheta)/x_num;
%                 
%                 dlnsigma2_dtheta = 1/sigma2*dsigma2_dtheta;
% 
%                 dlndetR = trace(inv_cov*dcov_dtheta);
% 
%                 gradient(vari_index) = x_num/2*dlnsigma2_dtheta+0.5*dlndetR;
%             end
%         end
%     end
%     function [cov, inv_cov, beta, sigma_sq, inv_FTRF, Y_Fmiu] = interpKriging...
%             (X_dis_sq, Y, x_num, vari_num, theta, F_reg)
%         % kriging interpolation kernel function
%         % Y(x) = beta+Z(x)
%         %
%         cov = zeros(x_num, x_num);
%         for vari_index = 1:vari_num
%             cov = cov+X_dis_sq(:, :, vari_index)*theta(vari_index);
%         end
%         cov = exp(-cov/vari_num)+eye(x_num)*1e-6;
% 
%         % coefficient calculation
%         inv_cov = cov\eye(x_num);
%         inv_FTRF = (F_reg'*inv_cov*F_reg)\eye(size(F_reg, 2));
% 
%         % basical bias
%         beta = inv_FTRF*(F_reg'*inv_cov*Y);
%         Y_Fmiu = Y-F_reg*beta;
%         sigma_sq = (Y_Fmiu'*inv_cov*Y_Fmiu)/x_num;
%         
%     end
%     function [Y_pred, Var_pred] = interpKrigingPredictor...
%             (X_pred, X_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
%             x_num, vari_num, theta, beta, gama, sigma_sq, ...
%             inv_cov, fval_reg, FTRF, reg_function)
%         % kriging interpolation predict function
%         % input predict_x and kriging model
%         % predict_x is row vector
%         % output the predict value
%         %
%         [x_pred_num, ~] = size(X_pred);
%         fval_reg_pred = reg_function(X_pred);
% 
%         % normalize data
%         X_pred_nomlz = (X_pred-aver_X)./stdD_X;
%         fval_reg_pred_nomlz = (fval_reg_pred-aver_Y)./stdD_Y;
%         
%         % predict covariance
%         predict_cov = zeros(x_num, x_pred_num);
%         for vari_index = 1:vari_num
%             predict_cov = predict_cov+...
%                 (X_nomlz(:, vari_index)-X_pred_nomlz(:, vari_index)').^2*theta(vari_index);
%         end
%         predict_cov = exp(-predict_cov/vari_num);
% 
%         % predict base fval
%         
%         Y_pred = fval_reg_pred_nomlz*beta+predict_cov'*gama;
%         
%         % predict variance
%         u__ = fval_reg'*inv_cov*predict_cov-fval_reg_pred_nomlz';
%         Var_pred = sigma_sq*...
%             (1+u__'/FTRF*u__+...
%             -predict_cov'*inv_cov*predict_cov);
%         
%         % normalize data
%         Y_pred = Y_pred*stdD_Y+aver_Y;
%         Var_pred = diag(Var_pred)*stdD_Y*stdD_Y;
%     end
%     function F_reg = regZero(X)
%         % zero order base funcion
%         %
%         F_reg = ones(size(X, 1), 1); % zero
%     end
%     function F_reg = regLinear(X)
%         % first order base funcion
%         %
%         F_reg = [ones(size(X, 1), 1), X]; % linear
%     end
% end

function [predict_function, HK_model] = interpHieraKrigingPreModel...
    (XHF, YHF, varargin)
% construct Hierarchical Kriging model
% XHF, YHF are x_HF_number x variable_number matrix
% XLF, YLF are x_LF_number x variable_number matrix
% aver_X, stdD_X is 1 x x_HF_number matrix
% theta beta gama sigma_sq is normalizede, so predict y is normalize
% hyp: hyp_HF, hyp_LF
% notice theta = exp(hyp)
%
% input:
% XHF, YHF, hyp_HF(can be []), XLF, YLF, hyp_LF(can be [])
% XHF, YHF, hyp_HF(can be []), LF_model
%
% output:
% predict_function, HK_model
%
% reference: [1] HAN Z-H, GöRTZ S. Hierarchical Kriging Model for
% Variable-Fidelity Surrogate Modeling [J]. AIAA Journal, 2012, 50(9):
% 1885-96.
%
% Copyright 2023.2 Adel
%
[x_HF_number, variable_number] = size(XHF);
switch nargin
    case 4
        hyp_HF = varargin{1};
        LF_model = varargin{2};

        % check whether LF model exist predict_function
        if ~isfield(LF_model, 'predict_function')
            error('interpHieraKrigingPreModel: low fidelity lack predict function');
        end
    case 6
        hyp_HF = varargin{1};
        XLF = varargin{2};
        YLF = varargin{3};
        hyp_LF = varargin{4};

        [x_LF_number, variable_number] = size(XLF);

        if isempty(hyp_LF)
            hyp_LF = 0;
        end

        % first step
        % construct low fidelity model

        % normalize data
        aver_X = mean(XLF);
        stdD_X = std(XLF);
        aver_Y = mean(YLF);
        stdD_Y = std(YLF);
        index__ = find(stdD_X == 0);
        if  ~isempty(index__), stdD_X(index__) = 1; end
        index__ = find(stdD_Y == 0);
        if  ~isempty(index__), stdD_Y(index__) = 1; end

        XLF_nomlz = (XLF-aver_X)./stdD_X;
        YLF_nomlz = (YLF-aver_Y)./stdD_Y;

        % initial X_dis_sq
        XLF_dis_sq = zeros(x_LF_number, x_LF_number, variable_number);
        for variable_index = 1:variable_number
            XLF_dis_sq(:, :, variable_index) = ...
                (XLF_nomlz(:, variable_index)-XLF_nomlz(:, variable_index)').^2;
        end

        % regression function define
        % notice reg_function process no normalization data
        % reg_function = @(X) regZero(X);
        reg_function = @(X) regLinear(X);

        % calculate reg
        fval_reg_LF = reg_function(XLF);
        fval_reg_nomlz_LF = (fval_reg_LF-aver_Y)./stdD_Y;

        % optimal to get hyperparameter
        low_bou_hyp = -3;
        up_bou_hyp = 3;
        fmincon_option = optimoptions('fmincon', 'Display', 'none', ...
            'OptimalityTolerance', 1e-2, ...
            'FiniteDifferenceStepSize', 1e-5, ..., 
            'MaxIterations', 10, 'SpecifyObjectiveGradient', true);
        prob_NLL_function = @(hyp) probNLLKriging...
            (XLF_dis_sq, YLF_nomlz, x_LF_number, variable_number, hyp, fval_reg_nomlz_LF);

        % [fval, gradient] = object_function_hyp(hyp_LF)
        % [~, gradient_differ] = differ(object_function_hyp, hyp_LF)

        hyp_LF = fmincon...
            (prob_NLL_function, hyp_LF, [], [], [], [], low_bou_hyp, up_bou_hyp, [], fmincon_option);

        % get parameter
        [cov_LF, inv_cov_LF, ~, beta_LF, sigma_sq_LF] = interpKriging...
            (XLF_dis_sq, YLF_nomlz, x_LF_number, variable_number, exp(hyp_LF), fval_reg_nomlz_LF);
        gama_LF = inv_cov_LF*(YLF_nomlz-fval_reg_nomlz_LF*beta_LF);
        FTRF_LF = fval_reg_nomlz_LF'*inv_cov_LF*fval_reg_nomlz_LF;

        % initialization predict function
        predict_function_LF = @(X_predict) interpKrigingPredictor...
            (X_predict, XLF_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
            x_LF_number, variable_number, exp(hyp_LF), beta_LF, gama_LF, sigma_sq_LF, ...
            inv_cov_LF, fval_reg_nomlz_LF, FTRF_LF, reg_function);

        LF_model.X = XLF;
        LF_model.Y = YLF;
        LF_model.fval_regression = fval_reg_LF;
        LF_model.covariance = cov_LF;
        LF_model.inv_covariance = inv_cov_LF;

        LF_model.hyp = hyp_LF;
        LF_model.beta = beta_LF;
        LF_model.gama = gama_LF;
        LF_model.sigma_sq = sigma_sq_LF;
        LF_model.aver_X = aver_X;
        LF_model.stdD_X = stdD_X;
        LF_model.aver_Y = aver_Y;
        LF_model.stdD_Y = stdD_Y;

        LF_model.predict_function = predict_function_LF;
    otherwise
        error('interpHieraKrigingPreModel: error input');
end
HK_model.LF_model = LF_model;

% second step
% construct hierarchical model
if isempty(hyp_HF)
    hyp_HF = 0;
end

predict_function_LF = LF_model.predict_function;

% normalize data
aver_X = mean(XHF);
stdD_X = std(XHF);
aver_Y = mean(YHF);
stdD_Y = std(YHF);
index__ = find(stdD_X == 0);
if  ~isempty(index__), stdD_X(index__) = 1; end
index__ = find(stdD_Y == 0);
if  ~isempty(index__), stdD_Y(index__) = 1; end
XHF_nomlz = (XHF-aver_X)./stdD_X;
YHF_nomlz = (YHF-aver_Y)./stdD_Y;

% initial X_dis_sq
XHF_dis_sq = zeros(x_HF_number, x_HF_number, variable_number);
for variable_index = 1:variable_number
    XHF_dis_sq(:, :, variable_index) = ...
        (XHF_nomlz(:, variable_index)-XHF_nomlz(:, variable_index)').^2;
end

% evaluate low fidelty predict value in high fidelity point as base fval
reg_function = @(X) predict_function_LF(X);

% calculate reg
fval_reg_HF = reg_function(XHF);
fval_reg_nomlz_HF = (fval_reg_HF-aver_Y)./stdD_Y;

% optimal to get hyperparameter
prob_NLL_function = @(hyp) probNLLKriging...
    (XHF_dis_sq, YHF_nomlz, x_HF_number, variable_number, hyp, fval_reg_nomlz_HF);

% [fval, gradient] = prob_NLL_function(hyp_HF)
% [~, gradient_differ] = differ(prob_NLL_function, hyp_HF)

% drawFunction(object_function_hyp, low_bou_hyp, up_bou_hyp);

hyp_HF = fmincon...
    (prob_NLL_function, hyp_HF, [], [], [], [], low_bou_hyp, up_bou_hyp, [], fmincon_option);

% calculate covariance and other parameter
[cov_HF, inv_cov_HF, ~, beta_HF, sigma_sq_HF] = interpKriging...
    (XHF_dis_sq, YHF_nomlz, x_HF_number, variable_number, exp(hyp_HF), fval_reg_nomlz_HF);
gama_HF = inv_cov_HF*(YHF_nomlz-fval_reg_nomlz_HF*beta_HF);
FTRF_HF = fval_reg_nomlz_HF'*inv_cov_HF*fval_reg_nomlz_HF;

% initialization predict function
predict_function = @(X_predict) interpKrigingPredictor...
    (X_predict, XHF_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
    x_HF_number, variable_number, exp(hyp_HF), beta_HF, gama_HF, sigma_sq_HF, ...
    inv_cov_HF, fval_reg_nomlz_HF, FTRF_HF, reg_function);

HK_model.X = {XHF, XLF};
HK_model.Y = {YHF, YLF};
HK_model.fval_regression = fval_reg_HF;
HK_model.covariance = cov_HF;
HK_model.inv_covariance = inv_cov_HF;

HK_model.hyp = hyp_HF;
HK_model.beta = beta_HF;
HK_model.gama = gama_HF;
HK_model.sigma_sq = sigma_sq_HF;

HK_model.aver_X = aver_X;
HK_model.stdD_X = stdD_X;
HK_model.aver_Y = aver_Y;
HK_model.stdD_Y = stdD_Y;

HK_model.predict_function = predict_function;

% abbreviation:
% num: number, pred: predict, vari: variable, hyp: hyper parameter
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

    function [Y_pred, Var_pred] = interpKrigingPredictor...
            (X_pred, X_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
            x_num, vari_num, theta, beta, gama, sigma_sq, ...
            inv_cov, fval_reg, FTRF, reg_function)
        % kriging interpolation predict function
        % input predict_x and kriging model
        % predict_x is row vector
        % output the predict value
        %
        [x_pred_num, ~] = size(X_pred);
        fval_reg_pred = reg_function(X_pred);

        % normalize data
        X_pred_nomlz = (X_pred-aver_X)./stdD_X;
        fval_reg_pred_nomlz = (fval_reg_pred-aver_Y)./stdD_Y;
        
        % predict covariance
        predict_cov = zeros(x_num, x_pred_num);
        for vari_index = 1:vari_num
            predict_cov = predict_cov+...
                (X_nomlz(:, vari_index)-X_pred_nomlz(:, vari_index)').^2*theta;
        end
        predict_cov = exp(-predict_cov/vari_num);

        % predict base fval
        
        Y_pred = fval_reg_pred_nomlz*beta+predict_cov'*gama;
        
        % predict variance
        u__ = fval_reg'*inv_cov*predict_cov-fval_reg_pred_nomlz';
        Var_pred = sigma_sq*...
            (1+u__'/FTRF*u__+...
            -predict_cov'*inv_cov*predict_cov);
        
        % normalize data
        Y_pred = Y_pred*stdD_Y+aver_Y;
        Var_pred = diag(Var_pred)*stdD_Y*stdD_Y;
    end
    
    function F_reg = regZero(X)
        % zero order base funcion
        %
        F_reg = ones(size(X, 1), 1); % zero
    end
    
    function F_reg = regLinear(X)
        % first order base funcion
        %
        F_reg = [ones(size(X, 1), 1), X]; % linear
    end
end

function [predict_function, kriging_model] = interpKrigingPreModel...
    (X, Y, hyp)
% nomalization method is grassian
% add multi x_predict input support
% prepare model, optimal theta and calculation parameter
% X, Y are x_number x variable_number matrix
% aver_X, stdD_X is 1 x x_number matrix
% theta beta gama sigma_sq is normalizede, so predict y is normalize
% theta = exp(hyp)
%
% input initial data X, Y, which are real data
%
% output is a kriging model, include predict_function...
% X, Y, base_function_list
%
% Copyright 2023.2 Adel
%
[x_number, variable_number] = size(X);
if nargin < 3
    hyp = zeros(1, variable_number);
end

% normalize data
aver_X = mean(X);
stdD_X = std(X);
aver_Y = mean(Y);
stdD_Y = std(Y);
index__ = find(stdD_X == 0);
if  ~isempty(index__), stdD_X(index__) = 1; end
index__ = find(stdD_Y == 0);
if  ~isempty(index__), stdD_Y(index__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;
Y_nomlz = (Y-aver_Y)./stdD_Y;

% initial X_dis_sq
X_dis_sq = zeros(x_number, x_number, variable_number);
for variable_index = 1:variable_number
    X_dis_sq(:, :, variable_index) = ...
        (X_nomlz(:, variable_index)-X_nomlz(:, variable_index)').^2;
end

% regression function define
% notice reg_function process no normalization data
% reg_function = @(X) regZero(X);
reg_function = @(X) regLinear(X);

% calculate reg
fval_reg_nomlz = (reg_function(X)-aver_Y)./stdD_Y;

% optimal to get hyperparameter
fmincon_option = optimoptions('fmincon', 'Display', 'none', ...
    'OptimalityTolerance', 1e-2, ...
    'FiniteDifferenceStepSize', 1e-5, ..., 
    'MaxIterations', 10, 'SpecifyObjectiveGradient', false);
low_bou_hyp = -3;
up_bou_hyp = 3;
object_function_hyp = @(hyp) probNLLKriging...
    (X_dis_sq, Y_nomlz, x_number, variable_number, hyp, fval_reg_nomlz);

% [fval, gradient] = object_function_hyp(hyp)
% [~, gradient_differ] = differ(object_function_hyp, hyp)

% drawFunction(object_function_hyp, low_bou_hyp, up_bou_hyp);

hyp = fmincon...
    (object_function_hyp, hyp, [], [], [], [], low_bou_hyp, up_bou_hyp, [], fmincon_option);

% get parameter
[covariance, inv_covariance, beta, sigma_sq] = interpKriging...
    (X_dis_sq, Y_nomlz, x_number, variable_number, exp(hyp), fval_reg_nomlz);
gama = inv_covariance*(Y_nomlz-fval_reg_nomlz*beta);
FTRF = fval_reg_nomlz'*inv_covariance*fval_reg_nomlz;

% initialization predict function
predict_function = @(X_predict) interpKrigingPredictor...
    (X_predict, X_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
    x_number, variable_number, exp(hyp), beta, gama, sigma_sq, ...
    inv_covariance, fval_reg_nomlz, FTRF, reg_function);

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
% num: number, pred: predict, vari: variable, hyp: hyper parameter
% NLL: negative log likelihood
    function [fval, gradient] = probNLLKriging...
            (X_dis_sq, Y, x_num, vari_num, hyp, F_reg)
        % function to minimize sigma_sq
        %
        theta = exp(hyp);
        [cov, inv_cov, ~, sigma2, inv_FTRF, Y_Fmiu] = interpKriging...
            (X_dis_sq, Y, x_num, vari_num, theta, F_reg);

        % calculation negative log likelihood
        L = chol(cov)';
        fval = x_num/2*log(sigma2)+sum(log(diag(L)));

        % calculate gradient
        if nargout > 1
            % gradient
            gradient = zeros(vari_num, 1);
            for vari_index = 1:vari_num
                dcov_dtheta = -(X_dis_sq(:, :, vari_index).*cov)*theta(vari_index)/vari_num;

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

                gradient(vari_index) = x_num/2*dlnsigma2_dtheta+0.5*dlndetR;
            end
        end
    end

    function [cov, inv_cov, beta, sigma_sq, inv_FTRF, Y_Fmiu] = interpKriging...
            (X_dis_sq, Y, x_num, vari_num, theta, F_reg)
        % kriging interpolation kernel function
        % Y(x) = beta+Z(x)
        %
        cov = zeros(x_num, x_num);
        for vari_index = 1:vari_num
            cov = cov+X_dis_sq(:, :, vari_index)*theta(vari_index);
        end
        cov = exp(-cov/vari_num)+eye(x_num)*1e-3;

        % coefficient calculation
        inv_cov = cov\eye(x_num);
        inv_FTRF = (F_reg'*inv_cov*F_reg)\eye(size(F_reg, 2));

        % basical bias
        beta = inv_FTRF*(F_reg'*inv_cov*Y);
        Y_Fmiu = Y-F_reg*beta;
        sigma_sq = (Y_Fmiu'*inv_cov*Y_Fmiu)/x_num;
        
    end

    function [Y_pred, Var_pred] = interpKrigingPredictor...
            (X_pred, X_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
            x_num, vari_num, theta, beta, gama, sigma_sq, ...
            inv_cov, fval_reg_nomlz, FTRF, reg_function)
        % kriging interpolation predict function
        % input predict_x and kriging model
        % predict_x is row vector
        % output the predict value
        %
        [x_pred_num, ~] = size(X_pred);
        fval_reg_pred = reg_function(X_pred);

        % normalize data
        X_pred_nomlz = (X_pred-aver_X)./stdD_X;
        fval_reg_pred_nomlz = (fval_reg_pred-aver_Y)./stdD_Y;
        
        % predict covariance
        predict_cov = zeros(x_num, x_pred_num);
        for vari_index = 1:vari_num
            predict_cov = predict_cov+...
                (X_nomlz(:, vari_index)-X_pred_nomlz(:, vari_index)').^2*theta(vari_index);
        end
        predict_cov = exp(-predict_cov/vari_num);

        % predict base fval
        
        Y_pred = fval_reg_pred_nomlz*beta+predict_cov'*gama;
        
        % predict variance
        u__ = fval_reg_nomlz'*inv_cov*predict_cov-fval_reg_pred_nomlz';
        Var_pred = sigma_sq*...
            (1+u__'/FTRF*u__+...
            -predict_cov'*inv_cov*predict_cov);
        
        % normalize data
        Y_pred = Y_pred*stdD_Y+aver_Y;
        Var_pred = diag(Var_pred)*stdD_Y*stdD_Y;
    end

    function F_reg = regZero(X)
        % zero order base funcion
        %
        F_reg = ones(size(X, 1), 1); % zero
    end

    function F_reg = regLinear(X)
        % first order base funcion
        %
        F_reg = [ones(size(X, 1), 1), X]; % linear
    end
end
