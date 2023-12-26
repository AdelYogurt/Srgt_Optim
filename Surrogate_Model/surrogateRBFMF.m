clc;
clear;
close all hidden;

load('Forrester.mat')

% [predict_function_RBF, model_RBF] = interpRadialBasisPreModel...
%     (XHF, YHF, []);
% [y_RBF] = predict_function_RBF(x);
% [x_best,Y_RBF_best]=fmincon(predict_function_RBF,0.2,[],[],[],[],0,1,[],optimoptions('fmincon','Display','none'));
% figure(1);
% line(x, y_real, 'Color', 'b', 'LineStyle', '-','LineWidth',1, 'Marker','o','MarkerIndices',[1,41,61,101]);
% line(x, y_real_low, 'Color', [0.9290 0.6940 0.1250] , 'LineStyle', '--','LineWidth',2, 'Marker','x','MarkerSize',10,'MarkerIndices',1:10:101);
% line(0.7572,-6.0167,'Marker','p','MarkerSize',10,'Color','b')
% line(x, y_RBF, 'Color', 'r', 'LineStyle', ':','LineWidth',2);
% line(x_best,Y_RBF_best,'Marker','p','MarkerSize',10, 'Color', 'r')
% legend('高精度模型及采样点', '低精度模型及采样点','高精度全局最优值','RBF','RBF全局最优值','Location','northwest')
% xlabel('x');ylabel('y')
% 
% [predict_function_RBFMF, model_RBFMF] = interpRadialBasisMultiFidelityPreModel...
%     (XHF, YHF, [], XLF, YLF, []);
% [y_RBFMF] = predict_function_RBFMF(x);
% [x_best,Y_RBFMF_best]=fmincon(predict_function_RBFMF,0.7,[],[],[],[],0,1,[],optimoptions('fmincon','Display','none'));
% figure(2);
% line(x, y_real, 'Color', 'b', 'LineStyle', '-','LineWidth',1, 'Marker','o','MarkerIndices',[1,41,61,101]);
% line(x, y_real_low, 'Color', [0.9290 0.6940 0.1250] , 'LineStyle', '--','LineWidth',2, 'Marker','x','MarkerSize',10,'MarkerIndices',1:10:101);
% line(0.7572,-6.0167,'Marker','p','MarkerSize',10,'Color','b')
% line(x, y_RBFMF, 'Color', 'k', 'LineStyle', '--','LineWidth',2);
% line(x_best,Y_RBFMF_best,'Marker','p','MarkerSize',10, 'Color', 'k')
% legend('高精度模型及采样点','低精度模型及采样点','高精度全局最优值','MFRBF','MFRBF全局最优值','Location','northwest')
% xlabel('x');ylabel('y')

fig_hdl=figure(1);
fig_hdl.set('Position',[488   200   680  420])

[predict_function_RBF, model_RBF] = interpRadialBasisPreModel...
    (XHF, YHF, []);
[y_RBF] = predict_function_RBF(x);
[x_best,Y_RBF_best]=fmincon(predict_function_RBF,0.2,[],[],[],[],0,1,[],optimoptions('fmincon','Display','none'));
subplot(1,2,1)
line_1=line(x, y_real, 'Color', 'b', 'LineStyle', '-','LineWidth',1, 'Marker','o','MarkerIndices',[1,41,61,101]);
line_2=line(x, y_real_low, 'Color', [0.9290 0.6940 0.1250] , 'LineStyle', '--','LineWidth',2, 'Marker','x','MarkerSize',10,'MarkerIndices',1:10:101);
line_3=line(0.7572,-6.0167,'Marker','p','MarkerSize',10,'Color','b', 'LineStyle', '-','LineWidth',1);
line_4=line(x, y_RBF, 'Color', 'r', 'LineStyle', ':','LineWidth',2);
line_5=line(x_best,Y_RBF_best,'Marker','p','MarkerSize',10, 'Color', 'r', 'LineStyle', ':','LineWidth',2);
axes_handle=gca;
axes_handle.set('Position',[0.0800,0.1500,0.38,0.6750]);
xlabel('x');ylabel('y');grid on;

[predict_function_RBFMF, model_RBFMF] = interpRadialBasisMultiFidelityPreModel...
    (XHF, YHF, [], XLF, YLF, []);
[y_RBFMF] = predict_function_RBFMF(x);
[x_best,Y_RBFMF_best]=fmincon(predict_function_RBFMF,0.7,[],[],[],[],0,1,[],optimoptions('fmincon','Display','none'));
subplot(1,2,2)
line(x, y_real, 'Color', 'b', 'LineStyle', '-','LineWidth',1, 'Marker','o','MarkerIndices',[1,41,61,101]);
line(x, y_real_low, 'Color', [0.9290 0.6940 0.1250] , 'LineStyle', '--','LineWidth',2, 'Marker','x','MarkerSize',10,'MarkerIndices',1:10:101);
line(0.7572,-6.0167,'Marker','p','MarkerSize',10,'Color','b', 'LineStyle', '-','LineWidth',1);
line_6=line(x, y_RBFMF, 'Color', 'k', 'LineStyle', '--','LineWidth',2);
line_7=line(x_best,Y_RBFMF_best,'Marker','p','MarkerSize',10, 'Color', 'k', 'LineStyle', '--','LineWidth',2);
axes_handle=gca;
axes_handle.set('Position',[0.5400,0.1500,0.38,0.6750]);
xlabel('x');ylabel('y');grid on;

legend([line_4,line_5,line_6,line_7,line_1,line_2,line_3],'RBF','RBF全局最优值','MFRBF','MFRBF全局最优值','高精度模型及采样点','低精度模型及采样点','高精度模型全局最优值','Location','northwest','Orientation','horizontal','NumColumns',4)

% load('2DMF.mat');
% [predict_function_HK, MFRBF_model] = interpRadialBasisMultiFidelityPreModel(XHF, YHF, [], XLF, YLF, []);
% interpVisualize(MFRBF_model, low_bou, up_bou);

function [predict_function_MFRBF, MFRBF_model] = interpRadialBasisMultiFidelityPreModel...
    (XHF, YHF, varargin)
% multi fildelity radial basis function interp pre model function
% XHF, YHF are x_HF_number x variable_number matrix
% XLF, YLF are x_LF_number x variable_number matrix
% aver_X, stdD_X is 1 x x_HF_number matrix
%
% input:
% XHF, YHF, basis_func_HF(can be []), XLF, YLF, basis_func_LF(can be [])
% XHF, YHF, basis_func_HF(can be []), LF_model
%
% output:
% predict_function, HK_model
%
% reference: [1] LIU Y, WANG S, ZHOU Q, et al. Modified Multifidelity
% Surrogate Model Based on Radial Basis Function with Adaptive Scale Factor
% [J]. Chinese Journal of Mechanical Engineering, 2022, 35(1): 77.
%
% Copyright 2023 Adel
%
[x_HF_number, variable_number] = size(XHF);
switch nargin
    case 4
        basis_func_HF = varargin{1};
        LF_model = varargin{2};

        % check whether LF model exist predict_function
        if ~isfield(LF_model, 'predict_function')
            error('interpRadialBasisMultiFidelityPreModel: low fidelity lack predict function');
        end
    case 6
        basis_func_HF = varargin{1};
        XLF = varargin{2};
        YLF = varargin{3};
        basis_func_LF = varargin{4};

        [x_LF_number, variable_number] = size(XLF);

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

%         aver_X = 0;
%         stdD_X = 1;
%         aver_Y = 0;
%         stdD_Y = 1;

        XLF_nomlz = (XLF-aver_X)./stdD_X;
        YLF_nomlz = (YLF-aver_Y)./stdD_Y;

        if isempty(basis_func_LF)
            basis_func_LF = @(r) r.^3;
%             c = (prod(max(XLF_nomlz) - min(XLF_nomlz))/x_LF_number)^(1/variable_number);
%             basis_func_LF = @(r) exp(-(r.^2)/c);
        end

        % initialization distance of XLF_nomlz
        XLF_dis = zeros(x_LF_number, x_LF_number);
        for variable_index = 1:variable_number
            XLF_dis = XLF_dis + ...
                (XLF_nomlz(:, variable_index) - XLF_nomlz(:, variable_index)').^2;
        end
        XLF_dis = sqrt(XLF_dis);

        [beta_LF, rdibas_matrix_LF] = interpRadialBasis...
            (XLF_dis, YLF_nomlz, basis_func_LF, x_LF_number);

        % initialization predict function
        predict_function_LF = @(X_predict) interpRadialBasisPredictor...
            (X_predict, XLF_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
            x_LF_number, variable_number, beta_LF, basis_func_LF);

        LF_model.X = XLF;
        LF_model.Y = YLF;
        LF_model.radialbasis_matrix = rdibas_matrix_LF;
        LF_model.beta = beta_LF;

        LF_model.aver_X = aver_X;
        LF_model.stdD_X = stdD_X;
        LF_model.aver_Y = aver_Y;
        LF_model.stdD_Y = stdD_Y;
        LF_model.basis_function = basis_func_LF;

        LF_model.predict_function = predict_function_LF;
    otherwise
        error('interpRadialBasisMultiFidelityPreModel: error input');
end
MFRBF_model.LF_model = LF_model;
predict_function_LF = LF_model.predict_function;

% second step
% construct MFRBF model

% normalize data
aver_X = mean(XHF);
stdD_X = std(XHF);
aver_Y = mean(YHF);
stdD_Y = std(YHF);
index__ = find(stdD_X == 0);
if ~isempty(index__), stdD_X(index__) = 1;end
index__ = find(stdD_Y == 0);
if ~isempty(index__), stdD_Y(index__) = 1;end

% aver_X = 0;
% stdD_X = 1;
% aver_Y = 0;
% stdD_Y = 1;

XHF_nomlz = (XHF - aver_X)./stdD_X;
YHF_nomlz = (YHF - aver_Y)./stdD_Y;

% predict LF value at XHF point
YHF_pred = predict_function_LF(XHF);

% nomalizae
YHF_pred_nomlz = (YHF_pred - aver_Y)./stdD_Y;

if isempty(basis_func_HF)
    basis_func_HF = @(r) r.^3;
%     c = (prod(max(XHF_nomlz) - min(XHF_nomlz))/x_HF_number)^(1/variable_number);
%     basis_func_HF = @(r) exp(-(r.^2)/c);
end

% initialization distance of XHF_nomlz
XHF_dis = zeros(x_HF_number, x_HF_number);
for variable_index = 1:variable_number
    XHF_dis = XHF_dis + ...
        (XHF_nomlz(:, variable_index) - XHF_nomlz(:, variable_index)').^2;
end
XHF_dis = sqrt(XHF_dis);

[omega, rdibas_matrix_HF] = interpRadialBasisMultiFidelity...
    (XHF_dis, YHF_nomlz, basis_func_HF, x_HF_number, YHF_pred_nomlz);

% initialization predict function
predict_function_MFRBF = @(X_predict) interpRadialBasisMultiFidelityPredictor...
    (X_predict, XHF_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
    x_HF_number, variable_number, omega, basis_func_HF, predict_function_LF);

MFRBF_model.X = XHF;
MFRBF_model.Y = YHF;
MFRBF_model.radialbasis_matrix = rdibas_matrix_HF;
MFRBF_model.alpha = omega(1:x_HF_number);
MFRBF_model.beta = omega(x_HF_number+1:end);

MFRBF_model.aver_X = aver_X;
MFRBF_model.stdD_X = stdD_X;
MFRBF_model.aver_Y = aver_Y;
MFRBF_model.stdD_Y = stdD_Y;
MFRBF_model.basis_function = basis_func_HF;

MFRBF_model.predict_function = predict_function_MFRBF;

% abbreviation:
% num: number, pred: predict, vari: variable
    function [beta, rdibas_matrix] = interpRadialBasis...
            (X_dis, Y, basis_function, x_number)
        % interp polynomial responed surface core function
        % calculation beta
        %
        % Copyright 2022 Adel
        %
        rdibas_matrix = basis_function(X_dis);

        % stabilize matrix
        rdibas_matrix = rdibas_matrix + eye(x_number)*1e-9;

        % solve beta
        beta = rdibas_matrix\Y;
    end

    function [Y_pred] = interpRadialBasisPredictor...
            (X_pred, X_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
            x_num, vari_num, beta, basis_function)
        % radial basis function interpolation predict function
        %
        [x_pred_num, ~] = size(X_pred);

        % normalize data
        X_pred_nomlz = (X_pred - aver_X)./stdD_X;

        % calculate distance
        X_dis_pred = zeros(x_pred_num, x_num);
        for vari_index = 1:vari_num
            X_dis_pred = X_dis_pred + ...
                (X_pred_nomlz(:, vari_index) - X_nomlz(:, vari_index)').^2;
        end
        X_dis_pred = sqrt(X_dis_pred);

        % predict variance
        Y_pred = basis_function(X_dis_pred)*beta;

        % normalize data
        Y_pred = Y_pred*stdD_Y + aver_Y;
    end

    function [omega, rdibas_matrix] = interpRadialBasisMultiFidelity...
            (X_dis, Y, basis_function, x_number, YHF_pred)
        % interp polynomial responed surface core function
        % calculation beta
        %
        % Copyright 2022 Adel
        %
        rdibas_matrix = basis_function(X_dis);

        % add low fildelity value
        H = [rdibas_matrix.*YHF_pred, rdibas_matrix];
        H_H = (H*H');

        % stabilize matrix
        H_H = H_H + eye(x_number)*1e-9;

        % solve omega
        omega = H'*(H_H\Y);

%         omega = H\Y;
    end

    function [Y_pred] = interpRadialBasisMultiFidelityPredictor...
            (X_pred, X_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
            x_num, vari_num, omega, basis_function, predict_function_LF)
        % radial basis function interpolation predict function
        %
        [x_pred_num, ~] = size(X_pred);

        % normalize data
        X_pred_nomlz = (X_pred - aver_X)./stdD_X;

        % calculate distance
        X_dis_pred = zeros(x_pred_num, x_num);
        for vari_index = 1:vari_num
            X_dis_pred = X_dis_pred + ...
                (X_pred_nomlz(:, vari_index) - X_nomlz(:, vari_index)').^2;
        end
        X_dis_pred = sqrt(X_dis_pred);

        % predict low fildelity value
        Y_pred_LF = predict_function_LF(X_pred);

        % nomalizae
        Y_pred_LF_nomlz = (Y_pred_LF - aver_Y)./stdD_Y;

        % combine two matrix
        rdibas_matrix_pred = basis_function(X_dis_pred);
        H = [rdibas_matrix_pred.*Y_pred_LF_nomlz, rdibas_matrix_pred];

        % predict variance
        Y_pred_nomlz = H*omega;

        % normalize data
        Y_pred = Y_pred_nomlz*stdD_Y + aver_Y;
    end

end

function [predict_function,radialbasis_model] = interpRadialBasisPreModel(X,Y,basis_function)
% radial basis function interp pre model function
%
% input data will be normalize by average and standard deviation of data
%
% input:
% X,Y(initial data,which are real data,x_number x variable_number matrix)
%
% output:
% predict_function,radial basis model(include X,Y,base_function,...)
%
% Copyright 2023 Adel
%
if nargin < 3
    basis_function = [];
end

[x_number,variable_number] = size(X);

% normalize data
aver_X = mean(X);
stdD_X = std(X);
aver_Y = mean(Y);
stdD_Y = std(Y);
index__ = find(stdD_X == 0);
if ~isempty(index__),stdD_X(index__) = 1;end
index__ = find(stdD_Y == 0);
if ~isempty(index__),stdD_Y(index__) = 1;end
X_nomlz = (X-aver_X)./stdD_X;
Y_nomlz = (Y-aver_Y)./stdD_Y;

if isempty(basis_function)
    basis_function = @(r) r.^3;
%     c = (prod(max(X_nomlz) - min(X_nomlz))/x_number)^(1/variable_number);
%     basis_function = @(r) exp( - (r.^2)/c);
end

% initialization distance of all X
X_dis = zeros(x_number,x_number);
for variable_index = 1:variable_number
    X_dis = X_dis + (X_nomlz(:,variable_index) - X_nomlz(:,variable_index)').^2;
end
X_dis = sqrt(X_dis);

[beta,rdibas_matrix] = interpRadialBasis...
    (X_dis,Y_nomlz,basis_function,x_number);

% initialization predict function
predict_function = @(X_predict) interpRadialBasisPredictor...
    (X_predict,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_number,variable_number,beta,basis_function);

radialbasis_model.X = X;
radialbasis_model.Y = Y;
radialbasis_model.radialbasis_matrix = rdibas_matrix;
radialbasis_model.beta = beta;

radialbasis_model.aver_X = aver_X;
radialbasis_model.stdD_X = stdD_X;
radialbasis_model.aver_Y = aver_Y;
radialbasis_model.stdD_Y = stdD_Y;
radialbasis_model.basis_function = basis_function;

radialbasis_model.predict_function = predict_function;

% abbreviation:
% num: number,pred: predict,vari: variable
    function [beta,rdibas_matrix] = interpRadialBasis...
            (X_dis,Y,basis_function,x_number)
        % interp polynomial responed surface core function
        % calculation beta
        %
        % Copyright 2022 Adel
        %
        rdibas_matrix = basis_function(X_dis);
        
        % stabilize matrix
        rdibas_matrix = rdibas_matrix + eye(x_number)*1e-6;
        
        % solve beta
        beta = rdibas_matrix\Y;
    end

    function [Y_pred] = interpRadialBasisPredictor...
            (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,beta,basis_function)
        % radial basis function interpolation predict function
        %
        [x_pred_num,~] = size(X_pred);

        % normalize data
        X_pred_nomlz = (X_pred - aver_X)./stdD_X;
        
        % calculate distance
        X_dis_pred = zeros(x_pred_num,x_num);
        for vari_index = 1:vari_num
            X_dis_pred = X_dis_pred + ...
                (X_pred_nomlz(:,vari_index) - X_nomlz(:,vari_index)').^2;
        end
        X_dis_pred = sqrt(X_dis_pred);
        
        % predict variance
        Y_pred = basis_function(X_dis_pred)*beta;
        
        % normalize data
        Y_pred = Y_pred*stdD_Y + aver_Y;
    end

end
