clc;
clear;
close all hidden;

benchmark=BenchmarkFunction();

load('matlab.mat');

%% test

% % calculate single loss
% variable_number=2;
% sample_number=100;
% x_list=lhsdesign...
%     (sample_number,variable_number);
% fval_list=zeros(sample_number,1);
% for x_index=1:sample_number
%     fval=benchmark.singleBRObject(x_list(x_index,:));
%     fval_list(x_index)=fval(1);
% end

K=10;
verify_type='R2';
basis_function=@(r) r.^3;
interp_pre_model_function=@(X,Y) interpRadialBasisPreModel(X,Y,basis_function);
loss=interpCrossVerify...
    (interp_pre_model_function,K,x_list,fval_list,verify_type)

%% compare loss each interp model

% variable_number=6;
% sample_number=100;
% K=10;
% verify_type='RMSE';
% 
% repeat_time=10;
% NRMSE_K_data=zeros(repeat_time,1);
% NRMSE_KS_data=zeros(repeat_time,1);
% 
% for repeat_index=1:repeat_time
%     x_list=lhsdesign...
%         (sample_number,variable_number);
%     fval_list=zeros(sample_number,1);
%     for x_index=1:sample_number
%        fval_list(x_index)=benchmark.singleHNObject(x_list(x_index,:));
%     end
%     
%     interp_pre_model_function=@(X,Y) interpKrigingPreModel(X,Y);
%     NRMSE_K_data(repeat_index)=interpCrossVerify...
%         (interp_pre_model_function,K,x_list,fval_list,verify_type);
%     
%     interp_pre_model_function=@(X,Y) interpKrigingPreModelM(X,Y);
%     NRMSE_KS_data(repeat_index)=interpCrossVerify...
%         (interp_pre_model_function,K,x_list,fval_list,verify_type);
% end
% 
% boxplot([NRMSE_K_data,NRMSE_KS_data]);

%% function

function [predict_function, radialbasis_model] = interpRadialBasisPreModel...
    (X, Y, basis_function)
% radial basis function interp pre model function
% input initial data X, Y, which are real data
% X, Y are x_number x variable_number matrix
% aver_X, stdD_X is 1 x x_number matrix
% output is a radial basis model, include X, Y, base_function
% and predict_function
%
% Copyright 2023 Adel
%
if nargin < 3
    basis_function = [];
end

[x_number, variable_number] = size(X);

% normalize data
aver_X = mean(X);
stdD_X = std(X);
aver_Y = mean(Y);
stdD_Y = std(Y);
index__ = find(stdD_X == 0);
if ~isempty(index__), stdD_X(index__) = 1;end
index__ = find(stdD_Y == 0);
if ~isempty(index__), stdD_Y(index__) = 1;end
X_nomlz = (X - aver_X)./stdD_X;
Y_nomlz = (Y - aver_Y)./stdD_Y;

if isempty(basis_function)
    c = (prod(max(X_nomlz) - min(X_nomlz))/x_number)^(1/variable_number);
    basis_function = @(r) exp( - (r.^2)/c);
end

% initialization distance of all X
X_dis = zeros(x_number, x_number);
for variable_index = 1:variable_number
    X_dis = X_dis + (X_nomlz(:, variable_index) - X_nomlz(:, variable_index)').^2;
end
X_dis = sqrt(X_dis);

[beta, rdibas_matrix] = interpRadialBasis...
    (X_dis, Y_nomlz, basis_function, x_number);

% initialization predict function
predict_function = @(X_predict) interpRadialBasisPredictor...
    (X_predict, X_nomlz, aver_X, stdD_X, aver_Y, stdD_Y, ...
    x_number, variable_number, beta, basis_function);

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
        rdibas_matrix = rdibas_matrix + eye(x_number)*1e-6;
        
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

end
