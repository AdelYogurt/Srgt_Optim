clc;
clear;
close all hidden;

load('C_30.mat');

[predict_function,SVM_model] = classifySupportVectorMachine...
    (X,Y,10);
figure_handle = figure(1);
classifyVisualization(SVM_model,low_bou,up_bou,[],figure_handle)

function [predict_function,SVM_model] = classifySupportVectorMachine...
    (X,Class,C,kernel_function)
% generate support vector machine model
% use fmincon to get alpha
% only support binary classification, -1 and 1
% X, Y is x_number x variable_number matrix
% C is penalty factor, default is empty
% kernel_function default is gauss kernal function
%
if nargin < 4
    kernel_function = [];
    if nargin < 3
        C = [];
    end
end

[x_number,variable_number] = size(X);

% normalization data
aver_X = mean(X);
stdD_X = std(X);
index__ = find(stdD_X == 0);
if  ~isempty(index__),  stdD_X(index__) = 1; end
X_nomlz = (X-aver_X)./stdD_X;

Y = Class;

% default kernal function
if isempty(kernel_function)
    % notice after standard normal distribution normalize
    % X usually distribution in -2 to 2, so divide by 16
%     sigma = 100*log(sqrt(x_number))/variable_number^2/16;
    sigma = 1/variable_number;
    kernel_function = @(U,V) kernelGaussian(U,V,sigma);
end

% initialization kernal function process X_cov
K = kernel_function(X_nomlz,X_nomlz);

% min SVM object function to get alpha
object_function = @(alpha) -objectFunction(alpha,K,Y);
alpha = ones(x_number,1)*0.5;
low_bou_fmincon = 0*ones(x_number,1);
if isempty(C) || C==0
    up_bou_fmincon = [];
else
    up_bou_fmincon = C*ones(x_number,1);
end
Aeq = Y';
fmincon_options = optimoptions('fmincon','Display','none','Algorithm','sqp');
alpha = fmincon(object_function,alpha,...
    [],[],Aeq,0,low_bou_fmincon,up_bou_fmincon,[],fmincon_options);

% obtain other paramter
alpha_Y = alpha.*Y;

w = sum(alpha_Y.*X_nomlz);
index_list = find(alpha > 1e-6); % support vector
alpha_Y_cov = K*alpha_Y;
b = sum(Y(index_list)-alpha_Y_cov(index_list))/length(index_list);

% generate predict function
predict_function = @(x) classifySupportVectorMachinePredictor...
    (x,X_nomlz,alpha_Y,b,aver_X,stdD_X,kernel_function);

% output model
SVM_model.X = X;
SVM_model.Class = Class;
SVM_model.Y = Y;
SVM_model.X_nomlz = X_nomlz;
SVM_model.aver_X = aver_X;
SVM_model.stdD_X = stdD_X;
SVM_model.alpha = alpha;
SVM_model.w = w;
SVM_model.b = b;
SVM_model.kernel_function = kernel_function;
SVM_model.predict_function = predict_function;

    function fval = objectFunction(alpha,K,Y)
        % support vector machine maximum object function
        %
        alpha = alpha(:);
        alpha_Y__ = alpha.*Y;
        fval = sum(alpha)-alpha_Y__'*K*alpha_Y__/2;
    end

    function [Class_pred,Probability] = classifySupportVectorMachinePredictor...
            (X_pred,X_nomlz,alpha_Y,b,aver_X,stdD_X,kernel_function)
        % predict_fval is 1 or -1, predict_class is 1 or 0
        %
        % x input is colume vector
        %
        X_pred_nomlz = (X_pred-aver_X)./stdD_X;
        K_pred = kernel_function(X_pred_nomlz,X_nomlz);
        Probability = K_pred*alpha_Y+b;
        Probability = 1./(1+exp(-Probability));
        Class_pred = Probability > 0.5;
    end

    function K = kernelGaussian(U,V,sigma)
        % gaussian kernal function
        %
        K = zeros(size(U,1),size(V,1));
        vari_num = size(U,2);
        for vari_index = 1:vari_num
            K = K+(U(:,vari_index)-V(:,vari_index)').^2;
        end
        K = exp(-K*sigma);
    end
end

function fval = lowFunction(x)
if 0.45+sin(2.2*pi*x(1))/2.5-x(2) > 0
    fval = 1;
else
    fval = -1;
end
end
function fval = highFunction(x)
if 0.5+sin(2.5*pi*x(1))/3-x(2) > 0
    fval = 1;
else
    fval = -1;
end
end