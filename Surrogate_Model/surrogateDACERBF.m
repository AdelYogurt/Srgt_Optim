clc;
clear;
close all hidden;

addpath(genpath([pwd,'/krg']));

load('PK.mat');
Y = Y./max(abs(Y));

[predict_function,kriging_model] = interpKrigingPreModel(X,Y);
figure_handle = figure(1);
interpVisualize(kriging_model,low_bou,up_bou,[],[],[],figure_handle)

function [predict_function,RBF_model] = interpKrigingPreModel...
    (X,Y)
% create RBF model by DACE code package
%
SRGTRBF_struct = RBFSetOption(X, Y, 'cubic', 0);

predict_function = @(X_pred) RBFPredict(X_pred, SRGTRBF_struct);

RBF_model.X = X;
RBF_model.Y = Y;

RBF_model.predict_function = predict_function;
end

