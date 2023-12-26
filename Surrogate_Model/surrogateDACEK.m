clc;
clear;
close all hidden;

addpath(genpath([pwd,'/krg']));

load('PK.mat');
Y = Y./max(abs(Y));

[predict_function,kriging_model]=interpKrigingPreModel(X,Y);
figure_handle=figure(1);
interpVisualize(kriging_model,low_bou,up_bou,[],[],[],figure_handle)

function [predict_function,kriging_model]=interpKrigingPreModel...
    (X,Y,hyp)
% create kriging model by DACE code package
%
srgtOPT  = srgtsKRGSetOptions(X, Y);
srgtSRGT = srgtsKRGFit(srgtOPT);

predict_function = @(Xtest) srgtsKRGPredictor(Xtest, srgtSRGT);

kriging_model.X=X;
kriging_model.Y=Y;

kriging_model.predict_function=predict_function;
end

