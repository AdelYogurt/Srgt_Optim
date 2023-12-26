clc;
clear;
close all hidden;

load matlab

SVMModel = fitcsvm(x_list,fval_label,'ClassNames',[0 1],'Standardize',true,...
    'KernelFunction','rbf','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('Verbose',1,'ShowPlots',true));

d = 0.02;
[x1Grid,x2Grid] = meshgrid(low_bou(1):d:up_bou(1),...
    low_bou(2):d:up_bou(2));
xGrid = [x1Grid(:),x2Grid(:)];
N = size(xGrid,1);

[class,score] = predict(SVMModel,xGrid);

figure
h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),class,...
    [0.1 0.5 0.5; 0.5 0.1 0.5; 0.5 0.5 0.1]);