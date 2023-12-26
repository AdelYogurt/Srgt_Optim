clc;
clear;
close all hidden;

low_bou = [-3,-3];
up_bou = [3,3];
x_list = [1,2;
    1.5,-3;
    -2.2,1.5;
    -1,3;
    0.6,0.2;
    -0.8,0.1];
fval_list = zeros(6,1);

for x_index = 1:size(fval_list,1)
    x = x_list(x_index,:);
    fval_list(x_index) = 1+2*x(1)+3*x(2)+4*x(1)*x(1)+5*x(2)*x(2)+6*x(1)*x(2);
end

[predict_function,respsurf_model] = interpRespSurfPreModel(x_list,fval_list);
figure_handle = figure(1);
interpVisualize(respsurf_model,low_bou,up_bou,[],[],[],figure_handle)

function [predict_function,respsurf_model] = interpRespSurfPreModel(X,Y)
% polynomial response surface interpolation pre model function
%
% input data will be normalize by average and standard deviation of data
%
% input:
% X,Y(initial data,which are real data,x_number x variable_number matrix)
%
% output:
% predict_function,respond surface model(include X,Y,base_function,...)
%
% Copyright 2022 Adel
%

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

beta = interpRespSurf(X_nomlz,Y_nomlz,x_number,variable_number);

% initialization predict function
predict_function = @(X_predict) interpolationRespSurfPredictor...
    (X_predict,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_number,variable_number,beta);

respsurf_model.X = X;
respsurf_model.Y = Y;
respsurf_model.beta = beta;

respsurf_model.aver_X = aver_X;
respsurf_model.stdD_X = stdD_X;
respsurf_model.aver_Y = aver_Y;
respsurf_model.stdD_Y = stdD_Y;

respsurf_model.predict_function = predict_function;

% abbreviation:
% num: number,pred: predict,vari: variable
    function beta = interpRespSurf(X,Y,x_num,vari_num)
        % interpolation polynomial responed surface core function
        % calculation beta
        %
        X_cross = zeros(x_num,(vari_num-1)*vari_num/2);

        cross_index = 1;
        for i_index = 1:vari_num
            for j_index = i_index+1:vari_num
                X_cross(:,cross_index) = X(:,i_index).*X(:,j_index);
                cross_index = cross_index+1;
            end
        end
        X_inter = [ones(x_num,1),X,X.^2,X_cross];
        
        X_inter_X_inter = X_inter'*X_inter;
        beta = X_inter_X_inter\X_inter'*Y;
    end
    function [Y_pred] = interpolationRespSurfPredictor...
            (X_pred,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,beta)
        % polynomial response surface interpolation predict function
        % input predict_x and respsurf_model model
        % predict_x is row vector
        % output the predict value
        %
        [x_pred_num,~] = size(X_pred);

        % normalize data
        X_pred = (X_pred-aver_X)./stdD_X;
        
        % predict value
        X_cross = zeros(x_pred_num,(vari_num-1)*vari_num/2);
        cross_index = 1;
        for i_index = 1:vari_num
            for j_index = i_index+1:vari_num
                X_cross(:,cross_index) = X_pred(:,i_index).*X_pred(:,j_index);
                cross_index = cross_index+1;
            end
        end
        X_pred_inter = [ones(x_pred_num,1),X_pred,X_pred.^2,X_cross];
        
        % predict variance
        Y_pred = X_pred_inter*beta;
        
        % normalize data
        Y_pred = Y_pred*stdD_Y+aver_Y;
    end
end
