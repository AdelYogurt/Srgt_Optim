function model_RSM=srgtRSM(X,Y)
% generarte polynomial response surface interpolation surrogate model
% input data will be normalize by average and standard deviation of data
%
% input:
% X (matrix): x_num x vari_num matrix
% Y (matrix): x_num x 1 matrix
%
% output:
% model_RSM(struct): a polynomial response surface model
%
% abbreviation:
% num: number, pred: predict, vari: variable, hyp: hyper parameter
%
% Copyright 2022 Adel
%

% normalize data
[x_num,vari_num]=size(X);
aver_X=mean(X);
stdD_X=std(X);stdD_X(stdD_X == 0)=1;
aver_Y=mean(Y);
stdD_Y=std(Y);stdD_Y(stdD_Y == 0)=1;
X_nomlz=(X-aver_X)./stdD_X;
Y_nomlz=(Y-aver_Y)./stdD_Y;

beta=calRSM(X_nomlz,Y_nomlz,x_num,vari_num);

% initialization predict function
pred_fcn=@(X_predict) predictRSM...
    (X_predict,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_num,vari_num,beta);

model_RSM.X=X;
model_RSM.Y=Y;

model_RSM.beta=beta;

model_RSM.predict=pred_fcn;

    function beta=calRSM(X,Y,x_num,vari_num)
        % interpolation polynomial responed surface core function
        % calculation beta
        %
        X_cross=zeros(x_num,(vari_num-1)*vari_num/2);

        cross_index=1;
        for i_index=1:vari_num
            for j_index=i_index+1:vari_num
                X_cross(:,cross_index)=X(:,i_index).*X(:,j_index);
                cross_index=cross_index+1;
            end
        end
        X_inter=[ones(x_num,1),X,X.^2,X_cross];
        
        X_inter_X_inter=X_inter'*X_inter;
        beta=X_inter_X_inter\X_inter'*Y;
    end

    function Y_pred=predictRSM...
            (X_pred,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,beta)
        % polynomial response surface interpolation predict function
        % input predict_x and respsurf_model model
        % predict_x is row vector
        % output the predict value
        %
        [x_pred_num,~]=size(X_pred);

        % normalize data
        X_pred=(X_pred-aver_X)./stdD_X;
        
        % predict value
        X_cross=zeros(x_pred_num,(vari_num-1)*vari_num/2);
        cross_index=1;
        for i_index=1:vari_num
            for j_index=i_index+1:vari_num
                X_cross(:,cross_index)=X_pred(:,i_index).*X_pred(:,j_index);
                cross_index=cross_index+1;
            end
        end
        X_pred_inter=[ones(x_pred_num,1),X_pred,X_pred.^2,X_cross];
        
        % predict variance
        Y_pred=X_pred_inter*beta;
        
        % normalize data
        Y_pred=Y_pred*stdD_Y+aver_Y;
    end
end
