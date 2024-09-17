function loss=cvSrgtError(srgt,X_test,Y_test,type)
% simple function to calculate loss
% only support RMSE R^2 NRMSE
%
num=length(Y_test);
if size(X_test,1) ~= size(Y_test,1)
   error('varifyMethodFunction: fval_real_list number do not equal to fval_real_list number');
end

% predict x_check fval
try
    Y_test_pred=srgt.predict(X_test);
catch
    Y_test_pred=zeros(size(X_test,1),1);
    for x_index=1:size(X_test,1)
        x_check=X_test(x_index,:);
        Y_test_pred(x_index)=srgt.predict(x_check);
    end
end

loss=0;
switch type
    case 'RMSE'
        sum_errr_sq=sum((Y_test_pred-Y_test).^2);
        RMSE=sqrt(sum_errr_sq/num);
        loss=loss+RMSE;
    case 'R2'
        if num == 1
            error('interpolationCrossVerify: R2 validation can not use single check point')
        end
        Y_avg=sum(Y_test)/num;
        sum_errr_sq=sum((Y_test_pred-Y_test).^2);
        sum_var=sum((Y_avg-Y_test).^2);
        R2=1-sum_errr_sq/sum_var;
        loss=loss+R2;
    case 'NRMSE'
        sum_errr_sq=sum((Y_test_pred-Y_test).^2);
        sum_fval_sq=sum(Y_test.^2);
        NRMSE=sqrt(sum_errr_sq/sum_fval_sq);
        loss=loss+NRMSE;
    case 'RMSE_sq'
        sum_errr_sq=sum((Y_test_pred-Y_test).^2);
        RMSE=(sum_errr_sq/num);
        loss=loss+RMSE;
    otherwise
         error('varifyMethodFunction: unsupported varify type')
end
end