function loss = calCrossVerify(fit_model_fcn,K,x_list,fval_list,verify_type)
% use cross validation to get model accuracy version 1
% The x_list were divided equally into n fold.
% support RMSE R2 NRMSE
%
[x_number,~] = size(x_list);

% K fold
loss_list = zeros(K,1);
x_K_index=crossvalind('KFold',x_number,K);
for K_index = 1:K
    % get a list in addition to the list used for checking
    train_index=1:x_number;
    test_index=find(x_K_index == K_index);
    train_index(test_index)=[];

    X_train = x_list(train_index,:);
    Y_train = fval_list(train_index,:);
    X_test = x_list(test_index,:);
    Y_test = fval_list(test_index,:);
    
    % generate interp model
    model = fit_model_fcn(X_train,Y_train);
    
    % predict x_check fval
    Y_test_pred = zeros(size(X_test,1),1);
    for x_index = 1:size(X_test,1)
        x_check = X_test(x_index,:);
        Y_test_pred(x_index) = model.predict(x_check);
    end

    % calculate loss
    loss_list(K_index) = calVerify(Y_test_pred,Y_test,verify_type);
end
loss = sum(loss_list)/K;
end

function loss = calVerify(Y_pred,Y,verify_type)
% simple function to calculate loss
% only support RMSE R^2 NRMSE
%
x_fold_number = length(Y_pred);
if length(Y) ~= length(Y_pred)
   error('varifyMethodFunction: fval_real_list number do not equal to fval_real_list number');
end

loss = 0;
switch verify_type
    case 'RMSE'
        sum_error_sq = sum((Y_pred-Y).^2);
        RMSE = sqrt(sum_error_sq/x_fold_number);
        loss = loss+RMSE;
    case 'R2'
        if x_fold_number == 1
            error('interpolationCrossVerify: R2 validation can not use single check point')
        end
        fval_check_average = sum(Y)/x_fold_number;
        sum_error_sq = sum((Y_pred-Y).^2);
        sum_variance = sum((fval_check_average-Y).^2);
        R_sq = 1-sum_error_sq/sum_variance;
        loss = loss+R_sq;
    case 'NRMSE'
        sum_error_sq = sum((Y_pred-Y).^2);
        sum_fval_sq = sum(Y.^2);
        NRMSE = sqrt(sum_error_sq/sum_fval_sq);
        loss = loss+NRMSE;
    case 'RMSE_sq'
        sum_error_sq = sum((Y_pred-Y).^2);
        RMSE = (sum_error_sq/x_fold_number);
        loss = loss+RMSE;
    otherwise
         error('varifyMethodFunction: unsupported varify type')
end
end
