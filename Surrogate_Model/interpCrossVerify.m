function loss = interpCrossVerify...
    (interp_model_function,K,x_list,fval_list,verify_type)
% use cross validation to get model accuracy version 1
% The x_list were divided equally into n fold.
% support RMSE R2 NRMSE
%
[x_number,~] = size(x_list);
x_fold_number = round(x_number/K); % each fold x number

% K fold
loss_list = zeros(K,1);
for K_index = 1:K
    % get a list in addition to the list used for checking
    x_fit_list = x_list([1:((K_index-1)*x_fold_number),(K_index*x_fold_number+1):x_number],:);
    fval_fit_list = fval_list([1:((K_index-1)*x_fold_number),(K_index*x_fold_number+1):x_number]);
    x_check_list = x_list([((K_index-1)*x_fold_number+1):(K_index*x_fold_number)],:);
    fval_check_list = fval_list([((K_index-1)*x_fold_number+1):(K_index*x_fold_number)],:);
    
    % generate interp model
    predict_function = interp_model_function(x_fit_list,fval_fit_list);
    
    % predict x_check fval
    fval_predict_list = zeros(x_fold_number,1);
    for x_index = 1:x_fold_number
        x_check = x_check_list(x_index,:);
        fval_predict_list(x_index) = predict_function(x_check);
    end

    % calculate loss
    loss_list(K_index) = verifyMethodFunction...
        (fval_predict_list,fval_check_list,verify_type);
end
loss = sum(loss_list)/K;
end

function loss = verifyMethodFunction...
    (fval_predict_list,fval_check_list,verify_type)
% simple function to calculate loss
% only support RMSE R^2 NRMSE
%
x_fold_number = length(fval_predict_list);
if length(fval_check_list) ~= length(fval_predict_list)
   error('varifyMethodFunction: fval_real_list number do not equal to fval_real_list number');
end

loss = 0;
switch verify_type
    case 'RMSE'
        sum_error_sq = sum((fval_predict_list-fval_check_list).^2);
        RMSE = sqrt(sum_error_sq/x_fold_number);
        loss = loss+RMSE;
    case 'R2'
        if x_fold_number == 1
            error('interpolationCrossVerify: R2 validation can not use single check point')
        end
        fval_check_average = sum(fval_check_list)/x_fold_number;
        sum_error_sq = sum((fval_predict_list-fval_check_list).^2);
        sum_variance = sum((fval_check_average-fval_check_list).^2);
        R_sq = 1-sum_error_sq/sum_variance;
        loss = loss+R_sq;
    case 'NRMSE'
        sum_error_sq = sum((fval_predict_list-fval_check_list).^2);
        sum_fval_sq = sum(fval_check_list.^2);
        NRMSE = sqrt(sum_error_sq/sum_fval_sq);
        loss = loss+NRMSE;
    case 'RMSE_sq'
        sum_error_sq = sum((fval_predict_list-fval_check_list).^2);
        RMSE = (sum_error_sq/x_fold_number);
        loss = loss+RMSE;
    otherwise
         error('varifyMethodFunction: unsupported varify type')
end
end
