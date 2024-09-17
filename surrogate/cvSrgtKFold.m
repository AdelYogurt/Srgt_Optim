function loss=cvSrgtKFold(srgt_fit_fcn,X,Y,K,type)
% use cross validation to get model accuracy version 1
% The x_list were divided equally into n fold.
% support RMSE R2 NRMSE
%
[x_num,~]=size(X);

% K fold
loss_list=zeros(K,1);
x_K_idx=crossvalind('KFold',x_num,K);
for K_idx=1:K
    % get a list in addition to the list used for checking
    train_idx=1:x_num;
    test_idx=find(x_K_idx == K_idx);
    train_idx(test_idx)=[];

    X_train=X(train_idx,:);
    Y_train=Y(train_idx,:);
    X_test=X(test_idx,:);
    Y_test=Y(test_idx,:);
    
    % generate interp model
    srgt=srgt_fit_fcn(X_train,Y_train);
    
    % calculate loss
    loss_list(K_idx)=cvSrgtError(srgt,X_test,Y_test,type);
end
loss=sum(loss_list)/K;
end