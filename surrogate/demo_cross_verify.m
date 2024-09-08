clc;
% clear;
close all hidden;

%% calculate CV value

% [~,vari_num,~,~,obj_fcn,~,~,~,~,nonlcon_fcn]=Benchmark().get('single','PK');
% X=lhsdesign(30,2);
% Y=obj_fcn(X);
% 
K=10;
verify_type='R2';
% 
% fit_model_fcn=@(X,Y) srgtRBF(X,Y);
% fit_model_fcn=@(X,Y) srgtKRG(X,Y);
% 
value=calCrossVerify(fit_model_fcn,K,X,Y,verify_type);
fprintf('loss: %f\n',value);

%% SSV

% [~,vari_num,~,~,obj_fcn,~,~,~,~,nonlcon_fcn]=Benchmark().get('single','PK');
% X=lhsdesign(600,2);
% Y=obj_fcn(X);
% 
% model=srgtRBF(X,Y);
% model=srgtKRG(X,Y);
% 
% X_pred=lhsdesign(10000,2);
% Y_pred=model.predict(X_pred);
% Y_real=obj_fcn(X_pred);
% R2=1-sum((Y_real-Y_pred).^2)/sum((Y_real-mean(Y_real)).^2);
% fprintf('R2: %f\n',R2);

%% compare loss each surrogate model

% [~,vari_num,~,~,obj_fcn,~,~,~,~,nonlcon_fcn]=Benchmark().get('single','PK');
% sample_num=100;
% K=10;
% verify_type='RMSE';
% 
% repeat_time=10;
% Loss_A=zeros(repeat_time,1);
% Loss_B=zeros(repeat_time,1);
% 
% for repeat_index=1:repeat_time
%     X=lhsdesign(sample_num,vari_num);
%     Y=obj_fcn(X);
%     
%     fit_model_fcn_A=@(X,Y) srgtRBF(X,Y);
%     Loss_A(repeat_index)=calCrossVerify...
%         (fit_model_fcn_A,K,X,Y,verify_type);
%     
%     fit_model_fcn_B=@(X,Y) srgtKRG(X,Y);
%     Loss_B(repeat_index)=calCrossVerify...
%         (fit_model_fcn_B,K,X,Y,verify_type);
% end
% 
% boxplot([Loss_A,Loss_B]);

