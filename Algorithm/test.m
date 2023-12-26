clc;
clear;
close all hidden;

% benchmark_type='single';
% benchmark_name='G23';
% 
% [model_function,variable_number,low_bou,up_bou,...
%     object_function,A,B,Aeq,Beq,nonlcon_function] = Benchmark().getBenchmark(benchmark_type,benchmark_name);
% x_initial=rand(1,variable_number).*(up_bou-low_bou)+low_bou;
% % [x_best,fval_best]=ga(object_function,variable_number,A,B,Aeq,Beq,low_bou,up_bou,nonlcon_function,optimoptions('ga','MaxGenerations',100));
% [x_best,fval_best]=fmincon(object_function,x_initial,A,B,Aeq,Beq,low_bou,up_bou,nonlcon_function,optimoptions('fmincon','Algorithm','sqp'))

benchmark_error_list = {
    [1,5,100;
    2,0.5,1000;
    3,0.5,1000;
    4,0.5,1000;
    1,0.5,1000;
    2,0.5,1000;
    3,0.5,1000;
    4,0.5,1000;
    1,0.5,1000];
    [1,5,100;
    2,0.5,1000;
    3,0.5,1000;
    4,0.5,1000;
    1,0.5,1000;]
    [1,2,200;
    2,1,1000;
    3,1,1000;
    4,1,1000;
    1,1,1000;
    2,1,1000;];
    [1,-2,100;
    2,0.05,1000;
    3,0.05,1000;
    4,0.05,1000;
    1,0.05,1000;
    2,0.05,1000;
    3,0.05,1000;];};
benchmark_name='G07';
benchmark_type='single';
benchmark_error=benchmark_error_list{1};
% benchmark_error=[
%     1,5,100;
%     2,0.5,1000;
%     3,0.5,1000;
%     4,0.5,1000;
%     1,0.5,1000;];
con_num=[8;4;5;6];
con_num=con_num(1);

[MF_model,variable_number,low_bou,up_bou]=Benchmark().getBenchmarkMF(benchmark_type,benchmark_name,benchmark_error);

X=lhsdesign(200,variable_number);
[Fval_HF,Con_HF]=MF_model{1}(X);
[Fval_LF,Con_LF]=MF_model{2}(X);

mat=corrcoef(Fval_HF,Fval_LF);
disp(mat(2));
for con_index=1:con_num
    mat=corrcoef(Con_HF(:,con_index),Con_LF(:,con_index));
    disp(mat(2));
end



% for number = 80:10:200
%     X_LF = lhsdesign(number,10).*(up_bou-low_bou)+low_bou;
%     % X_HF = getNestedHypercube(X_LF,20,10,up_bou,low_bou);
%     X_HF = lhsdesign(round(number/4),10).*(up_bou-low_bou)+low_bou;
%
%     Fval_LF = Model_function{2}(X_LF);
%     Fval_HF = Model_function{1}(X_HF);
%
%     predict_function=interpRadialBasisPreModel(X_HF,Fval_HF);
%     predict_function_RBFMF=interpRadialBasisMultiFidelityPreModel(X_HF,Fval_HF,[],X_LF,Fval_LF,[]);
%     X_test=lhsdesign(50,variable_number).*(up_bou-low_bou)+low_bou;
%     Fval_check=object_function(X_test);
%     error_check=sum((mean(Fval_check)-Fval_check).^2);
%     
%     error_SF=1-sum((predict_function(X_test)-Fval_check).^2)/error_check;
%     error_MF=1-sum((predict_function_RBFMF(X_test)-Fval_check).^2)/error_check;
% 
%     fprintf('%d, %f, %f\n',number,error_SF,error_MF);
% end


% X_LF = lhsdesign(12,10).*(up_bou-low_bou)+low_bou;
% X_HF = getNestedHypercube(X_LF,20,10,up_bou,low_bou);
% X_HF = lhsdesign(3,10).*(up_bou-low_bou)+low_bou;
% 
% Fval_LF = Model_function{2}(X_LF);
% Fval_HF = Model_function{1}(X_HF);
% 
% for iteration = 2: 39
% 
%     load(['iteration',num2str(iteration),'.mat']);
% 
%     [X_HF,Fval_HF,Con_HF,Coneq_HF,Vio_HF,Ks_HF]=dataLoad(data_library_HF);
%     [X_LF,Fval_LF,Con_LF,Coneq_LF,Vio_LF,Ks_LF]=dataLoad(data_library_LF);
% 
%     predict_function=interpRadialBasisPreModel(X_HF,Fval_HF);
% 
%     predict_function_RBFMF=interpRadialBasisMultiFidelityPreModel(X_HF,Fval_HF,[],X_LF,Fval_LF,[]);
% 
%     X_test=lhsdesign(50,variable_number).*(up_bou-low_bou)+low_bou;
% 
%     Fval_check=object_function(X_test);
%     error_check=sum((mean(Fval_check)-Fval_check).^2);
% 
%     error_SF=sum((predict_function(X_test)-Fval_check).^2);
%     error_MF=sum((predict_function_RBFMF(X_test)-Fval_check).^2);
% 
%     R_sq_SF=1-error_SF/error_check;
%     R_sq_MF=1-error_MF/error_check;
% 
%     fprintf('%s SF %f MF %f\n',num2str(iteration),R_sq_SF,R_sq_MF)
% 
% end

