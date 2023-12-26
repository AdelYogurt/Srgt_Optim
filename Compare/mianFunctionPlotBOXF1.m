clc;
clear;
close all hidden;

% object_function=@func_G2_GP;
% variable_number=2;
% A=[];
% B=[];
% Aeq=[];
% Beq=[];
% low_bou=[-1;-1]*2;
% up_bou=[1;1]*2;
% nonlcon_function=[];
% cheapcon_function=[];

% variable_number=2;
% object_function=@func_G3_G06;
% A=[];
% B=[];
% Aeq=[];
% Beq=[];
% low_bou=[13;0];
% up_bou=[100;100];
% nonlcon_function=@nonlcon_G06;
% cheapcon_function=[];

% variable_number=5;
% object_function=@func_G3_ICE;
% A=[];
% B=[];
% Aeq=[];
% Beq=[];
% low_bou=[50;2;20;20;4000];
% up_bou=[1e2;10;50;50;10000];
% nonlcon_function=@nonlcon_ICE;
% cheapcon_function=[];

% variable_number=2;
% object_function=@(x) functionBRObject(x);
% A=[];
% B=[];
% Aeq=[];
% Beq=[];
% low_bou=[-5;10];
% up_bou=[0;15];
% nonlcon_function=[];
% cheapcon_function=[];
% model_function=[];

variable_number=6;
object_function=@(x) functionHNObject(x);
A=[];
B=[];
Aeq=[];
Beq=[];
low_bou=zeros(variable_number,1);
up_bou=ones(variable_number,1);
nonlcon_function=[];
cheapcon_function=[];
model_function=[];

repeat_time=20;

% dataget
SRS_value_data=zeros(1,repeat_time);
SRS_NFE_data=zeros(1,repeat_time);

for repeat_index=1:repeat_time
%     x0=low_bou+(up_bou-low_bou).*rand(length(low_bou),1);
    
    delete('optimalSurrogate_SRBF_SVM_result.txt');
    [SRS_x,SRS_fval,SRS_NFE,SRS_output]=optimalSurrogateSRBFSVMM...
        (object_function,variable_number,low_bou,up_bou,nonlcon_function,...
        cheapcon_function,model_function);
    
    SRS_value_data(repeat_index)=SRS_fval;
    SRS_NFE_data(repeat_index)=SRS_NFE;
end

% average
SRS_value_average=mean(SRS_value_data);
SRS_NFE_average=mean(SRS_NFE_data);

% median
SRS_value_median=median(SRS_value_data);
SRS_NFE_median=median(SRS_NFE_data);

% variance
SRS_value_var=var(SRS_value_data);
SRS_NFE_var=var(SRS_NFE_data);

% change range
SRS_value_max=max(SRS_value_data);
SRS_NFE_max=max(SRS_NFE_data);
SRS_value_min=min(SRS_value_data);
SRS_NFE_min=min(SRS_NFE_data);

% boxplot
box_value_data=[SRS_value_data'];
box_NFE_data=[SRS_NFE_data'];
g_SRS = repmat({'SRS'},repeat_time,1);
label=[g_SRS];

value_figure=figure(1);
name=mfilename;
boxplot(box_value_data,label);
title('Objective Function Value at Solution')
xlabel('Method')
ylabel('Value')
NFE_figure=figure(2);
boxplot(box_NFE_data,label);
title('Number of Function Evaluation')
xlabel('Method')
ylabel('Value')

% output result
% print(value_figure,[name(13:end),'_value'],'-dpng');
% print(NFE_figure,[name(13:end),'_NFE'],'-dpng');
mat_data=zeros(2,5);
mat_data(:,1)=[SRS_value_average;
    SRS_NFE_average;];
mat_data(:,2)=[SRS_value_median;
    SRS_NFE_median;];
mat_data(:,3)=[SRS_value_var;
    SRS_NFE_var;];
mat_data(:,4)=[SRS_value_min;
    SRS_NFE_min;];
mat_data(:,5)=[SRS_value_max;
    SRS_NFE_max;];
disp('均值 中位数 标准差 变化范围1 变化范围2');
disp(mat_data);

% save([name(13:end),'_result'])