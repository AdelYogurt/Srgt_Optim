function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc10dA10()
% Ackley problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=10;low_bou=zeros(1,10);up_bou=ones(1,10);
A=[];b=[];Aeq=[];beq=[];
x_best=[0.5608,0.1000,0.6608,0.8000,0.9608,0.9997,0.7608,0.6000,1.0000,0.4000];obj_best=2.4968;
end

function obj=objFcn(x)
s=[1.3,0.1,1.4,0.8,1.7,1,1.5,0.6,2,0.4];
obj=-20*exp(-0.2*sqrt(sum((x-s).^2,2)/10))-exp(sum(cos(2*1.3*pi*(x-s))/10,2))+20+exp(1);
end

function obj=objFcnLF(x)
obj=-20*exp(-0.2*sqrt(sum(x.^2,2)/10))-exp(sum(cos(2*pi*x)/10,2))+20+exp(1);
end