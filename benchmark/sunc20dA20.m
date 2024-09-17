function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc20dA20()
% Ackley problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=20;low_bou=zeros(1,20);up_bou=ones(1,20);
A=[];b=[];Aeq=[];beq=[];
x_best=[0.5357,0.1000,0.6357,0.8000,0.9357,1.0000,0.7357,0.6000,0.4714,0.4000,0.5357,0.3000,0.7357,0.9000,0.3714,1.0000,0.9357,0.7000,0.5714,0.5000];obj_best=-0.6291;
end

function obj=objFcn(x)
s=[1.3,0.1,1.4,0.8,1.7,1,1.5,0.6,2,0.4,1.3,0.3,1.5,0.9,1.9,1.1,1.7,0.7,2.1,0.5];
obj=-20*exp(-0.2*sqrt(sum((x-s).^2,2)/10))-exp(sum(cos(2*1.3*pi*(x-s))/10,2))+20+exp(1);
end

function obj=objFcnLF(x)
obj=-20*exp(-0.2*sqrt(sum(x.^2,2)/10))-exp(sum(cos(2*pi*x)/10,2))+20+exp(1);
end