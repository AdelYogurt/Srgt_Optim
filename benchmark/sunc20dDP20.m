function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc20dDP20()
% Dixon and Price problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=20;low_bou=ones(1,vari_num)*-30;up_bou=ones(1,vari_num)*30;
A=[];b=[];Aeq=[];beq=[];
x_best=[2.1333,0.5000,2.0000,1.2000,0.4000,0.2000,1.4000,0.3000,1.6000,0.6000,0.8000,1.0000,1.3000,1.9000,0.7000,1.6000,0.3009,1.1214,2.1035,1.1725];obj_best=0.6667;
end

function obj=objFcn(x)
s=[1.8,0.5,2,1.2,0.4,0.2,1.4,0.3,1.6,0.6,0.8,1,1.3,1.9,0.7,1.6,0.3,1.1,2,1.4];
x=x-s;
obj=(x(:,1)-1).^2+sum((2:20).*(2*x(:,2:end).^2-x(:,1:end-1)).^2,2);
end

function obj=objFcnLF(x)
obj=(x(:,1)-1).^2+sum((2:20).*(2*x(:,2:end).^2-x(:,1:end-1)).^2,2);
end