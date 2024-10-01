function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc2_BR()
% Branin problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=2;low_bou=[-5,10];up_bou=[0,15];
A=[];b=[];Aeq=[];beq=[];
x_best=[-3.1416,12.2750];obj_best=0.3979;
end

function obj=objFcn(x)
x1=x(:,1);x2=x(:,2);
obj=(x2-5.1/4/pi/pi*x1.^2+5/pi*x1-6).^2+10*(1-1/8/pi)*cos(x1)+10;
end