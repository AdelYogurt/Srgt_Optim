function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=scon2dHaupt()
% Haupt problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=@(x)nonlconFcn(x);
vari_num=2;low_bou=[0,0];up_bou=[3.7,4];
A=[];b=[];Aeq=[];beq=[];
x_best=[2.9711,3.4035];obj_best=0.8871;
end

function obj=objFcn(x)
x1=x(:,1);x2=x(:,2);
obj=(x1-3.7).^2+(x2-4).^2;
end

function [con,coneq]=nonlconFcn(x)
x1=x(:,1);x2=x(:,2);
coneq=[];
con=[x1.*sin(4*x1)+1.1*x2.*sin(2*x2),-x1-x2+3];
end