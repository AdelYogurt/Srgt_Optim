function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc2_SC()
% problem 
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=2;low_bou=[-2,-2];up_bou=[2,2];
A=[];b=[];Aeq=[];beq=[];
x_best=[0.0898,-0.7127];obj_best=-1.0316;
end

function obj=objFcn(x)
x1=x(:,1);x2=x(:,2);
obj=4*x1.^2-2.1*x1.^4+x1.^6/3+x1.*x2-4*x2.^2+4*x2.^4;
end