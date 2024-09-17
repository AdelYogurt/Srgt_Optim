function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc2dGF()
% Generalized polynomial problem 
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=2;low_bou=[-5,-5];up_bou=[5,5];
A=[];b=[];Aeq=[];beq=[];
x_best=[3.0000,0.5000];obj_best=0;
end

function obj=objFcn(x)
c1=1.5;c2=2.25;c3=2.625;
x1=x(:,1);x2=x(:,2);
u1=c1-x1.*(1-x2);
u2=c2-x1.*(1-x2.^2);
u3=c3-x1.*(1-x2.^3);
obj=u1.^2+u2.^2+u3.^2;
end