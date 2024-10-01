function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc2_PK()
% Peak problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=2;low_bou=[-3,-3];up_bou=[3,3];
A=[];b=[];Aeq=[];beq=[];
x_best=[0.2283,-1.6255]; obj_best=-6.5511;
end

function obj=objFcn(x)
x1=x(:,1);x2=x(:,2);
obj=3*(1-x1).^2.*exp(-(x1.^2)-(x2+1).^2) ...
    -10*(x1/5-x1.^3-x2.^5).*exp(-x1.^2-x2.^2) ...
    -1/3*exp(-(x1+1).^2-x2.^2);
end