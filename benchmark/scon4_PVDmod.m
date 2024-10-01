function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=scon4_PVDMOD()
% Modify pressure vessel design problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=@(x)nonlconFcn(x);
vari_num=4;low_bou=[1,0.625,25,25];up_bou=[1.375,1,150,240];
A=[ -1,  0,  0.0193, 0;
     0, -1, 0.00954, 0;
     0,  0,       0, 1;
    -1,  0,       0, 0;
     0, -1,       0, 0];
b=[0;0;240;-1.1;-0.6];
Aeq=[];beq=[];
x_best=[1.1000,0.6250,56.9948,51.0013];obj_best=7.1637e+03;
end

function obj=objFcn(x)
x1=x(:,1);x2=x(:,2);x3=x(:,3);x4=x(:,4);
obj=0.6224*x1.*x3.*x4+1.7781*x2.*x3.^2+3.1661*x1.^2.*x4+19.84*x1.^2.*x3;
end

function [con,coneq]=nonlconFcn(x)
x1=x(:,1);x2=x(:,2);x3=x(:,3);x4=x(:,4);
con=-pi*x3.^2.*x4-4/3*pi*x3.^3+1296000;
coneq=[];
end