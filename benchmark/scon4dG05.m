function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=scon4dG05()
% CEC 2006 problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=@(x)nonlconFcn(x);
vari_num=4;low_bou=[0, 0, -0.55, -0.55];up_bou=[1200, 1200, 0.55, 0.55];
A=[ 0 0 1 -1;
    0 0 -1 1;];
b=[-0.55;-0.55];
Aeq=[];beq=[];
x_best=[0.679945319505338   1.026067132980041   0.000118876364740  -0.000396233553086]*1e3;obj_best=5.1265e+03;
end

function obj=objFcn(x)
obj=3*x(:,1)+1e-6*x(:,1).^3+2*x(:,2)+2e-6/3*x(:,2).^3;
end

function [con,coneq]=nonlconFcn(x)
coneq=[];
g1=1000*sin(-x(:,3)-0.25)+1000*sin(-x(:,4)-0.25)+894.8-x(:,1);
g2=1000*sin(x(:,3)-0.25)+1000*sin(x(:,3)-x(:,4)-0.25)+894.8-x(:,2);
g3=1000*sin(x(:,4)-0.25)+1000*sin(x(:,4)-x(:,3)-0.25)+1294.8;
con=[g1,g2,g3];
end