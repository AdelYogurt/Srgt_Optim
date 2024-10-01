function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=scon2_G06()
% CEC 2006 problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=@(x)nonlconFcn(x);
vari_num=2;low_bou=[13,0];up_bou=[100,100];
A=[];b=[];Aeq=[];beq=[];
x_best=[14.0950,0.8430];obj_best=-6.9618e+03;
end

function obj=objFcn(x)
obj=(x(:,1)-10).^3+(x(:,2)-20).^3;
end

function [con,coneq]=nonlconFcn(x)
g1=-(x(:,1)-5).^2-(x(:,2)-5).^2+100;
g2=(x(:,1)-6).^2+(x(:,2)-5).^2-82.81;
con=[g1,g2];
coneq=[];
end