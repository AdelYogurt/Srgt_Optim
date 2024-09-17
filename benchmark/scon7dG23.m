function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=scon7dG23()
% CEC 2006 problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=@(x)nonlconFcn(x);
vari_num=9;low_bou=[0 0 0 0 0 0 0 0 0.01];up_bou=[300 300 100 200 100 300 100 200 0.03];
A=[];b=[];Aeq=[];beq=[];
x_best=[0,0,0,68.8087,100.0000,0,0,200.0000,0.0100];obj_best=-3900;
end

function obj=objFcn(x)
x1=x(:,1);
x2=x(:,2);
x3=x(:,3);
x4=x(:,4);
x5=x(:,5);
x6=x(:,6);
x7=x(:,7);
x8=x(:,8);
x9=x(:,9);
obj=-9.*x5-15.*x8+6.*x1+16.*x2+10.*(x6+x7);
end

function [con,coneq]=nonlconFcn(x)
coneq=[];
x1=x(:,1);
x2=x(:,2);
x3=x(:,3);
x4=x(:,4);
x5=x(:,5);
x6=x(:,6);
x7=x(:,7);
x8=x(:,8);
x9=x(:,9);
con(:,1)=x9.*x3+0.02.*x6-0.025.*x5;
con(:,2)=x9.*x4+0.02.*x7-0.015.*x8;
con(:,3)=x1+x2-x3-x4;
con(:,4)=0.03.*x1+0.01.*x2-x9.*(x3+x4);
con(:,5)=x3+x6-x5;
con(:,6)=x4+x7-x8;
end