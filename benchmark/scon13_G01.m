function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=scon13_G01()
% CEC 2006 problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=@(x)nonlconFcn(x);
vari_num=13;
A=[
    2   2   0   0   0   0   0   0   0   1   1   0   0;
    2   0   2   0   0   0   0   0   0   1   0   1   0;
    0   2   2   0   0   0   0   0   0   0   1   1   0;
    -8  0   0   0   0   0   0   0   0   1   0   0   0;
    0   -8  0   0   0   0   0   0   0   0   1   0   0;
    0   0   -8  0   0   0   0   0   0   0   0   1   0
    0   0   0   -2  -1  0   0   0   0   1   0   0   0;
    0   0   0   0   0   -2  -1  0   0   0   1   0   0;
    0   0   0   0   0   0   0   -2  -1  0   0   1   0;
    ];
b=[10;10;10;0;0;0;0;0;0];
Aeq=[];
beq=[];
low_bou=[0,0,0,0,0,0,0,0,0,0,0,0,0];
up_bou=[1,1,1,1,1,1,1,1,1,100,100,100,1];
x_best=[1,1,1,1,1,1,1,1,1,3,3,3,1];
obj_best=-15;
end

function obj=objFcn(x)
obj=5*sum(x(:,1:4),2)-5*sum(x(:,1:4).^2,2)-sum(x(:,5:13),2);
end

function [con,coneq]=nonlconFcn(x)
coneq=[];
con=[];
end