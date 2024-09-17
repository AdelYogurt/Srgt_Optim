function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc1dForrester()
% Forrester problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=1;low_bou=0;up_bou=1;
A=[];b=[];Aeq=[];beq=[];
x_best=0.7572;obj_best=-6.0207;
end

function obj=objFcn(x)
obj=((x.*6-2).^2).*sin(x.*12-4);
end

function obj=objFcnLF(x)
A=0.5;B=10;C=-5;
obj=A*((x.*6-2).^2).*sin(x.*12-4)+B*(x-0.5)+C;
end