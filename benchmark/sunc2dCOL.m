function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc2dCOL()
% Colville problem 
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=4;low_bou=zeros(1,4);up_bou=ones(1,4);
A=[];b=[];Aeq=[];beq=[];
x_best=[1.0000,1.0000,0.1667,0.0000];obj_best=-9.3361;
end

function obj=objFcn(x)
x1=x(:,1);x2=x(:,2);x3=x(:,3);x4=x(:,4);
obj=100*(x1.^2-x2).^2+(x1-1).^2+(x3-1).^2+90*(x3.^2-x4).^2+...
    10.1*((x2-1).^2-(x4-1).^2)+19.8*(x2-1).*(x4-1);
end

function obj=objFcnLF(x)
obj=objFcn([0.8,0.8,0.5,0.5].*x);
end