function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc5dST()
% Styblinski-Tang problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=5;low_bou=-5*ones(1,5);up_bou=5*ones(1,5);
A=[];b=[];Aeq=[];beq=[];
x_best=[-2.6235,-2.3135,-2.4335,-2.7435,-2.5835];obj_best=-195.8308;
end

function obj=objFcn(x)
x=x-[0.28,0.59,0.47,0.16,0.32];
obj=0.5*sum(x.^4-16*x.^2+5*x,2);
end

function obj=objFcnLF(x)
obj=0.5*sum(x.^4-16*x.^2+5*x,2);
end