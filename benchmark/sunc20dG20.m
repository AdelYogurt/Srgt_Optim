function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc20dDG20()
% Griewank problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=20;A=[];b=[];Aeq=[];beq=[];
low_bou=ones(1,vari_num)*-600;up_bou=ones(1,vari_num)*600;
x_best=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];obj_best=0;
end

function obj=objFcn(x)
obj=sum(x.^2,2)/4000-prod(cos(x./sqrt(1:20)),2)+1;
end