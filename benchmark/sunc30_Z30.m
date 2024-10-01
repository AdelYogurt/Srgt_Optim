function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc30_Z30()
% Zakharov problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=30;A=[];b=[];Aeq=[];beq=[];
low_bou=-5*ones(1,vari_num);up_bou=10*ones(1,vari_num);
x_best=zeros(1,vari_num);obj_best=0;
end

function obj=objFcn(x)
obj=sum(x.^2,2)+(sum(0.5*(1:30).*x,2)).^2+(sum(0.5*(1:30).*x,2)).^4;
end