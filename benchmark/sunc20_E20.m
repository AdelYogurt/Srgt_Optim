function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc20_E20()
% Ellipsoid problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=20;A=[];b=[];Aeq=[];beq=[];
low_bou=ones(1,vari_num)*-20;up_bou=ones(1,vari_num)*20;
x_best=[1.8000,0.4000,2.0000,1.2000,1.4000,0.6000,1.6000,0.2000,0.8000,1.0000,1.3000,1.1000,2.0000,1.4000,0.5000,0.3000,1.6000,0.7000,0.3000,1.9000];obj_best=0;
end

function obj=objFcn(x)
ss=[1.8,0.4,2,1.2,1.4,0.6,1.6,0.2,0.8,1,1.3,1.1,2,1.4,0.5,0.3,1.6,0.7,0.3,1.9];
sh=[0.3,0.4,0.2,0.6,1,0.9,0.2,0.8,0.5,0.7,0.4,0.3,0.7,1,0.9,0.6,0.2,0.8,0.2,0.5];
obj=sum((1:20).*sh.*(x-ss).^2,2);
end

function obj=objFcnLF(x)
obj=sum((1:20).*x.^2,2);
end