function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc2_HIM()
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=2;low_bou=[-5,-5];up_bou=[5,5];
A=[];b=[];Aeq=[];beq=[];
x_best=[-3.6483,-0.0685];obj_best=272.5563;
end

function obj=objFcn(x)
x1=x(:,1);
x2=x(:,2);
obj=(x1.^2+x2-11).^2+(x2.^2+x1+20).^2;
end

function obj=objFcnLF(x)
x1=x(:,1);
x2=x(:,2);
obj=(0.9*x1.^2+0.8*x2-11).^2+(0.8*x2.^2+0.9*x1+20).^2-(x1+1).^2;
end