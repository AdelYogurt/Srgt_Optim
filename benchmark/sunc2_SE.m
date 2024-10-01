function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc2_SE()
% Sasena problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=2;low_bou=[0,0];up_bou=[5,5];
A=[];b=[];Aeq=[];beq=[];
x_best=[2.5044,2.5778];obj_best=-1.4565;
end

function obj=objFcn(x)
x1=x(:,1);x2=x(:,2);
obj=2+0.01*(x2-x1.^2).^2+(1-x1).^2+2*(2-x2).^2+7*sin(0.5*x1).*sin(0.7*x1.*x2);
end