function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=scon9dG18()
% CEC 2006 problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=@(x)nonlconFcn(x);
vari_num=9;low_bou=[-10,-10,-10,-10,-10,-10,-10,-10,0];up_bou=[10,10,10,10,10,10,10,10,20];
A=[];b=[];Aeq=[];beq=[];
x_best=[-0.657776192427943163,-0.153418773482438542,0.323413871675240938,-0.946257611651304398,-0.657776194376798906,-0.753213434632691414,0.323413874123576972,-0.346462947962331735,0.59979466285217542];
obj_best=-0.866025403784439;
end

function obj=objFcn(x)
obj=-0.5*(x(:,1).*x(:,4)-x(:,2).*x(:,3)+x(:,3).*x(:,9)-x(:,5).*x(:,9)+x(:,5).*x(:,8)-x(:,6).*x(:,7));
end

function [con,coneq]=nonlconFcn(x)
coneq=[];
con(:,1)=x(:,3).^2+x(:,4).^2-1;
con(:,2)=x(:,9).^2-1;
con(:,3)=x(:,5).^2+x(:,6).^2-1;
con(:,4)=x(:,1).^2+(x(:,2)-x(:,9)).^2-1;
con(:,5)=(x(:,1)-x(:,5)).^2+(x(:,2)-x(:,6)).^2-1;
con(:,6)=(x(:,1)-x(:,7)).^2+(x(:,2)-x(:,8)).^2-1;
con(:,7)=(x(:,3)-x(:,5)).^2+(x(:,4)-x(:,6)).^2-1;
con(:,8)=(x(:,3)-x(:,7)).^2+(x(:,4)-x(:,8)).^2-1;
con(:,9)=x(:,7).^2+(x(:,8)-x(:,9)).^2-1;
con(:,10)=x(:,2).*x(:,3)-x(:,1).*x(:,4);
con(:,11)=-x(:,3).*x(:,9);
con(:,12)=x(:,5).*x(:,9);
con(:,13)=x(:,6).*x(:,7)-x(:,5).*x(:,8);
end