function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=scon7_G09()
% CEC 2006 problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=@(x)nonlconFcn(x);
vari_num=7;low_bou=-10*ones(1,7);up_bou=10*ones(1,7);
A=[];b=[];Aeq=[];beq=[];
x_best=[2.330499020860739   1.951372301776673  -0.477543604918950   4.365726520068069  -0.624486603992181   1.038130162630999  1.594226316347420];obj_best=6.806300573697980e+02;
end

function obj=objFcn(x)
obj=(x(:,1)-10).^2+5*(x(:,2)-12).^2+x(:,3).^4+3*(x(:,4)-11).^2+10*x(:,5).^6+7*x(:,6).^2+x(:,7).^4-4*x(:,6).*x(:,7)-10*x(:,6)-8*x(:,7);
end

function [con,coneq]=nonlconFcn(x)
coneq=[];
con(:,1)=-127+2*x(:,1).^2+3*x(:,2).^4+x(:,3)+4*x(:,4).^2+5*x(:,5);
con(:,2)=-282+7*x(:,1)+3*x(:,2)+10*x(:,3).^2+x(:,4)-x(:,5);
con(:,3)=-196+23*x(:,1)+x(:,2).^2+6*x(:,6).^2-8*x(:,7);
con(:,4)=4*x(:,1).^2+x(:,2).^2-3*x(:,1).*x(:,2)+2*x(:,3).^2+5*x(:,6)-11*x(:,7);
end