function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc30dA30()
% Ackley problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=30;low_bou=-15*ones(1,vari_num);up_bou=20*ones(1,vari_num);
A=[];b=[];Aeq=[];beq=[];
x_best=zeros(1,vari_num);obj_best=-20-exp(1);
end

function obj=objFcn(x)
n=30;sum1=0;sum2=0;
for i=1:n
    sum1=sum1+x(:,i).^2;
    sum2=sum2+cos((2*pi)*x(:,i));
end
obj=-20*exp(-0.2*sqrt(1/n*sum1))-exp(1/n*sum2);
end