function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=scon10_G02mod()
% CEC 2006 problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=@(x)nonlconFcn(x);
vari_num=10;low_bou=zeros(1,10);up_bou=10*ones(1,10);
A=[];b=[];Aeq=[];beq=[];
x_best=[3.116140811878899,3.090864027952656,3.065674810682763,3.040414982246053,3.014922254392321,2.989019046699154,2.962489200838383,1.464885624078708,0.014643201033883,0.014588277700111];
obj_best=-0.533976523978862;
end

function obj=objFcn(x)
obj=-abs((sum(cos(x).^4,2)-2*prod(cos(x).^2,2))./sqrt(sum((1:10).*x.^2,2)));
end

function [con,coneq]=nonlconFcn(x)
g1=0.75-prod(x,2);
boolean=g1 >= 0;
g1(boolean)=log(1+g1(boolean));
g1(~boolean)=-log(1-g1(~boolean));
g2=sum(x,2)-7.5*10;

con=[g1,g2];
coneq=[];
end