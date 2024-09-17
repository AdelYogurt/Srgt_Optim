function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=muncvdZDT1(vn)
% ZTD problem 
%
if nargin < 1,vn=[];end
if isempty(vn),vn=2;end
obj_fcn=@(x)objFcn(x,vn);
nonlcon_fcn=[];
vari_num=vn;low_bou=zeros(1,vn);up_bou=ones(1,vn);
A=[];b=[];Aeq=[];beq=[];
sn=1000;t=linspace(0,1,sn)';
x_best=[t,zeros(sn,vn-1)];obj_best=[t,1-sqrt(t)];
end

function obj=objFcn(x,vn)
obj(:,1)=x(:,1);
gmat=1+9*(sum(x(:,2:vn),2)/(vn-1));
obj(:,2)=gmat.*(1-sqrt(x(:,1)./gmat));
end