function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=muncv_ZDT6(vn)
% ZTD problem 
%
if nargin < 1,vn=[];end
if isempty(vn),vn=2;end
obj_fcn=@(x)objFcn(x,vn);
nonlcon_fcn=[];
vari_num=vn;low_bou=zeros(1,vn);up_bou=ones(1,vn);
A=[];b=[];Aeq=[];beq=[];
sn=1000;t=linspace(0,1,sn)';
x_best=[t,zeros(sn,vn-1)];
f1=1-exp(-4*t).*sin(6*pi*t).^6;
obj_best=[f1,1-f1.^2];
end

function obj=objFcn(x,vn)
obj(:,1)=1-exp(-4*x(:,1)).*sin(6*pi*x(:,1)).^6;
g=1+9*((sum(x(:,2:vn),2)/(vn-1))).^0.25;
obj(:,2)=g.*(1-(obj(:,1)./g).^2);
end