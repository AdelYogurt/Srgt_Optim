function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=muncvdZDT3(vn)
% ZTD problem 
%
if nargin < 1,vn=[];end
if isempty(vn),vn=2;end
obj_fcn=@(x)objFcn(x,vn);
nonlcon_fcn=[];
vari_num=vn;low_bou=zeros(1,vn);up_bou=ones(1,vn);
A=[];b=[];Aeq=[];beq=[];
sn=1000;
t=[
    linspace(0.000000000000000,0.082985658931624,sn/5),...
    linspace(0.182228725594696,0.257766074719318,sn/5),...
    linspace(0.409313674672817,0.453881948115847,sn/5),...
    linspace(0.618396794416386,0.652524016312189,sn/5),...
    linspace(0.823331795926068,0.851826519735778,sn/5)]';
x_best=[t,zeros(sn,vn-1)];obj_best=[t,1-sqrt(t)-t.*sin(10*pi*t)];
end

function obj=objFcn(x,vn)
obj(:,1)=x(:,1);
g=1+9*(sum(x(:,2:vn),2)/(vn-1));
obj(:,2)=g.*(1-sqrt(x(:,1)./g)-(x(:,1)./g).*sin(10*pi*x(:,1)));
end