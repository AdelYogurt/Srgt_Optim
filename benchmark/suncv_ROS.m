function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=suncv_ROS(vn)
% Rosenbrock problem
%
if nargin < 1,vn=[];end
if isempty(vn),vn=2;end
obj_fcn=@(x)objFcn(x,vn);
nonlcon_fcn=[];
vari_num=vn;low_bou=zeros(1,vn);up_bou=ones(1,vn);
A=[];b=[];Aeq=[];beq=[];
x_best=ones(1,vn);obj_best=0;
end

function obj=objFcn(x,vn)
obj=sum(100*(x(:,2:vn)-x(:,1:(vn-1)).^2).^2+(x(:,1:(vn-1))-1).^2,2);
end