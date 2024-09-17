function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=snglVDRS(vn)
% Rastrigin problem
% multi local mFinimum problem
%
if nargin < 1,vn=[];end
if isempty(vn),vn=2;end
obj_fcn=@(x)objFcn(x,vn);
nonlcon_fcn=[];
vari_num=vn;low_bou=-ones(1,vn);up_bou=ones(1,vn);
A=[];b=[];Aeq=[];beq=[];
x_best=zeros(1,vn);obj_best=-vn;
end

function obj=objFcn(x,vn)
obj=sum(x.^2-10*cos(2*pi*x),2);
end