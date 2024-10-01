function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc16_F16()
% problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=16;low_bou=-1*ones(1,vari_num);up_bou=zeros(1,vari_num);
A=[];b=[];Aeq=[];beq=[];
x_best=[0.5608,0.1000,0.6608,0.8000,0.9608,0.9997,0.7608,0.6000,1.0000,0.4000];obj_best=2.4968;
end

function obj=objFcn(x)
AM=[1,0,0,1,0,0,1,1,0,0,0,0,0,0,0,1;
    0,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0;
    0,0,1,0,0,0,1,0,1,1,0,0,0,1,0,0;
    0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0;
    0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,1;
    0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0;
    0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0;
    0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0;
    0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1;
    0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0;
    0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0;
    0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0;
    0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0;
    0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0;
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0;
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1;];
X=x.^2+x+1;obj=zeros(size(x,1),1);
for i=1:16
    for j=1:16
        obj=obj+AM(i,j).*X(:,i).*X(:,j);
    end
end
end