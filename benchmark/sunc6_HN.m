function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=sunc6_HN()
% Hartman problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=[];
vari_num=6;low_bou=zeros(1,6);up_bou=ones(1,6);
A=[];b=[];Aeq=[];beq=[];
x_best=[0.2017;0.1500;0.4769;0.2753;0.3117;0.6573];obj_best=-3.3224;
end

function obj=objFcn(x)
coef=[
    1 10   3   17   3.5 1.7 8  1   0.1312 0.1696 0.5569 0.0124 0.8283 0.5886;
    2 0.05 10  17   0.1 8   14 1.2 0.2329 0.4135 0.8307 0.3736 0.1004 0.9991;
    3 3    3.5 1.7  10  17  8  3   0.2348 0.1451 0.3522 0.2883 0.3047 0.6650;
    4 17   8   0.05 10  0.1 14 3.2 0.4047 0.8828 0.8732 0.5743 0.1091 0.0381;];
alpha=coef(:,2:7)';
c=coef(:,8);
p=coef(:,9:14);
obj=0;
for i=1:4
    hari=(x-p(i,:)).^2*alpha(:,i);
    obj=c(i)*exp(-hari)+obj;
end
obj=-obj;
end