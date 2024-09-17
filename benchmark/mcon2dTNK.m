function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=mcon2dTNK()
% Tanaka problem 
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=@(x)nonlconFcn(x);
vari_num=2;low_bou=zeros(1,2);up_bou=ones(1,2)*pi;
A=[];b=[];Aeq=[];beq=[];

x_best=[];obj_best=[];

% sn=1000;t=linspace(0,1,sn)';
% x_best=[t,zeros(sn,vn-1)];
% f1=1-exp(-4*t).*sin(6*pi*t).^6;
% obj_best=[f1,1-f1.^2];
end

function obj=objFcn(x)
obj=x;
end

function [con,coneq]=nonlconFcn(x)
x1=x(:,1);x2=x(:,2);
con(:,1)=-(x1.^2+x2.^2-1-0.1*cos(16*atan(x1./x2)));
con(:,2)=(x1-0.5).^2+(x2-0.5).^2-0.5;
con(:,x1 == 0&x2 == 0)=1.1;  
coneq=[];
end