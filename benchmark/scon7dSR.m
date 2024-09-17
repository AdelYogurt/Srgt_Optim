function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=scon7dSR()
% Speed reducer design problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=@(x)nonlconFcn(x);
vari_num=7;low_bou=[2.6, 0.7, 17, 7.3, 7.3, 2.9, 5];up_bou=[3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5];
A=[];b=[];Aeq=[];beq=[];
x_best=[3.5000    0.7000   17.0000    7.3000    7.7153    3.3505    5.2867];obj_best=2.9944e+03;
end

function obj=objFcn(x)
matTemp1=3.3333*x(:,3).^2+14.9334*x(:,3)-43.0934;
matTemp2=x(:,6).^2+x(:,7).^2;
matTemp3=x(:,6).^3+x(:,7).^3;
matTemp4=x(:,4).*x(:,6).^2+x(:,5)...
    .*x(:,7).^2;
obj=0.7854*x(:,1).*x(:,2).^2.*matTemp1 ...
   -1.508*x(:,1).*matTemp2+7.477*matTemp3 ...
   +0.7854*matTemp4;
end

function [con,coneq]=nonlconFcn(x)
coneq=[];
matTemp1=sqrt((745*x(:,4)./(x(:,2).*x(:,3))).^2+16.91e6);
matTemp2=sqrt((745*x(:,5)./(x(:,2).*x(:,3))).^2+157.5e6);
con(:,1)=(27-x(:,1).*x(:,2).^2.*x(:,3) ...
    ) / 27;
con(:,2)=(397.5-x(:,1).*x(:,2).^2 ...
    .*x(:,3).^2) / 397.5;
con(:,3)=(1.93-(x(:,2).*x(:,6).^4 ...
    .*x(:,3)./(x(:,4).^3))) / 1.93;
con(:,4)=(1.93-(x(:,2).*x(:,7).^4 ...
    .*x(:,3)./(x(:,5).^3))) / 1.93;
con(:,5)=(matTemp1./(0.1*x(:,6).^3)-1100) / 1100;
con(:,6)=(matTemp2./(0.1*x(:,7).^3)-850) / 850;
con(:,7)=(x(:,2).*x(:,3)-40) / 40;
con(:,8)=(5-x(:,1)./x(:,2)) / 5;
con(:,9)=(x(:,1)./x(:,2)-12) / 12;
con(:,10)=(1.9+1.5*x(:,6)-x(:,4)) / 1.9;
con(:,11)=(1.9+1.1*x(:,7)-x(:,5)) / 1.9;
end