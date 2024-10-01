function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=scon20_G02()
% CEC 2006 problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=@(x)nonlconFcn(x);
vari_num=20;low_bou=zeros(1,20);up_bou=10*ones(1,20);
A=[];b=[];Aeq=[];beq=[];
x_best=[3.16246061572185,3.12833142812967,3.09479212988791,3.06145059523469,3.02792915885555,2.99382606701730,2.95866871765285,2.92184227312450,0.49482511456933,0.48835711005490,0.48231642711865,0.47664475092742,0.47129550835493,0.46623099264167,0.46142004984199,0.45683664767217,0.45245876903267,0.44826762241853,0.44424700958760,0.44038285956317];
obj_best=-0.80361910412559;
end

function obj=objFcn(x)
obj=-abs((sum(cos(x).^4,2)-2*prod(cos(x).^2,2))./sqrt(sum((1:20).*x.^2,2)));
end

function [con,coneq]=nonlconFcn(x)
coneq=[];
con(:,1)=0.75-prod(x,2);
con(:,2)=sum(x,2)-7.5*20;
end