function [obj_fcn,vari_num,A,b,Aeq,beq,low_bou,up_bou,nonlcon_fcn,x_best,obj_best]=scon5_ICE()
% Internal Combustion Engine Design problem
%
obj_fcn=@(x)objFcn(x);
nonlcon_fcn=@(x)nonlconFcn(x);
vari_num=5;low_bou=[75,6,25,35,5000];up_bou=[90,12,35,45,10000];
A=[];b=[];Aeq=[];beq=[];
x_best=[83.33,9.45,30.99,37.34,6070];obj_best=-55.67;
end

function obj=objFcn(x)
b=x(:,1);
c_r=x(:,2);
d_E=x(:,3);
d_I=x(:,4);
omega=x(:,5).*1e-3;

K_0=1./120;
rou=1.225;
gamma=1.33;
V=1.859.*1e6;
Q=43958;
N_c=4;
C_s=0.44;
A_f=14.6;

eta_vb=0.637+0.13.*omega-0.014.*omega.^2+0.00066.*omega.^3;
eta_vb(omega <= 5.25)=1.067-0.038.*exp(omega(omega <= 5.25)-5.25);

eta_tad=0.8595.*(1-c_r.^(-0.33));
eta_V=eta_vb.*(1+5.96.*1e-3.*omega.^2)./(1+((9.428.*1e-5).*(4.*V./pi./N_c./C_s).*(omega./d_I.^2)).^2);
S_V=0.83.*((8+4.*c_r)+1.5.*(c_r-1).*(pi.*N_c./V).*b.^3)./((2+c_r).*b);
eta_t=eta_tad-S_V.*(1.5./omega).^0.5;
V_P=(8.*V./pi./N_c).*omega.*b.^(-2);
FMEP=4.826.*(c_r-9.2)+(7.97+0.253.*V_P+9.7.*(1e-6).*V_P.^2);

obj=K_0.*(FMEP-(rou.*Q./A_f).*eta_t.*eta_V).*omega;
end

function [con,coneq]=nonlconFcn(x)
b=x(:,1);
c_r=x(:,2);
d_E=x(:,3);
d_I=x(:,4);
omega=x(:,5).*1e-3;

K_1=1.2;
K_2=2;
K_3=0.82;
K_4=(-1e-12+30.99)./37.34; K_5=0.89;
K_6=0.6;
K_7=6.5;
K_8=230.5;
L_1=400;
L_2=200;
rou=1.225;
gamma=1.33;
V=1.859.*1e6;
Q=43958;
N_c=4;
C_s=0.44;
A_f=14.6;

S_V=0.83.*((8+4.*c_r)+1.5.*(c_r-1).*(pi.*N_c./V).*b.^3)./((2+c_r).*b);
eta_tw=0.8595.*(1-c_r.^(-0.33))-S_V;

g1=K_1.*N_c.*b-L_1;
g2=(4.*K_2.*V./pi./N_c./L_2).^0.5-b;
g3=d_I+d_E-K_3.*b;
g4=K_4.*d_I-d_E;
g5=d_E-K_5.*d_I;
g6=9.428.*1e-5.*(4.*V./pi./N_c).*(omega./d_I.^2)-K_6.*C_s;
g7=c_r-13.2+0.045.*b;
g8=omega-K_7;
g9=3.6.*1e6-K_8.*Q.*eta_tw;

con=[g1,g2,g3,g4,g5,g6,g7,g8,g9];
coneq=[];
end