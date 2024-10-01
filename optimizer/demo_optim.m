clc;
clear;
close all;

%% add path

addpath ../benchmark/
addpath ../common/
addpath ../LHD/
addpath ../ML/
addpath ../surrogate/

%% define problem

% % unconstraint problem GF
% objcon_fcn=@(x) objconFcnGP(x);
% vari_num=2;low_bou=[-2,-2];up_bou=[2,2];

% % constraint problem G06
% objcon_fcn=@(x) objconFcnG06(x);
% vari_num=2;low_bou=[13,0];up_bou=[100,100];

%% optimize problem

% NFE_max=100;iter_max=150;obj_tol=1e-6;con_tol=0;

% optimizer=OptimDCSRBF(NFE_max,iter_max,obj_tol,con_tol);
% optimizer=OptimFSRBF(NFE_max,iter_max,obj_tol,con_tol);
% optimizer=OptimKRGCDE(NFE_max,iter_max,obj_tol,con_tol);
% optimizer=OptimSKO(NFE_max,iter_max,obj_tol,con_tol);
% optimizer=OptimPAKMCA(NFE_max,iter_max,obj_tol,con_tol);
% optimizer=OptimRBFCDE(NFE_max,iter_max,obj_tol,con_tol);
% optimizer=OptimSACORS(NFE_max,iter_max,obj_tol,con_tol);
% optimizer=OptimTRARSM(NFE_max,iter_max,obj_tol,con_tol);

% optimizer.FLAG_CONV_JUDGE=true;
% optimizer.FLAG_DRAW_FIGURE=true;
% optimizer.datalib_filestr='lib.mat';
% optimizer.dataoptim_filestr='optim.mat';

% [x_best,obj_best,NFE,output]=optimizer.optimize(objcon_fcn,vari_num,low_bou,up_bou);

% datalib=output.datalib;
% plot(1:length(datalib.Obj),datalib.Obj(datalib.Best_idx),'o-')
% line(1:length(datalib.Obj),datalib.Obj,'Marker','o','Color','g')
% line(1:length(datalib.Vio),datalib.Vio,'Marker','o','Color','r')

%% function

function [obj,con,coneq]=objconFcnGP(x)
% x_best=[0,-1];obj_best=3;
%
x1=x(:,1);x2=x(:,2);
obj=(1+(x1+x2+1).^2.*...
    (19-14*x1+3*x1.^2-14*x2+6*x1.*x2+3*x2.^2)).*...
    (30+(2*x1-3*x2).^2.*(18-32*x1+12*x1.^2+48*x2-36*x1.*x2+27*x2.^2));
con=[];
coneq=[];
end

function [obj,con,coneq]=objconFcnG06(x)
% x_best=[14.0950,0.8430]; obj_best=-6.9618e+03;
%
obj=(x(:,1)-10).^3+(x(:,2)-20).^3;
g1=-(x(:,1)-5).^2-(x(:,2)-5).^2+100;
g2=(x(:,1)-6).^2+(x(:,2)-5).^2-82.81;
con=[g1,g2];
coneq=[];
end
