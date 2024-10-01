clc;
clear;
close all hidden;

%% gen data

% fcn_LF=@(x) LFFcn(x);
% fcn_HF=@(x) HFFcn(x);
% 
% fcn_LF_bou=@(x) 0.45+sin(2.2*pi*x)/2.5;
% fcn_HF_bou=@(x) 0.5+sin(2.5*pi*x)/3;
% 
% vari_num=2;
% 
% x_LF_num=20;
% X_LF=lhsdesign(x_LF_num,vari_num);
% C_LF=fcn_LF(X_LF);
% 
% x_HF_num=10;
% X_HF=lhsdesign(x_HF_num,vari_num);
% C_HF=fcn_HF(X_HF);
% 
% low_bou=[0,0];
% up_bou=[1,1];

%% draw single-fidelity classification

% load('classify_30.mat');
% 
% classify=classifySVM(X_HF,C_HF);
% classify=classifyGPC(X_HF,C_HF);
% 
% fig_hdl=figure(1);
% displayClassify([],classify,low_bou,up_bou)
% drawFcn([],fcn_HF_bou,0,1);

%% draw multi-fidelity classification

% load('classify_30.mat');
% 
% classifymf=classifyCoGPC(X_HF,C_HF,X_LF,C_LF);
% fig_hdl=figure(1);
% displayClassify([],classify_mf,low_bou,up_bou)
% drawFcn([],fcn_HF_bou,0,1);
% drawFcn([],fcn_LF_bou,0,1);

%% drwa picture of comparison between GPC and CoGPC

% load('classify_30.mat');
% load('color.mat','my_color_map_light');
% 
% classifysf=classifyGPC(X_HF,C_HF);
% classifymf=classifyCoGPC(X_HF,C_HF,X_LF,C_LF);
% 
% % draw zero value line
% grid_num=100;
% d_bou=(up_bou-low_bou)/grid_num;
% [XMat_draw,YMat_draw]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
% 
% % generate all pred list
% X_pred=[XMat_draw(:),YMat_draw(:)];
% [~,Prblty_pred_SF]=classifysf.predict(X_pred);
% Prblty_pred_SF=reshape(Prblty_pred_SF,grid_num+1,grid_num+1);
% [~,Prblty_pred_MF]=classifymf.predict(X_pred);
% Prblty_pred_MF=reshape(Prblty_pred_MF,grid_num+1,grid_num+1);
% 
% % draw picture
% fig_hdl=figure(1);
% fig_hdl.set('Position',[488   200   680  420])
% 
% axes_hdl=subplot(1,2,1);
% [~,ctr_hdl]=contourf(axes_hdl,XMat_draw,YMat_draw,Prblty_pred_SF*2-1,'LineStyle','none');
% shading interp;
% colormap(my_color_map_light);
% hold(axes_hdl,'on');
% X_draw=0:0.01:1;
% scatter(X_HF(C_HF>0.5,1),X_HF(C_HF>0.5,2),'or')
% scatter(X_HF(C_HF<0.5,1),X_HF(C_HF<0.5,2),'ok')
% line(axes_hdl,X_draw,fcn_HF_bou(X_draw),'LineStyle','-');
% axes_hdl.set('Position',[0.0800,0.1200,0.38,0.6750],'FontSize',12,'FontName','Times New Roman');
% xlabel('${\textit{x}}_{1}$','Interpreter','latex');ylabel('${\textit{x}}_{2}$','Interpreter','latex');grid on;box on;
% 
% axes_hdl=subplot(1,2,2);
% [~,ctr_hdl]=contourf(axes_hdl,XMat_draw,YMat_draw,Prblty_pred_MF*2-1,'LineStyle','none');
% shading interp;
% colormap(my_color_map_light);
% hold(axes_hdl,'on');
% X_draw=0:0.01:1;
% SP_HF=scatter(X_HF(C_HF > 0.5,1),X_HF(C_HF > 0.5,2),'or');
% SN_HF=scatter(X_HF(C_HF < 0.5,1),X_HF(C_HF < 0.5,2),'ok');
% line_HF=line(axes_hdl,X_draw,fcn_HF_bou(X_draw),'LineStyle','-');
% SP_LF=scatter(X_LF(C_LF > 0.5,1),X_LF(C_LF > 0.5,2),'*r');
% SN_LF=scatter(X_LF(C_LF < 0.5,1),X_LF(C_LF < 0.5,2),'*k');
% line_LF=line(axes_hdl,X_draw,fcn_LF_bou(X_draw),'LineStyle','--');
% axes_hdl.set('Position',[0.5400,0.1200,0.38,0.6750],'FontSize',12,'FontName','Times New Roman');
% xlabel('${\textit{x}}_{1}$','Interpreter','latex');ylabel('${\textit{x}}_{2}$','Interpreter','latex');grid on;box on;
% 
% legend_handle=legend([SP_HF,SN_HF,line_HF,SP_LF,SN_LF,line_LF],'\fontname{宋体}高精度正样本点','\fontname{宋体}高精度负样本点','\fontname{宋体}高精度分类边界','\fontname{宋体}低精度正样本点','\fontname{宋体}低精度负样本点','\fontname{宋体}低精度分类边界');
% legend_handle.set('Orientation','horizontal','NumColumns',3,'Position',[0.2,0.85,0.60,0.1])
% 
% % print(fig_hdl,'MFGPC_GPC.emf', '-dmeta');

%% function

function C=LFFcn(X)
Bool=0.45+sin(2.2*pi*X(:,1))/2.5-X(:,2) > 0;
C=zeros(size(X,1),1);
C(Bool)=1;
end

function C=HFFcn(X)
Bool=0.5+sin(2.5*pi*X(:,1))/3-X(:,2) > 0;
C=zeros(size(X,1),1);
C(Bool)=1;
end
