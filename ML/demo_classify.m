clc;
clear;
close all hidden;

%% gen data

% fcn_lf=@(x) LFFcn(x);
% fcn_hf=@(x) HFFcn(x);
% 
% fcn_lf_bou=@(x) 0.45+sin(2.2*pi*x)/2.5;
% fcn_hf_bou=@(x) 0.5+sin(2.5*pi*x)/3;
% 
% vari_num=2;
% 
% xlf_num=20;
% XLF=lhsdesign(xlf_num,vari_num);
% CLF=fcn_lf(XLF);
% 
% xhf_num=10;
% XHF=lhsdesign(xhf_num,vari_num);
% CHF=fcn_hf(XHF);
% 
% low_bou=[0,0];
% up_bou=[1,1];

%% draw single-fidelity classification

% load('CMF_30.mat');
% 
% classify=classifySVM(XHF,CHF);
% classify=classifyGPC(XHF,CHF);
% 
% fig_hdl=figure(1);
% displayClassify([],classify,low_bou,up_bou)
% drawFcn([],fcn_hf_bou,0,1);

%% draw multi-fidelity classification

% load('CMF_30.mat');
% 
% classify_mf=classifyCoGPC(XHF,CHF,XLF,CLF);
% fig_hdl=figure(1);
% displayClassify([],classify_mf,low_bou,up_bou)
% drawFcn([],fcn_hf_bou,0,1);
% drawFcn([],fcn_lf_bou,0,1);

%% drwa picture of comparison between GPC and CoGPC

% load('CMF_30.mat');
% load('color.mat','my_color_map_light');
% 
% model_SF=classifyGPC(XHF,CHF);
% model_MF=classifyCoGPC(XHF,CHF,XLF,CLF);
% 
% % draw zero value line
% grid_num=100;
% d_bou=(up_bou-low_bou)/grid_num;
% [XMat_draw,YMat_draw]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
% 
% % generate all pred list
% X_pred=[XMat_draw(:),YMat_draw(:)];
% [~,Prob_pred_SF]=model_SF.predict(X_pred);
% Prob_pred_SF=reshape(Prob_pred_SF,grid_num+1,grid_num+1);
% [~,Prob_pred_MF]=model_MF.predict(X_pred);
% Prob_pred_MF=reshape(Prob_pred_MF,grid_num+1,grid_num+1);
% 
% % draw picture
% fig_hdl=figure(1);
% fig_hdl.set('Position',[488   200   680  420])
% 
% axes_hdl=subplot(1,2,1);
% [~,contour_handle]=contourf(axes_hdl,XMat_draw,YMat_draw,Prob_pred_SF*2-1,'LineStyle','none');
% shading interp;
% colormap(my_color_map_light);
% hold(axes_hdl,'on');
% X_draw=0:0.01:1;
% scatter(XHF((CHF>0),1),XHF((CHF>0),2),'or')
% scatter(XHF((CHF<0),1),XHF((CHF<0),2),'ok')
% line(axes_hdl,X_draw,fcn_hf_bou(X_draw),'LineStyle','-');
% axes_hdl.set('Position',[0.0800,0.1200,0.38,0.6750],'FontSize',12,'FontName','Times New Roman');
% xlabel('${\textit{x}}_{1}$','Interpreter','latex');ylabel('${\textit{x}}_{2}$','Interpreter','latex');grid on;box on;
% 
% axes_hdl=subplot(1,2,2);
% [~,contour_handle]=contourf(axes_hdl,XMat_draw,YMat_draw,Prob_pred_MF*2-1,'LineStyle','none');
% shading interp;
% colormap(my_color_map_light);
% hold(axes_hdl,'on');
% X_draw=0:0.01:1;
% SP_hf=scatter(XHF((CHF>0),1),XHF((CHF>0),2),'or');
% SN_hf=scatter(XHF((CHF<0),1),XHF((CHF<0),2),'ok');
% line_hf=line(axes_hdl,X_draw,fcn_hf_bou(X_draw),'LineStyle','-');
% SP_lf=scatter(XLF((CLF>0),1),XLF((CLF>0),2),'*r');
% SN_lf=scatter(XLF((CLF<0),1),XLF((CLF<0),2),'*k');
% line_lf=line(axes_hdl,X_draw,fcn_lf_bou(X_draw),'LineStyle','--');
% axes_hdl.set('Position',[0.5400,0.1200,0.38,0.6750],'FontSize',12,'FontName','Times New Roman');
% xlabel('${\textit{x}}_{1}$','Interpreter','latex');ylabel('${\textit{x}}_{2}$','Interpreter','latex');grid on;box on;
% 
% legend_handle=legend([SP_hf,SN_hf,line_hf,SP_lf,SN_lf,line_lf],'\fontname{宋体}高精度正样本点','\fontname{宋体}高精度负样本点','\fontname{宋体}高精度分类边界','\fontname{宋体}低精度正样本点','\fontname{宋体}低精度负样本点','\fontname{宋体}低精度分类边界');
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
