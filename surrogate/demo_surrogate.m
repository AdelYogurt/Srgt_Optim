clc;
clear;
close all hidden;

%% single-fidelity surrogate model test

% load('PK.mat');

% model_srgt=srgtKRG(X,Y);
% model_srgt=srgtGPR(X,Y);
% model_srgt=srgtRBF(X,Y);
% model_srgt=srgtRBFOpt(X,Y);
% model_srgt=srgtRBFQdPg(X,Y);
% model_srgt=srgtRSM(X,Y);
% surrogateVisualize(model_srgt,low_bou,up_bou);

%% multi-fidelity surrogate model test

% load('HIM.mat')
% model_MF=srgtCoKRG({X_LF,X_HF},{Y_LF,Y_HF});
% model_MF=srgtHrKRG({X_LF,X_HF},{Y_LF,Y_HF});
% model_MF=srgtMtKRG({X_LF,X_HF},{Y_LF,Y_HF});
% model_MF=srgtMtKRGQdPg(X_HF,Y_HF,{X_LF},{Y_LF});
% model_SF=srgtKRG(X_HF,Y_HF);

% model_MF=srgtMtRBF(X_HF,Y_HF,[],X_LF,Y_LF,[]);
% model_SF=srgtRBF(X_HF,Y_HF);
% surrogateVisualize(model_MF,low_bou,up_bou,[],[],[],figure(1));
% surrogateVisualize(model_SF,low_bou,up_bou,[],[],[],figure(2));

%% multi-fidelity surrogate model test

% load('Forrester.mat')

% model_MF=srgtCoKRG(X_LF,Y_LF,X_HF,Y_HF);
% model_MF=srgtExCoKRG({X_LF,X_HF},{Y_LF,Y_HF});
% model_MF=srgtHrKRG({X_LF,X_HF},{Y_LF,Y_HF});
% model_MF=srgtMtKRG({X_LF,X_HF},{Y_LF,Y_HF});
% model_MF=srgtMtKRGQdPg(X_HF,Y_HF,{X_LF},{Y_LF});
% model_SF=srgtKRG(X_LF,Y_LF);

% model_MF=srgtMmRBF(X_LF,Y_LF,X_HF,Y_HF);
% model_MF=srgtCoRBF(X_LF,Y_LF,X_HF,Y_HF);
% model_SF=srgtRBF(X_HF,Y_HF);

% [Y_pred_SF]=model_SF.predict(X);
% [Y_pred_MF]=model_MF.predict(X);
% line(X,Y_real_LF,'Color',[0.8500 0.3250 0.0980],'LineStyle','-','LineWidth',2,'Marker','x','MarkerSize',10,'MarkerIndices',1:10:101);
% line(X,Y_real,'Color',[0 0.4470 0.7410],'LineStyle','-','LineWidth',2,'Marker','o','MarkerIndices',[1,41,61,101]);
% line(X,Y_pred_SF,'Color',[0.9290 0.6940 0.1250],'LineStyle','--','LineWidth',2);
% line(X,Y_pred_MF,'Color',[0.3010 0.7450 0.9330],'LineStyle','--','LineWidth',2);
% legend({'low fidelity','high fidelity','predict of single fidelity','predict of multi fidelity'})

%% draw multi-fidelity surrogate model fit result

% load('Forrester.mat')
% 
% fig_hdl=figure(1);
% fig_hdl.set('Position',[488,200,700,420])
% 
% model_MF=srgtMtRBF(X_HF,Y_HF,[],X_LF,Y_LF,[]);
% model_SF=srgtRBF(X_HF,Y_HF,[]);
% 
% Y_pred_MF=model_MF.predict(X);
% Y_pred_SF=model_SF.predict(X);
% 
% [x_MF_best,y_MF_best]=fmincon(@(X) model_MF.predict(X),0.7,[],[],[],[],0,1,[],optimoptions('fmincon','Display','none'));
% [x_SF_best,y_SF_best]=fmincon(@(X) model_SF.predict(X),0.2,[],[],[],[],0,1,[],optimoptions('fmincon','Display','none'));
% 
% axes_hdl=subplot(1,2,1);
% line_LF=line(X,Y_real_LF,'Color',[0.8500 0.3250 0.0980],'LineStyle','-','LineWidth',2,'Marker','x','MarkerSize',10,'MarkerIndices',1:10:101);
% line_HF=line(X,Y_real,'Color',[0 0.4470 0.7410],'LineStyle','-','LineWidth',2,'Marker','o','MarkerIndices',[1,41,61,101]);
% line_SF=line(X,Y_pred_SF,'Color',[0.9290 0.6940 0.1250],'LineStyle','--','LineWidth',2);
% point_best=line(0.7572,-6.0167,'Marker','p','MarkerSize',10,'Color','b','LineStyle','none','LineWidth',2);
% point_SF=line(x_SF_best,y_SF_best,'Marker','p','MarkerSize',10,'Color','g','LineStyle','none','LineWidth',2);
% axes_hdl.set('Position',[0.1000,0.1500,0.38,0.6750],'FontSize',12,'FontName','Times New Roman');
% xlabel('$\textit{x}$','Interpreter','latex');ylabel('$\textit{y}$','Interpreter','latex');grid on;box on;
% 
% axes_hdl=subplot(1,2,2);
% line(X,Y_real_LF,'Color',[0.8500 0.3250 0.0980],'LineStyle','-','LineWidth',2,'Marker','x','MarkerSize',10,'MarkerIndices',1:10:101);
% line(X,Y_real,'Color',[0 0.4470 0.7410],'LineStyle','-','LineWidth',2,'Marker','o','MarkerIndices',[1,41,61,101]);
% line_MF=line(X,Y_pred_MF,'Color',[0.3010 0.7450 0.9330],'LineStyle','--','LineWidth',2);
% point_best=line(0.7572,-6.0167,'Marker','p','MarkerSize',10,'Color','b','LineStyle','none','LineWidth',2);
% point_MF=line(x_MF_best,y_MF_best,'Marker','p','MarkerSize',10,'Color','m','LineStyle','none','LineWidth',2);
% axes_hdl.set('Position',[0.5800,0.1500,0.38,0.6750],'FontSize',12,'FontName','Times New Roman');
% xlabel('$\textit{x}$','Interpreter','latex');ylabel('$\textit{y}$','Interpreter','latex');grid on;box on;
% 
% lgd_hdl=legend([line_SF,point_SF,line_MF,point_MF,...
%     line_HF,line_LF,point_best],...
%     {'RBF','RBF\fontname{宋体}全局最优值','MFRBF','MFRBF\fontname{宋体}全局最优值',...
%     '\fontname{宋体}高精度模型及采样点','\fontname{宋体}低精度模型及采样点','\fontname{宋体}高精度模型全局最优值'});
% lgd_hdl.set('Position',[0.01,0.87,0.98,0.1],'Orientation','horizontal','NumColumns',4,'FontSize',10)

% print(fig_hdl,'MFRBF_RBF.emf','-dmeta');
% print(fig_hdl,'MFRBF_RBF.png','-dpng');
