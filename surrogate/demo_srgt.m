clc;
clear;
close all;

%% single-fidelity surrogate model test

% load('PK.mat');

% mdl_srgt=srgtKRG(X,Y);
% mdl_srgt=srgtGPR(X,Y);
% mdl_srgt=srgtRBF(X,Y);
% mdl_srgt=srgtRBFOpt(X,Y);
% mdl_srgt=srgtRBFQdPg(X,Y);
% mdl_srgt=srgtRSM(X,Y);

% disp('predict error');
% Y_err=mdl_srgt.predict(X_test)-Y_test;
% disp(Y_err);

% R2=sqrt(sum((mdl_srgt.predict(X_test)-Y_test).^2)/length(Y_test));
% disp(R2);

% figure(1);title('surrogate');
% srgtVisualize([],mdl_srgt,low_bou,up_bou);

%% one dimension double-fidelity surrogate model test

% load('FT.mat');
% 
% mdl_SF=srgtRBF(X_HF,Y_HF);
% mdl_SF=srgtKRG(X_HF,Y_HF);
% 
% mdl_DF=srgtMmRBF(X_LF,Y_LF,X_HF,Y_HF);
% mdl_DF=srgtCoRBF(X_LF,Y_LF,X_HF,Y_HF);
% mdl_DF=srgtCoKRG(X_LF,Y_LF,X_HF,Y_HF);
% mdl_DF=srgtMtKRGQdPg(X_HF,Y_HF,{X_LF},{Y_LF});
% 
% mdl_DF=srgtMFCoRBF({X_LF,X_HF},{Y_LF,Y_HF});
% mdl_DF=srgtMFKRG({X_LF,X_HF},{Y_LF,Y_HF});
% mdl_DF=srgtMFCoKRG({X_LF,X_HF},{Y_LF,Y_HF});
% mdl_DF=srgtMFHrKRG({X_LF,X_HF},{Y_LF,Y_HF});
% 
% [Y_pred_SF]=mdl_SF.predict(X);
% [Y_pred_MF]=mdl_DF.predict(X);
% line(X,Y_real_LF,'Color',[0.8500 0.3250 0.0980],'LineStyle','-','LineWidth',2,'Marker','x','MarkerSize',10,'MarkerIndices',1:10:101);
% line(X,Y_real,'Color',[0 0.4470 0.7410],'LineStyle','-','LineWidth',2,'Marker','o','MarkerIndices',[1,41,61,101]);
% line(X,Y_pred_SF,'Color',[0.9290 0.6940 0.1250],'LineStyle','--','LineWidth',2);
% line(X,Y_pred_MF,'Color',[0.3010 0.7450 0.9330],'LineStyle','--','LineWidth',2);
% legend({'low fidelity','high fidelity','predict of single fidelity','predict of multi fidelity'})

%% two dimension double-fidelity surrogate model test

% load('HIM.mat');

% mdl_MF=srgtCoKRG(X_LF,Y_LF,X_HF,Y_HF);
% mdl_MF=srgtMFHrKRG({X_LF,X_HF},{Y_LF,Y_HF});
% mdl_MF=srgtMFKRG({X_LF,X_HF},{Y_LF,Y_HF});
% mdl_MF=srgtMtKRGQdPg(X_HF,Y_HF,{X_LF},{Y_LF});
% mdl_SF=srgtKRG(X_HF,Y_HF);

% mdl_MF=srgtCoRBF(X_LF,Y_LF,X_HF,Y_HF);
% mdl_MF=srgtMmRBF(X_LF,Y_LF,X_HF,Y_HF);
% mdl_SF=srgtRBF(X_HF,Y_HF);

% disp('MF predict error');
% Y_err=mdl_MF.predict(X_test)-Y_test;
% disp(Y_err);

% disp('SF predict error');
% Y_err=mdl_SF.predict(X_test)-Y_test;
% disp(Y_err);

% figure(1);title('MF surrogate');
% srgtVisualize([],mdl_MF,low_bou,up_bou,[],[],[]);
% figure(2);title('SF surrogate');
% srgtVisualize([],mdl_SF,low_bou,up_bou,[],[],[]);

%% one dimension multi-fidelity surrogate model test

% Y_HF_fcn=@(x) (6*x-2).^2.*sin(12*x-4);
% Y_MF_fcn=@(x) 0.5*Y_HF_fcn(x)+10*(x-0.5);
% Y_LF_fcn=@(x) 0.4*Y_HF_fcn(x)-x-1;
% 
% X_HF=[0;0.5;1];
% X_MF=linspace(0,1,5)';
% X_LF=linspace(0,1,9)';
% 
% Y_HF=Y_HF_fcn(X_HF);
% Y_MF=Y_MF_fcn(X_MF);
% Y_LF=Y_LF_fcn(X_LF);
% 
% mdl_CK=srgtMFCoKRG({X_LF,X_MF,X_HF},{Y_LF,Y_MF,Y_HF});
% mdl_HK=srgtMFHrKRG({X_LF,X_MF,X_HF},{Y_LF,Y_MF,Y_HF});
% 
% X_real=linspace(0,1,101)';
% Y_real=Y_HF_fcn(X_real);
% Y_pred_CK=mdl_CK.predict(X_real);
% Y_pred_HK=mdl_HK.predict(X_real);
% 
% fig_hdl=figure(1);
% fig_hdl.set('Position',[200,200,360,300]);
% 
% axe_hdl=axes(fig_hdl);
% 
% line_real=line(axe_hdl,X_real,Y_real,'Color',[0 0.4470 0.7410],'LineStyle','-','LineWidth',1);
% line_CK=line(axe_hdl,X_real,Y_pred_CK,'Color',[0.9290 0.6940 0.1250],'LineStyle','--','LineWidth',2);
% line_HK=line(axe_hdl,X_real,Y_pred_HK,'Color',[0.3010 0.7450 0.9330],'LineStyle','--','LineWidth',2);
% hold on;grid on;box on;
% 
% pnt_HF=scatter(axe_hdl,X_HF,Y_HF,'Marker','o');
% pnt_MF=scatter(axe_hdl,X_MF,Y_MF,'Marker','s');
% pnt_LF=scatter(axe_hdl,X_LF,Y_LF,'Marker','d');
% legend([line_real,line_CK,line_HK,pnt_HF,pnt_MF,pnt_LF],{'high fidelity','predict of CK','predict of HK','HF','MF','LF'},...
%     'Location','northwest','box','on')
% axe_hdl.GridAlpha=0.2;
% xlabel('x');ylabel('y');
