clc;
clear;
close all;

%% path

% addpath ../benchmark/
% addpath ../common/

%% single-fidelity surrogate model

% [obj_fcn,vari_num,~,~,~,~,low_bou,up_bou]=suncvdROS(2);
% X=lhsdesign(20,2);
% Y=obj_fcn(X);
% X_test=lhsdesign(100,2);
% Y_test=obj_fcn(X_test);

% srgt=srgtsfPRS(X,Y);
% srgt=srgtsfRBF(X,Y);
% srgt=srgtsfKRG(X,Y);
% srgt=srgtsfRBFQdPg(X,Y);

% RMSE=cvSrgtError(srgt,X_test,Y_test,'RMSE');
% disp('predict RMSE');disp(RMSE);
% R2=cvSrgtError(srgt,X_test,Y_test,'R2');
% disp('predict R^2');disp(R2);

% figure(1);title('surrogate');
% displaySrgt([],srgt,low_bou,up_bou);

%% one dimension double-fidelity surrogate model test

% load('forrester.mat');

% srgtsf=srgtsfPRS(XHF,YHF);
% srgtsf=srgtsfRBF(XHF,YHF);
% srgtsf=srgtsfKRG(XHF,YHF);
% srgtsf=srgtsfRBFQdPg(XHF,YHF);

% srgtmf=srgtdfCoRBF(XLF,YLF,XHF,YHF);
% srgtmf=srgtdfMmRBF(XLF,YLF,XHF,YHF);
% srgtmf=srgtdfCoKRG(XLF,YLF,XHF,YHF);
% srgtmf=srgtdfKRGQdPg(XLF,YLF,XHF,YHF);

% srgtmf=srgtmfCoRBF({XLF,XHF},{YLF,YHF});
% srgtmf=srgtmfKRG({XLF,XHF},{YLF,YHF});
% srgtmf=srgtmfHrKRG({XLF,XHF},{YLF,YHF});
% srgtmf=srgtmfCoKRG({XLF,XHF},{YLF,YHF});

% [Y_pred_SF]=srgtsf.predict(X);
% [Y_pred_DF]=srgtmf.predict(X);
% line(X,YLF_real,'Color',[0.8500 0.3250 0.0980],'LineStyle','-','LineWidth',2,'Marker','x','MarkerSize',10,'MarkerIndices',1:10:101);
% line(X,YHF_real,'Color',[0 0.4470 0.7410],'LineStyle','-','LineWidth',2,'Marker','o','MarkerIndices',[1,41,61,101]);
% line(X,Y_pred_SF,'Color',[0.9290 0.6940 0.1250],'LineStyle','--','LineWidth',2);
% line(X,Y_pred_DF,'Color',[0.3010 0.7450 0.9330],'LineStyle','--','LineWidth',2);
% legend({'low fidelity','high fidelity','predict of single fidelity','predict of multi fidelity'})

%% two dimension double-fidelity surrogate model test

% load('HIM.mat');

% srgtsf=srgtsfPRS(XHF,YHF);
% srgtsf=srgtsfRBF(XHF,YHF);
% srgtsf=srgtsfKRG(XHF,YHF);
% srgtsf=srgtsfRBFQdPg(XHF,YHF);

% srgtmf=srgtdfCoRBF(XLF,YLF,XHF,YHF);
% srgtmf=srgtdfMmRBF(XLF,YLF,XHF,YHF);
% srgtmf=srgtdfCoKRG(XLF,YLF,XHF,YHF);
% srgtmf=srgtdfKRGQdPg(XLF,YLF,XHF,YHF);

% srgtmf=srgtmfCoRBF({XLF,XHF},{YLF,YHF});
% srgtmf=srgtmfKRG({XLF,XHF},{YLF,YHF});
% srgtmf=srgtmfHrKRG({XLF,XHF},{YLF,YHF});
% srgtmf=srgtmfCoKRG({XLF,XHF},{YLF,YHF});

% figure(1);title('MF surrogate');
% displaySrgt([],srgtmf,low_bou,up_bou,[],[],[]);
% figure(2);title('SF surrogate');
% displaySrgt([],srgtsf,low_bou,up_bou,[],[],[]);

%% one dimension multi-fidelity surrogate model test

% YHF_fcn=@(x) (6*x-2).^2.*sin(12*x-4);
% YMF_fcn=@(x) 0.5*YHF_fcn(x)+10*(x-0.5);
% YLF_fcn=@(x) 0.4*YHF_fcn(x)-x-1;

% XHF=[0;0.5;1];
% XMF=linspace(0,1,5)';
% XLF=linspace(0,1,9)';
% X_real=linspace(0,1,101)';

% YHF=YHF_fcn(XHF);
% YMF=YMF_fcn(XMF);
% YLF=YLF_fcn(XLF);
% Y_real=YHF_fcn(X_real);

% srgtmf=srgtmfCoRBF({XLF,XMF,XHF},{YLF,YMF,YHF});
% srgtmf=srgtmfKRG({XLF,XMF,XHF},{YLF,YMF,YHF});
% srgtmf=srgtmfHrKRG({XLF,XMF,XHF},{YLF,YMF,YHF});
% srgtmf=srgtmfCoKRG({XLF,XMF,XHF},{YLF,YMF,YHF});

% Y_pred=srgtmf.predict(X_real);
% fig_hdl=figure(1);
% fig_hdl.set('Position',[200,200,360,300]);
% axe_hdl=axes(fig_hdl);
% line_real=line(axe_hdl,X_real,Y_real,'Color',[0 0.4470 0.7410],'LineStyle','-','LineWidth',1);
% line_CK=line(axe_hdl,X_real,Y_pred,'Color',[0.9290 0.6940 0.1250],'LineStyle','--','LineWidth',2);
% hold on;grid on;box on;
% pnt_hf=scatter(axe_hdl,XHF,YHF,'Marker','o');
% pnt_MF=scatter(axe_hdl,XMF,YMF,'Marker','s');
% pnt_lf=scatter(axe_hdl,XLF,YLF,'Marker','d');
% legend([line_real,line_CK,pnt_hf,pnt_MF,pnt_lf],{'high fidelity','predict of Multi-Level surrogate','HF','MF','LF'},...
%     'Location','northwest','box','on')
% axe_hdl.GridAlpha=0.2;
% xlabel('x');ylabel('y');

%% K-fold verification

% [obj_fcn,vari_num,~,~,~,~,low_bou,up_bou]=suncvdROS(2);
% 
% X=lhsdesign(20,2);
% Y=obj_fcn(X);
% K=10;
% 
% srgt_fit_fcn=@(X,Y)srgtsfPRS(X,Y);
% srgt_fit_fcn=@(X,Y)srgtsfRBF(X,Y);
% srgt_fit_fcn=@(X,Y)srgtsfKRG(X,Y);
% 
% RMSE=cvSrgtKFold(srgt_fit_fcn,X,Y,K,'RMSE');
% disp('K-fold RMSE');disp(RMSE);
% R2=cvSrgtKFold(srgt_fit_fcn,X,Y,K,'R2');
% disp('K-fold R^2');disp(R2);
