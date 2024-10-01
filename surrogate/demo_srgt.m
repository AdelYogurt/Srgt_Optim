clc;
clear;
close all;

%% path

% addpath ../benchmark/
% addpath ../common/

%% single-fidelity surrogate model

% [obj_fcn,vari_num,~,~,~,~,low_bou,up_bou]=suncv_ROS(2);
% X=lhsdesign(20,vari_num);
% Y=obj_fcn(X);
% X_test=lhsdesign(100,vari_num);
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

% srgtsf=srgtsfPRS(X_HF,Y_HF);
% srgtsf=srgtsfRBF(X_HF,Y_HF);
% srgtsf=srgtsfKRG(X_HF,Y_HF);
% srgtsf=srgtsfRBFQdPg(X_HF,Y_HF);

% srgtmf=srgtdfCoRBF(X_LF,Y_LF,X_HF,Y_HF);
% srgtmf=srgtdfMmRBF(X_LF,Y_LF,X_HF,Y_HF);
% srgtmf=srgtdfCoKRG(X_LF,Y_LF,X_HF,Y_HF);
% srgtmf=srgtdfKRGQdPg(X_LF,Y_LF,X_HF,Y_HF);

% srgtmf=srgtmfCoRBF({X_LF,X_HF},{Y_LF,Y_HF});
% srgtmf=srgtmfKRG({X_LF,X_HF},{Y_LF,Y_HF});
% srgtmf=srgtmfHrKRG({X_LF,X_HF},{Y_LF,Y_HF});
% srgtmf=srgtmfCoKRG({X_LF,X_HF},{Y_LF,Y_HF});

% [Y_pred_SF]=srgtsf.predict(X);
% [Y_pred_DF]=srgtmf.predict(X);
% line(X,Y_LF_real,'Color',[0.8500 0.3250 0.0980],'LineStyle','-','LineWidth',2,'Marker','x','MarkerSize',10,'MarkerIndices',1:10:101);
% line(X,Y_HF_real,'Color',[0 0.4470 0.7410],'LineStyle','-','LineWidth',2,'Marker','o','MarkerIndices',[1,41,61,101]);
% line(X,Y_pred_SF,'Color',[0.9290 0.6940 0.1250],'LineStyle','--','LineWidth',2);
% line(X,Y_pred_DF,'Color',[0.3010 0.7450 0.9330],'LineStyle','--','LineWidth',2);
% legend({'low fidelity','high fidelity','predict of single fidelity','predict of multi fidelity'})

%% two dimension double-fidelity surrogate model test

% load('HIM.mat');

% srgtsf=srgtsfPRS(X_HF,Y_HF);
% srgtsf=srgtsfRBF(X_HF,Y_HF);
% srgtsf=srgtsfKRG(X_HF,Y_HF);
% srgtsf=srgtsfRBFQdPg(X_HF,Y_HF);

% srgtmf=srgtdfCoRBF(X_LF,Y_LF,X_HF,Y_HF);
% srgtmf=srgtdfMmRBF(X_LF,Y_LF,X_HF,Y_HF);
% srgtmf=srgtdfCoKRG(X_LF,Y_LF,X_HF,Y_HF);
% srgtmf=srgtdfKRGQdPg(X_LF,Y_LF,X_HF,Y_HF);

% srgtmf=srgtmfCoRBF({X_LF,X_HF},{Y_LF,Y_HF});
% srgtmf=srgtmfKRG({X_LF,X_HF},{Y_LF,Y_HF});
% srgtmf=srgtmfHrKRG({X_LF,X_HF},{Y_LF,Y_HF});
% srgtmf=srgtmfCoKRG({X_LF,X_HF},{Y_LF,Y_HF});

% figure(1);title('MF surrogate');
% displaySrgt([],srgtmf,low_bou,up_bou,[],[],[]);
% figure(2);title('SF surrogate');
% displaySrgt([],srgtsf,low_bou,up_bou,[],[],[]);

%% one dimension multi-fidelity surrogate model test

% Y_HF_fcn=@(x) (6*x-2).^2.*sin(12*x-4);
% Y_MF_fcn=@(x) 0.5*Y_HF_fcn(x)+10*(x-0.5);
% Y_LF_fcn=@(x) 0.4*Y_HF_fcn(x)-x-1;

% X_HF=[0;0.5;1];
% X_MF=linspace(0,1,5)';
% X_LF=linspace(0,1,9)';
% X_real=linspace(0,1,101)';

% Y_HF=Y_HF_fcn(X_HF);
% Y_MF=Y_MF_fcn(X_MF);
% Y_LF=Y_LF_fcn(X_LF);
% Y_real=Y_HF_fcn(X_real);

% srgtmf=srgtmfCoRBF({X_LF,X_MF,X_HF},{Y_LF,Y_MF,Y_HF});
% srgtmf=srgtmfKRG({X_LF,X_MF,X_HF},{Y_LF,Y_MF,Y_HF});
% srgtmf=srgtmfHrKRG({X_LF,X_MF,X_HF},{Y_LF,Y_MF,Y_HF});
% srgtmf=srgtmfCoKRG({X_LF,X_MF,X_HF},{Y_LF,Y_MF,Y_HF});

% Y_pred=srgtmf.predict(X_real);
% fig_hdl=figure(1);
% fig_hdl.set('Position',[200,200,360,300]);
% axe_hdl=axes(fig_hdl);
% line_real=line(axe_hdl,X_real,Y_real,'Color',[0 0.4470 0.7410],'LineStyle','-','LineWidth',1);
% line_CK=line(axe_hdl,X_real,Y_pred,'Color',[0.9290 0.6940 0.1250],'LineStyle','--','LineWidth',2);
% hold on;grid on;box on;
% pnt_HF=scatter(axe_hdl,X_HF,Y_HF,'Marker','o');
% pnt_MF=scatter(axe_hdl,X_MF,Y_MF,'Marker','s');
% pnt_LF=scatter(axe_hdl,X_LF,Y_LF,'Marker','d');
% legend([line_real,line_CK,pnt_HF,pnt_MF,pnt_LF],{'high fidelity','predict of Multi-Level surrogate','HF','MF','LF'},...
%     'Location','northwest','box','on')
% axe_hdl.GridAlpha=0.2;
% axe_hdl.FontName='times new roman';
% xlabel('x');ylabel('y');

%% K-fold verification

% [obj_fcn,vari_num,~,~,~,~,low_bou,up_bou]=suncv_ROS(2);
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
