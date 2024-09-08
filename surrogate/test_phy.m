clc;
clear;
close all;

fcn=@(x) 1+2*sqrt(x);
X=[0;1;2;4];Y=fcn(X);

mdl_KRG=srgtKRG(X,Y,struct('optimize_hyp',true,'hyp',-3));

X_dis_sq=zeros(size(X,1),size(X,1),size(X,2));
for vari_idx=1:size(X,2)
    X_dis_sq(:,:,vari_idx)=(X(:,vari_idx)-X(:,vari_idx)').^2;
end
cov_fcn=@(X,X_pred,hyp) covPhy(X,X_pred,hyp,X_dis_sq);
mdl_PHY=srgtKRGNew(X,Y,struct('cov_fcn',cov_fcn,'optimize_hyp',true,'hyp',-3));

X_real=(0:0.01:15)';
Y_real=fcn(X_real);
[Y_pred,Var_pred]=mdl_KRG.predict(X_real);
[Y_pred_phy,Var_pred_phy]=mdl_PHY.predict(X_real);

fig_hdl=figure(1);
fig_hdl.set('Position',[200,200,540,320]);

axes_hdl=subplot(1,2,1);
X_fill=[X_real; flipud(X_real)];
Y_fill=[Y_pred+Var_pred*3; flipud(Y_pred-Var_pred*3)];

fill_N=fill(X_fill,Y_fill,[0.9290 0.6940 0.1250],'edgealpha', 0, 'facealpha', 0.4);
line_real=line(X_real,Y_real,'Color','k','LineStyle','-','LineWidth',2);
line_N=line(X_real,Y_pred,'Color',[0.3010 0.7450 0.9330],'LineStyle','--','LineWidth',2);
point_obv=line(X,Y,'Marker','p','MarkerSize',10,'Color','g','LineStyle','none','LineWidth',2);
axes_hdl.set('Position',[0.1000,0.1500,0.38,0.6750],'FontSize',12,'FontName','Times New Roman');
xlabel('$\textit{x}$','Interpreter','latex');ylabel('$\textit{y}$','Interpreter','latex');box on;
ylim([-2,12]);

axes_hdl=subplot(1,2,2);
X_fill=[X_real; flipud(X_real)];
Y_fill=[Y_pred_phy+Var_pred_phy*3; flipud(Y_pred_phy-Var_pred_phy*3)];

fill_N=fill(X_fill,Y_fill,[0.9290 0.6940 0.1250],'edgealpha', 0, 'facealpha', 0.4);
line_real=line(X_real,Y_real,'Color','k','LineStyle','-','LineWidth',2);
line_phy=line(X_real,Y_pred_phy,'Color',[0.3010 0.7450 0.9330],'LineStyle','--','LineWidth',2);
point_obv=line(X,Y,'Marker','p','MarkerSize',10,'Color','g','LineStyle','none','LineWidth',2);
axes_hdl.set('Position',[0.5800,0.1500,0.38,0.6750],'FontSize',12,'FontName','Times New Roman');
xlabel('$\textit{x}$','Interpreter','latex');ylabel('$\textit{y}$','Interpreter','latex');box on;
ylim([-2,12]);

lgd_hdl=legend([line_real,point_obv,line_phy,fill_N],...
    {'\fontname{宋体}真实函数','\fontname{宋体}采样点','\fontname{宋体}KRG预测均值','\fontname{宋体}95%置信区间'});
lgd_hdl.set('Position',[0.01,0.87,0.98,0.1],'Orientation','horizontal','NumColumns',4,'FontSize',10,'box','off')

% print(fig_hdl,'phy_compare.emf','-dmeta');
% print(fig_hdl,'phy_compare.png','-dpng');


function [cov,dcov_dhyp]=covPhy(X,X_pred,hyp,X_dis_sq)
if nargin < 4,X_dis_sq=[];end
[x_n,vari_n]=size(X);
theta=exp(hyp);
if isempty(X_pred)
    if isempty(X_dis_sq)
        % initial X_dis_sq
        X_dis_sq=zeros(x_n,x_n,vari_n);
        for vari_idx=1:vari_n
            X_dis_sq(:,:,vari_idx)=(sqrt(X(:,vari_idx))-sqrt(X(:,vari_idx)')).^2;
        end
    end

    % calculate covariance
    cov=zeros(x_n,x_n);
    for vari_i=1:vari_n
        cov=cov+X_dis_sq(:,:,vari_i)*theta(vari_i);
    end
    cov=exp(-cov/vari_n^2)+eye(x_n)*((1000+x_n)*eps);
else
    x_pred_num=size(X_pred,1);
    X_dis_sq=[];
    % predict covariance
    cov=zeros(x_n,x_pred_num);
    for vari_i=1:vari_n
        cov=cov+(sqrt(X(:,vari_i))-sqrt(X_pred(:,vari_i))').^2*theta(vari_i);
    end
    cov=exp(-cov/vari_n^2);
end

if nargout > 2
    dcov_dhyp=zeros(x_n,x_n,vari_n);
    for vari_i=1:vari_n
        dcov_dhyp=-(X_dis_sq(:,:,vari_i).*R)*theta(vari_i)/vari_n^2;
    end
end
end