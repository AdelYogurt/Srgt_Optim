function displayCluster(axe_hdl,cluster)
% visualization SVM_model
% red is 1, blue is -1
%
if isempty(axe_hdl),axe_hdl=gca();end

X=cluster.X;
if iscell(X) X=X{1}; end
center_list=cluster.center_list;
if iscell(center_list) center_list=center_list{1}; end

% check dimension
[x_num,vari_num]=size(X);

switch vari_num
    case 1
        hold on;
        scatter(X(:,1),zeros(x_num,1));
        line(axe_hdl,center_list(:,1),zeros(x_num,1),'Marker','o','LineStyle','None','Color','r');
        hold off;
        xlabel('X');
        ylabel('Y');
    case 2
        hold on;
        scatter(X(:,1),X(:,2));
        line(axe_hdl,center_list(:,1),center_list(:,2),'Marker','o','LineStyle','None','Color','r');
        hold off;
        xlabel('X');
        ylabel('Y');
    case 3
        hold on;
        scatter3(X(:,1),X(:,2),X(:,3));
        line(axe_hdl,center_list(:,1),center_list(:,2),center_list(:,3),'Marker','o','LineStyle','None','Color','r');
        hold off;
        xlabel('X');
        ylabel('Y');
        zlabel('Z');
        view(3);
    otherwise
        hold on;
        scatter3(X(:,1),X(:,2),X(:,3));
        line(axe_hdl,center_list(:,1),center_list(:,2),center_list(:,3),'Marker','o','LineStyle','None','Color','r');
        hold off;
        xlabel('X');
        ylabel('Y');
        zlabel('Z');
        view(3);
end
end
