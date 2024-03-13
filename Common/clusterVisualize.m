function clusterVisualize(clustering_model,figure_handle)
% visualization SVM_model
% red is 1, blue is -1
%
if nargin < 2
    figure_handle=[];
end
if nargin < 1
    error('classifyVisualization: not enough input');
end
X=clustering_model.X;
if iscell(X) X=X{1}; end
center_list=clustering_model.center_list;
if iscell(center_list) center_list=center_list{1}; end

if isempty(figure_handle)
    figure_handle=figure(10);
end
axes_handle=figure_handle.CurrentAxes;
if isempty(axes_handle)
    axes_handle=axes(figure_handle);
end

% check dimension
[x_number,variable_number]=size(X);

switch variable_number
    case 1
        hold on;
        scatter(X(:,1),zeros(x_number,1));
        line(axes_handle,center_list(:,1),zeros(x_number,1),'Marker','o','LineStyle','None','Color','r');
        hold off;
        xlabel('X');
        ylabel('Y');
    case 2
        hold on;
        scatter(X(:,1),X(:,2));
        line(axes_handle,center_list(:,1),center_list(:,2),'Marker','o','LineStyle','None','Color','r');
        hold off;
        xlabel('X');
        ylabel('Y');
    case 3
        hold on;
        scatter3(X(:,1),X(:,2),X(:,3));
        line(axes_handle,center_list(:,1),center_list(:,2),center_list(:,3),'Marker','o','LineStyle','None','Color','r');
        hold off;
        xlabel('X');
        ylabel('Y');
        zlabel('Z');
        view(3);
    otherwise
        hold on;
        scatter3(X(:,1),X(:,2),X(:,3));
        line(axes_handle,center_list(:,1),center_list(:,2),center_list(:,3),'Marker','o','LineStyle','None','Color','r');
        hold off;
        xlabel('X');
        ylabel('Y');
        zlabel('Z');
        view(3);
end
end
