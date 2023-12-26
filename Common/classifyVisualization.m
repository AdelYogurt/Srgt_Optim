function classifyVisualization...
    (classify_model,low_bou,up_bou,grid_number,figure_handle)
% visualization SVM_model
% red is 1, blue is -1
%
if nargin < 5
    figure_handle=[];
    if nargin < 4
        grid_number=100;
    end
end
if nargin < 1
    error('classifyVisualization: not enough input');
end
X=classify_model.X;
if iscell(X) X=X{1}; end
Class=classify_model.Class;
if iscell(Class) Class=Class{1}; end
predict_function=classify_model.predict_function;

if nargin < 3
    up_bou=max(X);
else
    up_bou=up_bou(:)';
end
if nargin < 2
    low_bou=min(X);
else
    low_bou=low_bou(:)';
end

if isempty(figure_handle)
    figure_handle=figure(10);
end
axes_handle=figure_handle.CurrentAxes;
if isempty(axes_handle)
    axes_handle=axes(figure_handle);
end

dimension=length(low_bou);

% check dimension
[x_number,variable_number]=size(X);
if variable_number > 2
    error('classifyVisualization: dimension large than 2');
end

% identify point properties
positive_index=[];
negative_index=[];
for x_index=1:x_number
    x=X(x_index,:);
    if ~(sum(x > up_bou) || sum(x < low_bou))
        if Class(x_index) > 0
            positive_index=[positive_index;x_index];
        else
            negative_index=[negative_index;x_index];
        end
    end
end

switch dimension
    case 1
        d_bou=(up_bou-low_bou)/grid_number;
        X__=low_bou:d_bou:up_bou;
        predict_class=zeros(grid_number+1,1);
        predict_fval=zeros(grid_number+1,1);
        for x_index__=1:(grid_number+1)
            predict_x=X__(x_index__);
            [predict_class(x_index),predict_fval(x_index)]=...
                predict_function(predict_x);
        end
        line(axes_handle,X__,predict_class,'LineStyle','none','Marker','d','Color','k');
        line(axes_handle,X__,predict_fval);
        xlabel('X');
        ylabel('Possibility of 1');
        if ~isempty(positive_index)
            line(axes_handle,X(positive_index),Class(positive_index),'LineStyle','none','Marker','o','Color','r');
        end
        if ~isempty(negative_index)
            line(axes_handle,X(negative_index),Class(negative_index),'LineStyle','none','Marker','o','Color','b');
        end
    case 2
        % draw zero value line
        grid_number=100;
        d_bou=(up_bou-low_bou)/grid_number;
        [X_draw,Y_draw]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
        try
            % generate all predict list
            predict_x=[X_draw(:),Y_draw(:)];
            [predict_class,predict_probability]=predict_function(predict_x);
            predict_class=reshape(predict_class,grid_number+1,grid_number+1);
            predict_probability=reshape(predict_probability,grid_number+1,grid_number+1);
        catch
            predict_class=zeros(grid_number+1);
            predict_probability=zeros(grid_number+1);
            for x_index=1:grid_number+1
                for y_index=1:grid_number+1
                    predict_x=([x_index,y_index]-1).*d_bou+low_bou;
                    [predict_class(y_index,x_index),predict_probability(y_index,x_index)]=...
                        predict_function(predict_x);
                end
            end
        end
        contour(axes_handle,X_draw,Y_draw,predict_class);
        hold on;
        contour(axes_handle,X_draw,Y_draw,predict_probability*2-1);
        hold off;
        xlabel('X');
        ylabel('Y');

        % draw point
        if ~isempty(positive_index)
            line(axes_handle,X(positive_index,1),X(positive_index,2),...
                'LineStyle','none','Marker','o','Color','r');
        end
        if ~isempty(negative_index)
            line(axes_handle,X(negative_index,1),X(negative_index,2),...
                'LineStyle','none','Marker','o','Color','b');
        end
end
end
