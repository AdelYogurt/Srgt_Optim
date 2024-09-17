function displayClassify(axe_hdl,classify,low_bou,up_bou,grid_num)
% visualization SVM_model
% red is 1, blue is -1
%
if nargin < 5
    grid_num=[];
end

if isempty(grid_num),grid_num=100;end
if isempty(axe_hdl),axe_hdl=gca();end

X=classify.X;
if iscell(X) X=X{1}; end
Y=classify.Y;
if iscell(Y) Y=Y{1}; end
predict_function=@(X_pred)classify.predict(X_pred);

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

dim=length(low_bou);

% check dimension
[x_num,vari_num]=size(X);
if vari_num > 2
    error('classifyVisualization: dimension large than 2');
end

% identify point properties
positive_idx=[];
negative_idx=[];
for x_index=1:x_num
    x=X(x_index,:);
    if ~(sum(x > up_bou) || sum(x < low_bou))
        if Y(x_index) > 0
            positive_idx=[positive_idx;x_index];
        else
            negative_idx=[negative_idx;x_index];
        end
    end
end

switch dim
    case 1
        d_bou=(up_bou-low_bou)/grid_num;
        X=low_bou:d_bou:up_bou;
        class_pred=zeros(grid_num+1,1);
        predict_fval=zeros(grid_num+1,1);
        for x_index=1:(grid_num+1)
            X_predict=X(x_index);
            [class_pred(x_index),predict_fval(x_index)]=predict_function(X_predict);
        end
        line(axe_hdl,X,class_pred,'LineStyle','none','Marker','d','Color','k');
        line(axe_hdl,X,predict_fval);
        xlabel('X');
        ylabel('Possibility of 1');
        if ~isempty(positive_idx)
            line(axe_hdl,X(positive_idx),Y(positive_idx),'LineStyle','none','Marker','o','Color','r');
        end
        if ~isempty(negative_idx)
            line(axe_hdl,X(negative_idx),Y(negative_idx),'LineStyle','none','Marker','o','Color','b');
        end
    case 2
        % draw zero value line
        grid_num=100;
        d_bou=(up_bou-low_bou)/grid_num;
        [X_draw,Y_draw]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
        try
            % generate all predict list
            X_predict=[X_draw(:),Y_draw(:)];
            [class_pred,prblty_pred]=predict_function(X_predict);
            class_pred=reshape(class_pred,grid_num+1,grid_num+1);
            prblty_pred=reshape(prblty_pred,grid_num+1,grid_num+1);
        catch
            class_pred=zeros(grid_num+1);
            prblty_pred=zeros(grid_num+1);
            for x_index=1:grid_num+1
                for y_index=1:grid_num+1
                    X_predict=([x_index,y_index]-1).*d_bou+low_bou;
                    [class_pred(y_index,x_index)]=...
                        predict_function(X_predict);
                end
            end
        end
        contour(axe_hdl,X_draw,Y_draw,class_pred);
        hold on;
%         contour(axe_hdl,X_draw,Y_draw,prblty_pred);
        hold off;
        xlabel('X');
        ylabel('Y');

        % draw point
        if ~isempty(positive_idx)
            line(axe_hdl,X(positive_idx,1),X(positive_idx,2),...
                'LineStyle','none','Marker','o','Color','r');
        end
        if ~isempty(negative_idx)
            line(axe_hdl,X(negative_idx,1),X(negative_idx,2),...
                'LineStyle','none','Marker','o','Color','b');
        end
end
end
