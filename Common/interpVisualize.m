function interpVisualize(model,low_bou,up_bou,...
    grid_number,Y_min,Y_max,figure_handle)
% visualization polynamial respond surface model
% figrue is 100
%
% Copyright 2022 Adel
%
if nargin < 7
    figure_handle=[];
    if nargin < 6
        Y_max=[];
        if nargin < 5
            Y_min=[];
            if nargin < 4
                grid_number=[];
            end
        end
    end
end

if isempty(figure_handle)
    figure_handle=figure(51);
end
axes_handle=figure_handle.CurrentAxes;
if isempty(axes_handle)
    axes_handle=axes(figure_handle);
end
axes_context=axes_handle.Children;
if ~isempty(axes_context)
    delete(axes_context);
    axes_context=[];
end

x_list=model.X;
y_list=model.Y;
if iscell(x_list)
    x_list=x_list{1};
    y_list=y_list{1};
end
predict_function=model.predict_function;

% get boundary
if (nargin < 2 || isempty(low_bou))
    low_bou=min(x_list,[],1);
end
if (nargin < 3 || isempty(up_bou))
    up_bou=max(x_list,[],1);
end

if size(low_bou,1) ~= size(low_bou,1)
    error('interpolationRadialBasisVisualize: boundary incorrect');
end
if size(low_bou,1) > 2
    error('interpolationRadialBasisVisualize: dimension large than two');
end

if isempty(grid_number)
    grid_number=200;
end
if isempty(Y_max)
    Y_max=inf;
end
if isempty(Y_min)
    Y_min=-inf;
end

d_bou=(up_bou-low_bou)/grid_number;

if size(x_list,2) == 1
    predict_result=zeros(grid_number+1,1);
    X_draw=low_bou:d_bou:(low_bou+grid_number*d_bou);
    for x_index=1:grid_number+1
        predict_x=(x_index-1).*d_bou+low_bou;
        predict_result(x_index)=predict_function(predict_x);
    end
    predict_result=min(predict_result,Y_max);
    predict_result=max(predict_result,Y_min);
    axes_context=[axes_context;line(X_draw,predict_result)];
    axes_context=[axes_context;line(x_list,y_list,'Marker','o','LineStyle','none')];
    axes_handle.set('Children',axes_context);
    xlabel('X');
    ylabel('Y');
elseif size(x_list,2) == 2
    [X_draw,Y_draw]=meshgrid(low_bou(1):d_bou(1):(low_bou(1)+grid_number*d_bou(1)),...
        low_bou(2):d_bou(2):(low_bou(2)+grid_number*d_bou(2)));
    try
        % generate all predict list
        predict_x=[X_draw(:),Y_draw(:)];
        [~,predict_result]=predict_function(predict_x);
        predict_result=reshape(predict_result,grid_number+1,grid_number+1);
    catch
        % if not support multi input
        predict_result=zeros(grid_number+1);
        for x_index=1:grid_number+1
            for y_index=1:grid_number+1
                predict_x=([x_index,y_index]-1).*d_bou+low_bou;
                [predict_result(y_index,x_index)]=predict_function(predict_x);
            end
        end
    end
    predict_result=min(predict_result,Y_max);
    predict_result=max(predict_result,Y_min);
%     contour(axes_handle,X_draw,Y_draw,predict_result);
    axes_context=[axes_context;surface(X_draw,Y_draw,predict_result,'FaceAlpha',0.5,'EdgeColor','none')];
    axes_context=[axes_context;line(x_list(:,1),x_list(:,2),y_list,'Marker','o','LineStyle','none')];
    axes_handle.set('Children',axes_context);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    view(3);
end
end
