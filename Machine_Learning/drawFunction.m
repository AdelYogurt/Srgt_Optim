function drawFunction(draw_function,low_bou,up_bou,...
    grid_number,Y_min,Y_max,figure_handle,draw_dimension)
if nargin < 8
    draw_dimension=[];
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
end
low_bou=low_bou(:)';
up_bou=up_bou(:)';

if isempty(figure_handle)
    figure_handle=figure(101);
end
axes_handle=figure_handle.CurrentAxes;
if isempty(axes_handle)
    axes_handle=axes(figure_handle);
end
axes_context=axes_handle.Children;
dimension=length(low_bou);

if isempty(grid_number)
    grid_number=100;
end
if isempty(Y_max)
    Y_max=inf;
end
if isempty(Y_min)
    Y_min=-inf;
end

switch dimension
    case 1
        d_bou=(up_bou-low_bou)/grid_number;
        X__=low_bou:d_bou:up_bou;
        fval__=zeros(grid_number+1,1);
        for x_index__=1:(grid_number+1)
            predict_x=X__(x_index__);
            fval__(x_index__)=draw_function(predict_x);
        end
        line(axes_handle,X__,fval__);
        xlabel('X');
        ylabel('value');
        
    case 2
        d_bou=(up_bou-low_bou)/grid_number;
        [X__,Y__]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
        fval__=zeros(grid_number+1,grid_number+1);
        for x_index__=1:grid_number+1
            for y_index__=1:grid_number+1
                predict_x=([x_index__,y_index__]-1).*d_bou+low_bou;
                fval__(y_index__,x_index__)=draw_function(predict_x);
            end
        end
        fval__(find(fval__ > Y_max))=Y_max;
        fval__(find(fval__ < Y_min))=Y_min;
        axes_context=[axes_context;surface(X__,Y__,fval__(:,:),'FaceAlpha',0.5,'EdgeColor','none');];
        axes_handle.set('Children',axes_context);
        xlabel('X');
        ylabel('Y');
        zlabel('value');
        view(3);

    otherwise
        if isempty(draw_dimension)
            warning('drawFunction: lack draw_dimension input, using default value dimension [1 2]')
            draw_dimension=[1,2];
        end
        d_bou=(up_bou(draw_dimension)-low_bou(draw_dimension))/grid_number;
        middle=(up_bou+low_bou)/2;
        [X__,Y__]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
        fval__=zeros(grid_number+1,grid_number+1);
        for x_index__=1:grid_number+1
            for y_index__=1:grid_number+1
                predict_x=middle;
                predict_x(draw_dimension)=([x_index__,y_index__]-1).*d_bou+low_bou(draw_dimension);
                fval__(y_index__,x_index__)=draw_function(predict_x);
            end
        end
        fval__(find(fval__ > Y_max))=Y_max;
        fval__(find(fval__ < Y_min))=Y_min;
        axes_context=[axes_context;surface(X__,Y__,fval__(:,:),'FaceAlpha',0.5,'EdgeColor','none');];
        axes_handle.set('Children',axes_context);
        xlabel('X');
        ylabel('Y');
        zlabel('value');
        view(3);
end
end