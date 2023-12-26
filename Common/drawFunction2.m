
function drawFunction2(draw_function,low_bou,up_bou,...
    grid_number,Y_min,Y_max,figure_handle)
if nargin < 7
    figure_handle=figure(10);
    if nargin < 6
        Y_max=inf;
        if nargin < 5
            Y_min=-inf;
            if nargin < 4
                grid_number=100;
            end
        end
    end
end
low_bou=low_bou(:)';
up_bou=up_bou(:)';

axes_handle=figure_handle.CurrentAxes;
if isempty(axes_handle)
    axes_handle=axes(figure_handle);
end
axes_context=axes_handle.Children;
dimension=length(low_bou);

switch dimension
    case 1
        d_bou=(up_bou-low_bou)/grid_number;
        X__=low_bou:d_bou:up_bou;
        fval__=zeros(grid_number+1,2);
        for x_index__=1:(grid_number+1)
            fval__(x_index__,:)=draw_function(X__(x_index__));
        end
        line(axes_handle,X__,fval__(:,1));
        line(axes_handle,X__,fval__(:,2));
        xlabel('X');
        ylabel('value');

    case 2
        d_bou=(up_bou-low_bou)/grid_number;
        [X__,Y]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
        fval__=zeros(grid_number+1,grid_number+1,2);
        for x_index__=1:grid_number+1
            for y_index__=1:grid_number+1
                predict_x=([x_index__,y_index__]-1).*d_bou+low_bou;
                fval__(y_index__,x_index__,:)=draw_function(predict_x);
            end
        end
        fval__(find(fval__ > Y_max))=Y_max;
        fval__(find(fval__ < Y_min))=Y_min;
        axes_context=[axes_context;
            surface(X__,Y,fval__(:,:,1),'FaceAlpha',0.5,'EdgeColor','none');
            surface(X__,Y,fval__(:,:,2),'FaceAlpha',0.5,'EdgeColor','none');];
        axes_handle.set('Children',axes_context);
        xlabel('X');
        ylabel('Y');
        zlabel('value');
        view(3);
end
end
