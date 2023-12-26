function drawFcn2(draw_fcn,low_bou,up_bou,...
    grid_num,Y_min,Y_max,fig_hdl)
if nargin < 7
    fig_hdl=figure(102);
    if nargin < 6
        Y_max=inf;
        if nargin < 5
            Y_min=-inf;
            if nargin < 4
                grid_num=100;
            end
        end
    end
end
low_bou=low_bou(:)';
up_bou=up_bou(:)';

axes_handle=fig_hdl.CurrentAxes;
if isempty(axes_handle)
    axes_handle=axes(fig_hdl);
end
axes_context=axes_handle.Children;
dimension=length(low_bou);

switch dimension
    case 1
        d_bou=(up_bou-low_bou)/grid_num;
        draw_X=low_bou:d_bou:up_bou;
        draw_Fval=zeros(grid_num+1,2);
        for x_index__=1:(grid_num+1)
            draw_Fval(x_index__,:)=draw_fcn(draw_X(x_index__));
        end
        line(axes_handle,draw_X,draw_Fval(:,1));
        line(axes_handle,draw_X,draw_Fval(:,2));
        xlabel('X');
        ylabel('value');

    case 2
        d_bou=(up_bou-low_bou)/grid_num;
        [draw_X,draw_Y]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
        draw_Fval=zeros(grid_num+1,grid_num+1,2);
        for x_index__=1:grid_num+1
            for y_index__=1:grid_num+1
                predict_x=([x_index__,y_index__]-1).*d_bou+low_bou;
                draw_Fval(y_index__,x_index__,:)=draw_fcn(predict_x);
            end
        end
        draw_Fval(draw_Fval > Y_max)=Y_max;
        draw_Fval(draw_Fval < Y_min)=Y_min;
        axes_context=[axes_context;
            surface(draw_X,draw_Y,draw_Fval(:,:,1),'FaceAlpha',0.5,'EdgeColor','none');
            surface(draw_X,draw_Y,draw_Fval(:,:,2),'FaceAlpha',0.5,'EdgeColor','none');];
        axes_handle.set('Children',axes_context);
        xlabel('X');
        ylabel('Y');
        zlabel('value');
        view(3);
end
end
