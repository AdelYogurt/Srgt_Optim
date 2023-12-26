function drawFcn(draw_fcn,low_bou,up_bou,...
    grid_num,Y_min,Y_max,fig_hdl,draw_dimension)
if nargin < 8
    draw_dimension=[];
    if nargin < 7
        fig_hdl=[];
        if nargin < 6
            Y_max=[];
            if nargin < 5
                Y_min=[];
                if nargin < 4
                    grid_num=[];
                end
            end
        end
    end
end
low_bou=low_bou(:)';
up_bou=up_bou(:)';

if isempty(fig_hdl)
    fig_hdl=figure(101);
end
axes_handle=fig_hdl.CurrentAxes;
if isempty(axes_handle)
    axes_handle=axes(fig_hdl);
end
axes_context=axes_handle.Children;
dimension=length(low_bou);

if isempty(grid_num)
    grid_num=100;
end
if isempty(Y_max)
    Y_max=inf;
end
if isempty(Y_min)
    Y_min=-inf;
end

switch dimension
    case 1
        d_bou=(up_bou-low_bou)/grid_num;
        draw_X=low_bou:d_bou:up_bou;
        draw_Fval=zeros(grid_num+1,1);
        for x_index__=1:(grid_num+1)
            predict_x=draw_X(x_index__);
            draw_Fval(x_index__)=draw_fcn(predict_x);
        end
        line(axes_handle,draw_X,draw_Fval);
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
        axes_context=[axes_context;surface(draw_X,draw_Y,draw_Fval(:,:,1),'FaceAlpha',0.5,'EdgeColor','none');];
        axes_handle.set('Children',axes_context);
        xlabel('X');
        ylabel('Y');
        zlabel('value');
        view(3);

    otherwise
        if isempty(draw_dimension)
            warning('drawFcn: lack draw_dimension input, using default value dimension [1 2]')
            draw_dimension=[1,2];
        end
        d_bou=(up_bou(draw_dimension)-low_bou(draw_dimension))/grid_num;
        middle=(up_bou+low_bou)/2;
        [draw_X,draw_Y]=meshgrid(...
            low_bou(draw_dimension(1)):d_bou(1):up_bou(draw_dimension(1)),...
            low_bou(draw_dimension(2)):d_bou(2):up_bou(draw_dimension(2)));
        draw_Fval=zeros(grid_num+1,grid_num+1);
        for x_index__=1:grid_num+1
            for y_index__=1:grid_num+1
                predict_x=middle;
                predict_x(draw_dimension)=([x_index__,y_index__]-1).*d_bou+low_bou(draw_dimension);
                draw_Fval(y_index__,x_index__)=draw_fcn(predict_x);
            end
        end
        draw_Fval(draw_Fval > Y_max)=Y_max;
        draw_Fval(draw_Fval < Y_min)=Y_min;
        axes_context=[axes_context;surface(draw_X,draw_Y,draw_Fval(:,:),'FaceAlpha',0.5,'EdgeColor','none');];
        axes_handle.set('Children',axes_context);
        xlabel('X');
        ylabel('Y');
        zlabel('value');
        view(3);
end
end