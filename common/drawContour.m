function contour_handle=drawContour(draw_fcn,low_bou,up_bou,...
    grid_num,threshold_contour,draw_option,fig_hdl,draw_dimension)
if nargin < 8
    draw_dimension=[];
    if nargin < 7
        fig_hdl=[];
        if nargin < 6
            draw_option=[];
            if nargin < 5
                threshold_contour=[];
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
axes_hdl=fig_hdl.CurrentAxes;
if isempty(axes_hdl)
    axes_hdl=axes(fig_hdl);
end
axes_context=axes_hdl.Children;
dimension=length(low_bou);

if isempty(grid_num)
    grid_num=100;
end
if isempty(draw_option)
    draw_option=struct('ShowText','on','Fill','on');
end
p_list=fieldnames(draw_option);

switch dimension
    case 2
        d_bou=(up_bou-low_bou)/grid_num;
        [draw_X,draw_Y]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
        draw_Fval=zeros(grid_num+1,grid_num+1);
        for x_idx=1:grid_num+1
            for y_idx=1:grid_num+1
                x_pred=([x_idx,y_idx]-1).*d_bou+low_bou;
                draw_Fval(y_idx,x_idx)=draw_fcn(x_pred);
            end
        end
        
        hold(axes_hdl,'on');
        if isempty(threshold_contour)
            [~,contour_handle]=contour(axes_hdl,draw_X,draw_Y,draw_Fval);
        else
            [~,contour_handle]=contour(axes_hdl,draw_X,draw_Y,draw_Fval,threshold_contour);
        end
        for p_idx=1:length(p_list)
            contour_handle.set(p_list{p_idx},draw_option.(p_list{p_idx}))
        end

        xlabel('X');
        ylabel('Y');
        view(2);

    otherwise
        if isempty(draw_dimension)
            warning('drawContour: lack draw_dimension input, using default value dimension [1 2]')
            draw_dimension=[1,2];
        end
        d_bou=(up_bou(draw_dimension)-low_bou(draw_dimension))/grid_num;
        middle=(up_bou+low_bou)/2;
        [draw_X,draw_Y]=meshgrid(...
            low_bou(draw_dimension(1)):d_bou(1):up_bou(draw_dimension(1)),...
            low_bou(draw_dimension(2)):d_bou(2):up_bou(draw_dimension(2)));
        draw_Fval=zeros(grid_num+1,grid_num+1);

        for x_idx=1:grid_num+1
            for y_idx=1:grid_num+1
                x_pred=middle;
                x_pred(draw_dimension)=([x_idx,y_idx]-1).*d_bou+low_bou(draw_dimension);
                draw_Fval(y_idx,x_idx)=draw_fcn(x_pred);
            end
        end

        hold(axes_hdl,'on');
        if isempty(threshold_contour)
            [~,contour_handle]=contour(axes_hdl,draw_X,draw_Y,draw_Fval);
        else
            [~,contour_handle]=contour(axes_hdl,draw_X,draw_Y,draw_Fval,threshold_contour);
        end
        for p_idx=1:length(p_list)
            contour_handle.set(p_list{p_idx},draw_option.(p_list{p_idx}))
        end

        xlabel('X');
        ylabel('Y');
        view(2);
end
end