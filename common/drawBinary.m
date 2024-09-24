function drawBinary(draw_fcn,low_bou,up_bou,...
    grid_num,threshold,draw_option,fig_hdl)
% draw only two value function
% only support two dimension function
%
if nargin < 7
    fig_hdl=[];
    if nargin < 6
        draw_option=[];
        if nargin < 5
            threshold=[];
            if nargin < 4
                grid_num=[];
            end
        end
    end
end
low_bou=low_bou(:)';
up_bou=up_bou(:)';

if isempty(fig_hdl)
    fig_hdl=figure(103);
end
axes_handle=fig_hdl.CurrentAxes;
if isempty(axes_handle)
    axes_handle=axes(fig_hdl);
end
axes_context=axes_handle.Children;
dimension=length(low_bou);

if isempty(grid_num)
    grid_num=500;
end
if isempty(threshold)
    threshold=0;
end

if isempty(draw_option)
    draw_option=struct('FaceColor','#ffa7a7');
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
        draw_Fval((draw_Fval > threshold))=nan;
        draw_Fval((draw_Fval < threshold))=threshold-0.1;
        hold(axes_handle,'on');
        %         contourf(axes_handle,draw_X,draw_Y,draw_Fval,'LevelList', [-inf, 0.5, inf],'LineStyle','none')
        [surface_hdl]=surface(axes_handle,draw_X,draw_Y,draw_Fval,'EdgeColor','none');
        for p_idx=1:length(p_list)
            surface_hdl.set(p_list{p_idx},draw_option.(p_list{p_idx}))
        end
        xlabel('X');
        ylabel('Y');
        zlabel('value');
        view(2);

    otherwise
        error('drawBinary: dimension no support')
end

end