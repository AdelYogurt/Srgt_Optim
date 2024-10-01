function displaySrgt(axe_hdl,srgt,low_bou,up_bou,...
    grid_num,v_min,v_max,draw_dim)
% visualization polynamial respond surface model
% figrue is 100
%
% Copyright 2022 Adel
%
if nargin < 8
    draw_dim=[];
    if nargin < 7
        v_max=[];
        if nargin < 6
            v_min=[];
            if nargin < 5
                grid_num=[];
            end
        end
    end
end

if isempty(grid_num),grid_num=100;end
if isempty(v_max),v_max=inf;end
if isempty(v_min),v_min=-inf;end
if isempty(axe_hdl),axe_hdl=gca();end

x_list=srgt.X;
y_list=srgt.Y;
if iscell(x_list)
    x_list=x_list{end};
    y_list=y_list{end};
end
pred_fcn=@(x)srgt.predict(x);

% get boundary
if (nargin < 3 || isempty(low_bou)),low_bou=min(x_list,[],1);end
if (nargin < 4 || isempty(up_bou)),up_bou=max(x_list,[],1);end

if numel(low_bou) ~= numel(low_bou)
    error('displaySrgt: boundary incorrect');
end

low_bou=low_bou(:)';
up_bou=up_bou(:)';
dim=length(low_bou);
d_bou=(up_bou-low_bou)/grid_num;

switch dim
    case 1
        x_mat=low_bou:d_bou:up_bou;
        V_mat=zeros(grid_num+1,1);
        for x_index=1:grid_num+1
            x_pred=(x_index-1).*d_bou+low_bou;
            V_mat(x_index)=pred_fcn(x_pred);
        end

        V_mat=min(V_mat,v_max);
        V_mat=max(V_mat,v_min);
        y_list=min(y_list,v_max);
        y_list=max(y_list,v_min);

        line(axe_hdl,x_mat,V_mat);
        line(axe_hdl,x_list,y_list,'Marker','o','LineStyle','none');

        xlabel('x');
        ylabel('y');

    case 2
        [x_mat,y_mat]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
        try
            % generate all predict list
            x_pred=[x_mat(:),y_mat(:)];
            [V_mat]=pred_fcn(x_pred);
            V_mat=reshape(V_mat,grid_num+1,grid_num+1);
        catch
            % if not support multi input
            V_mat=zeros(grid_num+1);
            for x_index=1:grid_num+1
                for y_index=1:grid_num+1
                    x_pred=([x_index,y_index]-1).*d_bou+low_bou;
                    [V_mat(y_index,x_index)]=pred_fcn(x_pred);
                end
            end
        end

        V_mat=min(V_mat,v_max);
        V_mat=max(V_mat,v_min);
        y_list=min(y_list,v_max);
        y_list=max(y_list,v_min);

        surface(axe_hdl,x_mat,y_mat,V_mat,'EdgeColor','none','FaceAlpha',0.5)
        line(axe_hdl,x_list(:,1),x_list(:,2),y_list,'Marker','o','LineStyle','none')

        xlabel('x');
        ylabel('y');
        zlabel('value');
        view(3);

    otherwise
        if isempty(draw_dim)
            warning('displaySrgt: lack draw_dimension input, using default value dimension [1 2]')
            draw_dim=[1,2];
        end
        d_bou=(up_bou(draw_dim)-low_bou(draw_dim))/grid_num;
        mid=(up_bou+low_bou)/2;
        [x_mat,y_mat]=meshgrid(...
            low_bou(draw_dim(1)):d_bou(1):up_bou(draw_dim(1)),...
            low_bou(draw_dim(2)):d_bou(2):up_bou(draw_dim(2)));

        try
            % generate all predict list
            x_pred=repmat(mid,(grid_num+1)^2,1);
            x_pred(:,draw_dim)=[x_mat(:),y_mat(:)];
            [V_mat]=pred_fcn(x_pred);
            V_mat=reshape(V_mat,grid_num+1,grid_num+1);
        catch
            % if not support multi input
            V_mat=zeros(grid_num+1);
            for x_idx=1:grid_num+1
                for y_idx=1:grid_num+1
                    predict_x=mid;
                    predict_x(draw_dim)=([x_idx,y_idx]-1).*d_bou+low_bou(draw_dim);
                    V_mat(y_idx,x_idx)=pred_fcn(predict_x);
                end
            end
        end

        V_mat=min(V_mat,v_max);
        V_mat=max(V_mat,v_min);
        y_list=min(y_list,v_max);
        y_list=max(y_list,v_min);

        surface(x_mat,y_mat,V_mat,'FaceAlpha',0.5,'EdgeColor','none');
        line(axe_hdl,x_list(:,draw_dim(1)),x_list(:,draw_dim(2)),y_list,'Marker','o','LineStyle','none')

        xlabel('x');
        ylabel('y');
        zlabel('value');
        view(3);
end

grid on;
drawnow;
end
