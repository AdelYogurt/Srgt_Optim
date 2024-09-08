function drawFcn(axe_hdl,draw_fcn,low_bou,up_bou,...
    grid_num,Y_min,Y_max,draw_dim)

if nargin < 8
    draw_dim=[];
    if nargin < 7
        Y_max=[];
        if nargin < 6
            Y_min=[];
            if nargin < 5
                grid_num=[];
            end
        end
    end
end

if isempty(grid_num), grid_num=100; end
if isempty(Y_max), Y_max=inf;end
if isempty(Y_min), Y_min=-inf;end
if isempty(axe_hdl), axe_hdl=gca();end

low_bou=low_bou(:)';
up_bou=up_bou(:)';
dim=length(low_bou);
d_bou=(up_bou-low_bou)/grid_num;

switch dim
    case 1
        x_mat=low_bou:d_bou:up_bou;
        fval_mat=zeros(grid_num+1,1);
        for x_idx=1:(grid_num+1)
            predict_x=x_mat(x_idx);
            fval_mat(x_idx)=draw_fcn(predict_x);
        end
        line(axe_hdl,x_mat,fval_mat);

        xlabel('x');
        zlabel('value');
    case 2
        [x_mat,y_mat]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
        fval_mat=zeros(grid_num+1,2);
        for x_idx=1:grid_num+1
            for y_idx=1:grid_num+1
                predict_x=([x_idx,y_idx]-1).*d_bou+low_bou;
                fval_mat(y_idx,x_idx,:)=draw_fcn(predict_x);
            end
        end
        fval_mat(fval_mat > Y_max)=Y_max;
        fval_mat(fval_mat < Y_min)=Y_min;
        surface(x_mat,y_mat,fval_mat(:,:,1),'FaceAlpha',0.5,'EdgeColor','none');

        xlabel('x');
        ylabel('y');
        zlabel('value');
        view(3);
    otherwise
        if isempty(draw_dim)
            warning('drawFcn: lack draw_dimension input, using default value dimension [1 2]')
            draw_dim=[1,2];
        end
        d_bou=(up_bou(draw_dim)-low_bou(draw_dim))/grid_num;
        mid=(up_bou+low_bou)/2;
        [x_mat,y_mat]=meshgrid(...
            low_bou(draw_dim(1)):d_bou(1):up_bou(draw_dim(1)),...
            low_bou(draw_dim(2)):d_bou(2):up_bou(draw_dim(2)));
        fval_mat=zeros(grid_num+1);
        for x_idx=1:grid_num+1
            for y_idx=1:grid_num+1
                predict_x=mid;
                predict_x(draw_dim)=([x_idx,y_idx]-1).*d_bou+low_bou(draw_dim);
                fval_mat(y_idx,x_idx)=draw_fcn(predict_x);
            end
        end
        fval_mat(fval_mat > Y_max)=Y_max;
        fval_mat(fval_mat < Y_min)=Y_min;

        surface(x_mat,y_mat,fval_mat(:,:),'FaceAlpha',0.5,'EdgeColor','none');

        xlabel('x');
        ylabel('y');
        zlabel('value');
        view(3);
end

grid on;
drawnow;
end