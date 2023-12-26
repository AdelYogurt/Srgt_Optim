function [x_best,fval_best,NFE,output]=optimalPenaltyMethodNew...
    (object_function,variable_number,A,B,Aeq,Beq,low_bou,up_bou,nonlcon_function,...
    tolerance,iteration_max)
% constrain optimal function
% use out point penalty
% penalty coefficient iter every times
%
done=0;
NFE=0;
information_flag=0;
if nargin < 11
    iteration_max = 100;
    if nargin < 10
        tolerance=1e-6;
        if nargin < 9
            nonlcon_function=[];
            if nargin < 8
                up_bou=[];
                if nargin < 7
                    low_bou=[];
                    if narginr < 6
                        Beq=[];
                        if nargin < 5
                            Aeq=[];
                            if nargin < 4
                                B=[];
                                if nargin < 3
                                    A=[];
                                    if nargin < 2
                                        error('lack x initial');
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
result_x_best=zeros(iteration_max,variable_number);
result_fval_best=zeros(iteration_max,1);

iteration=0;

con_total_function=@(x) conTotalFunction...
    (x,A,B,Aeq,Beq,low_bou,up_bou,nonlcon_function);

x=low_bou+(up_bou-low_bou).*rand(length(low_bou),1);
[con,coneq]=con_total_function(x);
iteration__=0;
while sum(~(con < 0)) && iteration__ < iteration_max*10
    x=low_bou+(up_bou-low_bou).*rand(length(low_bou),1);
    [con,coneq]=con_total_function(x);
    iteration__=iteration__+1;
end

lambda=0;
miu=1;
miu_max=1000;
gama=2;

iteration=iteration+1;

% get constraint
con_number=size(con,1);
coneq_number=size(coneq,1);
lambda_list=lambda*ones(con_number+coneq_number,1);

% main loop
% each iteration call QNM/DFP to optimal
%
while ~done
    
    % generate penalty function
    penalty_function=@(x) penaltyFunction...
        (x,object_function,con_total_function,lambda_list(1:con_number),lambda_list(con_number+1:coneq_number),miu);
    
%     drawFunction(penalty_function,low_bou,up_bou,-inf,200);
    %     drawFunction(object_function,low_bou-[100;100],up_bou+[100;100]);
    
    % first iteraiton, due to lack fval gradient, so move out main loop
    [x_best,fval_best,NFE_QN,output_QN]=...
        optimalQuasiNewtonMethod(penalty_function,x);
    NFE=NFE+NFE_QN;
    
    fval_best=object_function(x_best);
    
    result_x_best_QN=output_QN.result_x_best;
%     line([x(1);result_x_best_QN(:,1)],[x(2);result_x_best_QN(:,2)],'Marker','o');
    
    result_x_best(iteration,:)=x_best';
    result_fval_best(iteration,:)=fval_best;
    iteration=iteration+1;
    
    if ((iteration > 2) && (norm(x_best-x_best_old,2) <= tolerance)) ||...
            iteration > iteration_max
        done=1;
    end
    
    if information_flag
        disp(['iteration:',num2str(iteration),'  NFE:',num2str(NFE),'  fval:',num2str(fval_best)]);
    end
    
    % updata lambda and miu
    [con,coneq]=con_total_function(x);
    lambda_list(1:con_number)=lambda_list(1:con_number)+2*miu*max(con,-lambda_list/2/miu);
    lambda_list(con_number+1:coneq_number)=lambda_list(con_number+1:coneq_number)+2*miu*coneq;
    if miu < miu_max
        miu=gama*miu;
    else
        miu=miu_max;
    end
    
    x_best_old=x_best;
    x=x_best;
end

result_x_best=result_x_best(1:iteration-1,:);
result_fval_best=result_fval_best(1:iteration-1);

output.result_x_best=result_x_best;
output.result_fval_best=result_fval_best;
output.lambda_list=lambda_list;
% drawFunction(penalty_function);
    function [con,coneq]=conTotalFunction...
            (x,A,B,Aeq,Beq,low_bou,up_bou,nonlcon_function)
        % function include all constraint
        %
        con=[];
        coneq=[];
        if ~isempty(A)
            if ~isempty(B)
                con=A*x-B;
            else
                con=A*x;
            end
        end
        if ~isempty(Aeq)
            if ~isempty(Beq)
                coneq=Aeq*x-Beq;
            else
                coneq=Aeq*x;
            end
        end
        if ~isempty(low_bou)
            con=[con;low_bou-x];
        end
        if ~isempty(up_bou)
            con=[con;x-up_bou];
        end
        if ~isempty(nonlcon_function)
            [con_nonl,coneq_nonl]=nonlcon_function(x);
            con=[con;con_nonl];
            coneq=[coneq;coneq_nonl];
        end
    end
    function fval=penaltyFunction...
            (x,object_function,con_total_function,lambda_list,lambdaeq_list,miu)
        % penalty function
        % augmented lagrange multiplier method was used
        %
        fval=object_function(x);
        if ~isempty(con_total_function)
            [con__,coneq__]=con_total_function(x);
            if ~isempty(con__)
                psi=max(con__,-lambda_list/2/miu);
                fval=fval+sum(lambda_list.*psi+miu*psi.*psi);
            end
            if ~isempty(coneq__)
                fval=fval+sum(lambdaeq_list.*coneq__+miu*coneq__.*coneq__);
            end
        end
    end
end

function drawFunction(object_function,low_bou,up_bou,...
    Y_min,Y_max,grid_number,figure_handle)
if nargin < 7
    figure_handle=figure(10);
    if nargin < 6
        grid_number=100;
        if nargin < 5
            Y_max=inf;
            if nargin < 4
                Y_min=-inf;
            end
        end
    end
end
axes_handle=figure_handle.CurrentAxes;
if isempty(axes_handle)
    axes_handle=axes(figure_handle);
end
dimension=size(low_bou,1);

switch dimension
    case 1
        d_bou=(up_bou-low_bou)/grid_number;
        X=low_bou:d_bou:up_bou;
        fval=zeros(grid_number+1,1);
        for x_index=1:(grid_number+1)
            fval(x_index)=object_function(X(x_index));
        end
        line(axes_handle,X,fval);
        xlabel('X');
        ylabel('value');
        
    case 2
        d_bou=(up_bou-low_bou)/grid_number;
        [X,Y]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
        fval=zeros(grid_number+1);
        for x_index=1:grid_number+1
            for y_index=1:grid_number+1
                predict_x=([x_index;y_index]-1).*d_bou+low_bou;
                fval(y_index,x_index)=object_function(predict_x);
            end
        end
        fval(find(fval > Y_max))=Y_max;
        fval(find(fval < Y_min))=Y_min;
        surf(axes_handle,X,Y,fval,'FaceAlpha',0.5,'EdgeColor','none');
        xlabel('X');
        ylabel('Y');
        zlabel('value');
        
    case 3
        d_bou=(up_bou-low_bou)/grid_number;
        [X,Y]=meshgrid(low_bou(1):d_bou(1):up_bou(1),low_bou(2):d_bou(2):up_bou(2));
        fval=zeros(grid_number+1);
        for x_index=1:grid_number+1
            for y_index=1:grid_number+1
                for z_index=1:grid_number+1
                    predict_x=([x_index;y_index;z_index]-1).*d_bou+low_bou;
                    fval(x_index,y_index)=object_function(predict_x);
                end
            end
        end
        fval(find(fval > Y_max))=Y_max;
        fval(find(fval < Y_min))=Y_min;
        surf(axes_handle,X',Y',fval,'FaceAlpha',0.5,'EdgeColor','none');
        xlabel('X');
        ylabel('Y');
        zlabel('value');
end

end

function [x_best,fval_best,NFE,output]=optimalQuasiNewtonMethod...
    (object_function,x_initial,tolrance,iteration_max)
% optimal funtion QuasiNewtonMethod version 1
% lineSearch method use wolfe Guidelines to decrease NFE
% do not test robustness!!!
%
% Copyright 2022 Adel
%
if nargin<4
    variable_number=size(x_initial,1);
    iteration_max=10*variable_number;
    if nargin<3
        tolrance=1e-6;
        if nargin<2
            error('lack x initial');
        end
    end
end
NFE=0;
done=0;
iteration=0;
information_flag=0;
hess_update_flag=2; % hess updata method, 1 is DFP, 2 is BFGS

result_x_best=zeros(iteration_max,variable_number);
result_fval_best=zeros(iteration_max,1);

if information_flag
    fprintf('%8s %8s  %12s    %8s\n','iteration','NFE','fval','lamada');
end
information_format='%6d    %8d %16.8g  %8.4g\n';

if hess_update_flag==1
    hessian_function=@(x_delta,gradient_delta,H) hessianDFPMethod(x_delta,gradient_delta,H);
elseif hess_update_flag==2
    hessian_function=@(x_delta,gradient_delta,H) hessianBFGSMethod(x_delta,gradient_delta,H);
end

x=x_initial;
fval=object_function(x);NFE=NFE+1;
[gradient,NFE_g]=getGradient(object_function,x,fval,tolrance);
gradient_initial=gradient;
NFE=NFE+NFE_g;
if norm(gradient,Inf)<=tolrance
    done=1;
    x_best=x;
    fval_best=fval;
    gradient_best=gradient;
end

if information_flag
    fprintf('%6d    %8d %16.8g\n',iteration,NFE,fval);
end

iteration=iteration+1;
H=eye(size(x_initial,1));

% main loop
%
while ~done
    
    lamada=1;
    if iteration==1
        lamada=min(1/norm(gradient,Inf),1);
    end
    
    %     if iteration==6
    %        disp('stop');
    %     end
    
    direction=-H*gradient;
    
    % line search best step
    [x_new,fval_new,gradient_new,lamada_best,NFC_line]=...
        lineSearch(object_function,direction,x,fval,gradient,lamada);
    NFE=NFE+NFC_line;
    
    if judgeQuit(tolrance,x,x_new,gradient_initial,...
            gradient_new,iteration,iteration_max)
        done=1;
        x_best=x;
        fval_best=fval;
        gradient_best=gradient;
    end
    
    x_delta=x_new-x;
    gradient_delta=gradient_new-gradient;
    if iteration==1
        H=x_delta'*gradient_delta/(gradient_delta'*gradient_delta)*H;
    end
    H=hessian_function(x_delta,gradient_delta,H);
    
    x=x_new;
    fval=fval_new;
    gradient=gradient_new;
    
    if information_flag
        fprintf(information_format,iteration,NFE,fval,lamada_best);
    end
    
    % record data
    result_x_best(iteration,:)=x';
    result_fval_best(iteration,:)=fval;
    iteration=iteration+1;
end

result_x_best=result_x_best(1:iteration-1,:);
result_fval_best=result_fval_best(1:iteration-1);

output.result_x_best=result_x_best;
output.result_fval_best=result_fval_best;
output.gradient_best=gradient_best;
output.iteration=iteration;

    function judgement=judgeQuit...
            (tolrance,x,x_new,gradient_initial,gradient_new,iteration,iteration_max)
        judgement=0;
        if (norm((x-x_new)./(1+abs(x)),2)<=tolrance)
            judgement=1;
        end
        if (norm(gradient_new,2)<=tolrance*(norm(gradient_initial,2)+1))
            judgement=1;
        end
        if iteration>iteration_max
            judgement=1;
        end
    end
    function H=hessianDFPMethod(x_delta,g_delta,H)
        % updata matrix H DFP method
        A=x_delta*x_delta'/(x_delta'*g_delta);
        B=H*g_delta*(H*g_delta)'/(g_delta'*H*g_delta);
        if isnan(A(1,1)) || isnan(B(1,1))
            index__=find(isnan(A));
            A(index__)=0;
%             disp('hessianDFPMethod: nan!')
        end
        H=H+A-B;
    end
    function H=hessianBFGSMethod(x_delta,g_delta,H)
        % updata matrix H BFGS method
        A=(1+g_delta'*H*g_delta/(x_delta'*g_delta))*x_delta*x_delta'/(x_delta'*g_delta);
        B=(x_delta*(H*g_delta)'+H*g_delta*x_delta')/(x_delta'*g_delta);
        if isnan(A(1,1)) || isnan(B(1,1))
            disp('hessianDFPMethod: nan!')
        end
        H=H+A-B;
    end
end

function [x_best,fval_best,gradient_best,lamada_best,NFE]=lineSearch...
    (object_function,direction,x_initial,fval_initial,gradient_initial,lamada)
% quadratic/cubic interpolation method version 10
% quadratic/cubic interpolation optimization was used
% direction is search line, direction*gradient is one dimension differ
% do not move x_initial, fval_initial, gradient_initial
% just change lamada to move x change fval and gradient
% _old is record of last time
% search range default is 1e-6 to 1e6
% each time's fval should be recond if it's a rand search
%
cubic_done=0;
quad_done=0;
iteration_max=length(x_initial)+10;
c1=0.01;
c2=0.9;
NFE=0;
tolorance=1e-1;
max_change=1e-1;
serach_range=[1e-6,1e6];

dir=1;
if direction'*gradient_initial > 0
    lamada=-lamada;
    dir=-1;
end

draw_range=0.00001;
draw_interval=draw_range*0.02;
draw=0;

if draw
    close all hidden;
    figure(1);
    for draw_lamada=0:dir*draw_interval:dir*draw_range
        line(draw_lamada,object_function(x_initial+draw_lamada*direction),'Marker','o');
    end
    %     axis([-dra+w_range,draw_range,-5,20]);
end

x=x_initial+lamada*direction;
fval=object_function(x);NFE = NFE + 1;
gradient=NaN;

data_list=[0,lamada;fval_initial,fval;gradient_initial'*direction,NaN];

if (fval<=(fval_initial+c1*lamada*gradient_initial'*direction))
    [gradient,NFE_g]=getGradient(object_function,x,fval);NFE=NFE+NFE_g;
    data_list(3,2)=direction'*gradient;
    if (abs(gradient'*direction)<=c2*abs(gradient_initial'*direction)) % whether x satisfy torlance/volfe
        quad_done=1;
        cubic_done=1;
        x_best=x;
        fval_best=fval;
        gradient_best=gradient;
        lamada_best=lamada;
    end
end

iteration=1;

interp_x0=0;
interp_fval0=fval_initial;

% quadratic interpolation optimal
% quadratic interpolation is to find another point for cubic interpolation
% so out point fval should large than mid point
% 0 < 1 < 2, f(1) < f(2), f(1) < f(0)
interp_x1=lamada;
interp_fval1=fval;
while ~quad_done
    if interp_x1 > serach_range(2)
        error('lineSearch: can not find min in search range');
        %     elseif interp_x1 < serach_range(1)
        %         quad_done=1;
        %         cubic_done=1;
        %
        %         lamada=interp_x1;
        %         flag=find(data_list(1,:)==interp_x1);
        %         if isempty(flag)
        %             x=x_initial+lamada*direction;
        %             fval=object_function(x);NFE = NFE + 1;
        %             [gradient,NFE_g]=getGradient(object_function,x,fval);NFE=NFE+NFE_g;
        %         else
        %             x=x_initial+lamada*direction;
        %             fval=data_list(2,flag);
        %             gradient=data_list(3,flag);
        %             if isnan(gradient)
        %                 [gradient,NFE_g]=getGradient(object_function,x,fval);NFE=NFE+NFE_g;
        %             end
        %         end
        %         x_best=x;
        %         fval_best=fval;
        %         gradient_best=gradient;
        %         lamada_best=lamada;
    end
    % quadratic interpolation equation
    interp_base=interp_x1;
    interp_matrix=[0,0,1;
        0,1,0
        1,1,1;];
    interp_value=[fval_initial;direction'*gradient_initial*interp_base;interp_fval1];
    %     if rcond(interp_matrix)<1e-16
    %        disp('stop');
    %     end
    coefficient_quad=interp_matrix\interp_value;
    
    if draw
        x_draw=0:dir*draw_interval:dir*draw_range;
        x_draw=x_draw/interp_base;
        line(x_draw*interp_base,coefficient_quad(1)*x_draw.^2+coefficient_quad(2)*x_draw+coefficient_quad(3));
    end
    
    if coefficient_quad(1) <= 0
        % this mean function still decline, need to add more
        interp_x1=interp_x1*2;
        interp_fval1=object_function(x_initial+interp_x1*direction);NFE = NFE + 1;
        if isempty(find(data_list(1,:)==interp_x1))
            flag=find(dir*data_list(1,:)>dir*interp_x1);
            if isempty(flag)
                flag=size(data_list,2)+1;
            end
            data_list=[data_list(:,1:(flag(1)-1)),[interp_x1;interp_fval1;NaN],data_list(:,(flag(1)):end)];
        end
        
    else
        interp_xp_rel=-coefficient_quad(2)/coefficient_quad(1)/2;
        interp_xp=interp_base*interp_xp_rel;
        % limit the max change
        if abs(interp_xp/interp_x1) < max_change
            interp_xp=interp_x1/2;
        elseif abs(interp_x1/interp_xp) < max_change
            interp_xp=interp_x1*2;
        end
        interp_fvalp=object_function(x_initial+interp_xp*direction);NFE = NFE + 1;
        if isempty(find(data_list(1,:)==interp_xp))
            flag=find(dir*data_list(1,:)>dir*interp_xp);
            if isempty(flag)
                flag=size(data_list,2)+1;
            end
            data_list=[data_list(:,1:(flag(1)-1)),[interp_xp;interp_fvalp;NaN],data_list(:,(flag(1)):end)];
        end
        
        % there are five situations to discuss
        % interp_xp is new point to discuss
        if dir*interp_xp < dir*interp_x1
            
            if interp_fvalp < interp_fval0
                % go to cubic interpolate
                quad_done = 1;
                
                interp_x2=interp_x1;
                interp_fval2=interp_fval1;
                interp_x1=interp_xp;
                interp_fval1=interp_fvalp;
                
                lamada_old=lamada;
                fval_old=fval;
                gradient_old=gradient;
                
                lamada=interp_x1;
                fval=interp_fval1;
                x=x_initial+lamada*direction;
                [judgement,NFE,gradient]=judgeQuit...
                    (fval_initial,gradient_initial,object_function,x,lamada,fval,NFE,direction,c1,c2,...
                    iteration,iteration_max,tolorance,lamada_old);
                if ~isempty(gradient)
                    data_list(3,find(data_list(1,:)==interp_x1))=direction'*gradient;
                end
                if judgement
                    if isempty(gradient)
                        [gradient,NFE_g]=getGradient(object_function,x,fval);NFE=NFE+NFE_g;
                    end
                    cubic_done=1;
                    x_best=x;
                    fval_best=fval;
                    gradient_best=gradient;
                    lamada_best=lamada;
                end
            else
                % quadratic interpolate again
                interp_x1=interp_xp;
                interp_fval1=interp_fvalp;
            end
        else
            while interp_fval1 > interp_fvalp
                % interp_fvalp is low than interp_fval1
                % mean fval can decrease if move forward
                
                if interp_x1>serach_range(2)
                    error('lineSearch: can not find min in search range');
                end
                
                interp_x1=interp_xp;
                interp_fval1=interp_fvalp;
                
                lamada_old=lamada;
                fval_old=fval;
                gradient_old=gradient;
                
                lamada=interp_x1;
                fval=interp_fval1;
                x=x_initial+lamada*direction;
                [judgement,NFE,gradient]=judgeQuit...
                    (fval_initial,gradient_initial,object_function,x,lamada,fval,NFE,direction,c1,c2,...
                    iteration,iteration_max,tolorance,lamada_old);
                if ~isempty(gradient)
                    data_list(3,find(data_list(1,:)==interp_x1))=direction'*gradient;
                end
                if judgement
                    cubic_done=1;
                    x_best=x;
                    fval_best=fval;
                    gradient_best=gradient;
                    lamada_best=lamada;
                end
                
                interp_xp=2*interp_xp;
                interp_fvalp=object_function(x_initial+interp_xp*direction);NFE = NFE + 1;
                if isempty(find(data_list(1,:)==interp_xp))
                    flag=find(data_list(1,:)>interp_xp);
                    if isempty(flag)
                        flag=size(data_list,2)+1;
                    end
                    data_list=[data_list(:,1:(flag(1)-1)),[interp_xp;interp_fvalp;NaN],data_list(:,(flag(1)):end)];
                end
            end
            
            % go to cubic interpolate
            quad_done = 1;
            
            interp_x2=interp_xp;
            interp_fval2=interp_fvalp;
        end
    end
    iteration=iteration+1;
    if iteration >= iteration_max
        quad_done=1;
        cubic_done=1;
        x_best=x;
        fval_best=fval;
        if isempty(gradient) || isnan(gradient(1))
            [gradient,NFE_g]=getGradient(object_function,x,fval);NFE=NFE+NFE_g;
        end
        gradient_best=gradient;
        lamada_best=lamada;
    end
end

% cubic interpolation optimal loop
% lamada=interpolate_x1
while ~cubic_done
    % cubic interpolation equation
    % to avoid singular matrix occur, use relative coordinate
    %     if dir*interp_x2 < dir*interp_x1
    %         disp('stop');
    %     end
    if abs(interp_x1/interp_x2-1)<0.001
        interp_x1=interp_x1/2;
        interp_fval1=object_function(x_initial+interp_x1*direction);
        if isempty(find(data_list(1,:)==interp_x1))
            flag=find(dir*data_list(1,:)>dir*interp_x1);
            if isempty(flag)
                flag=size(data_list,2)+1;
            end
            data_list=[data_list(:,1:(flag(1)-1)),[interp_x1;interp_fval1;NaN],data_list(:,(flag(1)):end)];
        end
    end
    interp_base=interp_x2;
    interp_x1_relative=interp_x1/interp_x2;
    interp_matrix=[0,0,0,1;
        0,0,1,0;
        interp_x1_relative^3,interp_x1_relative^2,interp_x1_relative,1;
        1,1,1,1;];
    
    interp_value=[fval_initial;direction'*gradient_initial*interp_base;interp_fval1;interp_fval2];
    [interp_p_rel,coefficient_cubic]=minCubicInterpolate(interp_matrix,interp_value);
    
    if draw
        x_draw=0:dir*draw_interval:dir*draw_range;
        x_draw=x_draw/interp_base;
        line(x_draw*interp_base,coefficient_cubic(1)*x_draw.^3+coefficient_cubic(2)*x_draw.^2+...
            coefficient_cubic(3)*x_draw+coefficient_cubic(4));
    end
    
    interp_xp=interp_p_rel*interp_base;
    % limit the max change
    if abs(interp_xp/interp_x1) < max_change
        interp_xp=dir*interp_x1/2;
    elseif abs(interp_x1/interp_xp) < max_change
        interp_xp=dir*interp_x1*2;
    elseif abs(interp_xp-interp_x1)/interp_x1 < max_change
        interp_xp=interp_x2/2;
    end
    interp_fvalp=object_function(x_initial+interp_xp*direction);NFE = NFE + 1;
    if isempty(find(data_list(1,:)==interp_xp))
        flag=find(dir*data_list(1,:)>dir*interp_xp);
        if isempty(flag)
            flag=size(data_list,2)+1;
        end
        data_list=[data_list(:,1:(flag(1)-1)),[interp_xp;interp_fvalp;NaN],data_list(:,(flag(1)):end)];
    end
    
    % there are six situations to discuss
    if dir*interp_xp ==  dir*interp_x1
        interp_xp=(interp_x1+interp_x2)/2; % interp_xp > interp_x1
        interp_fvalp=object_function(x_initial+interp_xp*direction);NFE = NFE + 1;
        if isempty(find(data_list(1,:)==interp_xp))
            flag=find(dir*data_list(1,:)>dir*interp_xp);
            if isempty(flag)
                flag=size(data_list,2)+1;
            end
            data_list=[data_list(:,1:(flag(1)-1)),[interp_xp;interp_fvalp;NaN],data_list(:,(flag(1)):end)];
        end
    end
    if dir*interp_xp <  dir*interp_x1
        if interp_fvalp < interp_fval1
            interp_x2=interp_x1;
            interp_fval2=interp_fval1;
            interp_x1=interp_xp;
            interp_fval1=interp_fvalp;
            
            lamada_old=lamada;
            fval_old=fval;
            gradient_old=gradient;
            lamada=interp_x1;
            fval=interp_fval1;
            x=x_initial+lamada*direction;
            [judgement,NFE,gradient]=judgeQuit...
                (fval_initial,gradient_initial,object_function,x,lamada,fval,NFE,direction,c1,c2,...
                iteration,iteration_max,tolorance,lamada_old);
            if ~isempty(gradient)
                data_list(3,find(data_list(1,:)==interp_x1))=direction'*gradient;
            end
            if judgement
                if isempty(gradient)
                    [gradient,NFE_g]=getGradient(object_function,x,fval);NFE=NFE+NFE_g;
                end
                cubic_done=1;
                x_best=x;
                fval_best=fval;
                gradient_best=gradient;
                lamada_best=lamada;
            end
        else
            % interp_fvalp > interp_fval1
            flag=find(data_list(1,:)==interp_x1);
            interp_fval1=data_list(2,flag);
            gradient_x1=data_list(3,flag);
            if isnan(gradient_x1)
                [gradient_x1,NFE_g]=getGradient...
                    (object_function,x_initial+direction*interp_x1,interp_fval1);NFE=NFE+NFE_g;
            end
            
            if dir*gradient_x1 > 0
                interp_x2=interp_x1;
                interp_fval2=interp_fval1;
                interp_x1=interp_xp;
                interp_fval1=interp_fvalp;
            else
                % interp_xp > interp_x1
                interp_xp=(interp_x1+interp_x2)/2;
                interp_fvalp=object_function(x_initial+interp_xp*direction);NFE = NFE + 1;
                if isempty(find(data_list(1,:)==interp_xp))
                    flag=find(dir*data_list(1,:)>dir*interp_xp);
                    if isempty(flag)
                        flag=size(data_list,2)+1;
                    end
                    data_list=[data_list(:,1:(flag(1)-1)),[interp_xp;interp_fvalp;NaN],data_list(:,(flag(1)):end)];
                end
            end
        end
    end
    if dir*interp_x1 < dir*interp_xp
        if interp_fval1 < interp_fvalp
            interp_x2=interp_xp;
            interp_fval2=interp_fvalp;
        else
            lamada_old=lamada;
            fval_old=fval;
            gradient_old=gradient;
            lamada=interp_xp;
            fval=interp_fvalp;
            x=x_initial+lamada*direction;
            [judgement,NFE,gradient]=judgeQuit...
                (fval_initial,gradient_initial,object_function,x,lamada,fval,NFE,direction,c1,c2,...
                iteration,iteration_max,tolorance,lamada_old);
            if ~isempty(gradient)
                data_list(3,find(data_list(1,:)==interp_xp))=direction'*gradient;
            end
            if judgement
                if isempty(gradient)
                    [gradient,NFE_g]=getGradient(object_function,x,fval);NFE=NFE+NFE_g;
                end
                cubic_done=1;
                x_best=x;
                fval_best=fval;
                gradient_best=gradient;
                lamada_best=lamada;
            end
            
            flag=find(data_list(1,:)==interp_xp);
            interp_fvalp=data_list(2,flag);
            gradient_xp=data_list(3,flag);
            if isnan(gradient_xp)
                [gradient_xp,NFE_g]=getGradient...
                    (object_function,x_initial+direction*interp_xp,interp_fvalp);NFE=NFE+NFE_g;
            end
            
            if dir*gradient_xp > 0
                interp_x2=interp_xp;
                interp_fval2=interp_fvalp;
            else
                interp_x1=interp_xp;
                interp_fval1=interp_fvalp;
            end
        end
    end
    iteration=iteration+1;
    if iteration >= iteration_max
        cubic_done=1;
        x_best=x;
        fval_best=fval;
        if isempty(gradient) || isnan(gradient(1))
            [gradient,NFE_g]=getGradient(object_function,x,fval);NFE=NFE+NFE_g;
        end
        gradient_best=gradient;
        lamada_best=lamada;
    end
end

% if isempty(gradient) || isnan(gradient(1))
%    disp('stop');
% end

    function [lamada,coefficient_cubic]=minCubicInterpolate(interpolate_matrix,interpolate_value)
        
        coefficient_cubic=interpolate_matrix\interpolate_value;
        
        temp_sqrt=4*coefficient_cubic(2)^2-12*coefficient_cubic(1)*coefficient_cubic(3);
        if temp_sqrt>=0
            temp_lamada=-coefficient_cubic(2)/3/coefficient_cubic(1)+...
                sqrt(temp_sqrt)/6/coefficient_cubic(1);
            if (temp_lamada*6*coefficient_cubic(1)+2*coefficient_cubic(2))>0
                lamada=temp_lamada;
            else
                lamada=-coefficient_cubic(2)/3/coefficient_cubic(1)-...
                    sqrt(temp_sqrt)...
                    /6/coefficient_cubic(1);
            end
        else
            lamada=-coefficient_cubic(2)/3/coefficient_cubic(1);
        end
    end
    function [judgement,NFE,gradient]=judgeQuit...
            (fval_initial,gradient_initial,object_function,x,lamada,fval,NFE,direction,c1,c2,...
            iteration,iteration_max,tolorance,lamada_old)
        judgement=0;
        gradient=[];
        if (fval<=(fval_initial+c1*lamada*gradient_initial'*direction))
            [gradient,NFE_gj]=getGradient(object_function,x,fval);NFE=NFE+NFE_gj;
            if (abs(gradient'*direction)<=c2*abs(gradient_initial'*direction))
                % whether x satisfy volfe
                judgement=1;
            end
        end
        if abs(lamada-lamada_old)/lamada_old < tolorance^2
            judgement=1;
        end
        if iteration >= iteration_max
            judgement=1;
        end
        if abs(lamada) < 1e-6
            judgement=1;
        end
    end
end

function [gradient,NFE]=getGradient(object_function,x,fval,step)
% difference to find the gradient, forward difference
% default diff step is 1e-6
%
NFE=0;
if nargin < 4
    step=1e-6;
    if nargin < 3
        fval=object_function(x);
        NFE=1;
        if nargin < 2
            error('gradient:lack x initial');
        end
    end
end

fval_origin=fval;
gradient=zeros(size(x,1),size(x,2));
for x_index=1:size(x,1)
    x_forward=x;
    x_forward(x_index)=x_forward(x_index)+step;
    gradient(x_index,1)=(object_function(x_forward)-fval_origin)/step;NFE=NFE+1;
end
end