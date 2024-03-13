function [x_best,fval_best,NFE,output]=optimCubicInterp...
    (obj_fcn,x_init,low_bou,up_bou,obj_torl,iter_max)
% cubic interp optimization, should provide fval and gradient
% only work for one best(convex)
%
if nargin < 6
    iter_max=[];
    if nargin < 5
        obj_torl=[];
        if nargin < 4
            up_bou=[];
            if nargin < 3
                low_bou=[];
                if nargin < 2
                    error('lack x initial');
                end
            end
        end
    end
end

draw_range=0.001;
draw_interval=draw_range*0.02;
DRAW_FLAG=0;

if isempty(iter_max)
    iter_max=10*length(x_init);
end

if isempty(obj_torl)
    obj_torl=1e-6;
end

x=x_init;
done=0;
iter=0;
NFE=0;
result_x_list=[];
result_obj_list=[];

% decide which turn to search
[fval,gradient]=obj_fcn(x);NFE=NFE+1;
result_x_list=[result_x_list;x];
result_obj_list=[result_obj_list;fval];
if gradient < -obj_torl
    direction=1;
elseif gradient > obj_torl
    direction=-1;
else
    done=1;
    x_best=x;
    fval_best=fval;
end

x_old=x;
fval_old=fval;
gradient_old=gradient;
iter=iter+1;

% move forward to first point
if ~done
    x=x_old+direction*0.01;
    if x > up_bou
        x=up_bou;
    elseif x < low_bou
        x=low_bou;
    end
    [fval,gradient]=obj_fcn(x);NFE=NFE+1;
    result_x_list=[result_x_list;x];
    result_obj_list=[result_obj_list;fval];
    quit_flag=judgeQuit...
        (x,x_old,fval,fval_old,gradient,obj_torl,iter,iter_max);
    if quit_flag
        done=1;
        x_best=x;
        fval_best=fval;
    end
    iter=iter+1;
end

% main loop for cubic interp
while ~done

    x_base=x_old;
    x_relative=x/x_old;
    interp_matrix=[1,1,1,1;
        3,2,1,0;
        x_relative^3,x_relative^2,x_relative,1;
        3*x_relative^2,2*x_relative,1,0];

    if rcond(interp_matrix) < eps
        disp('error');
    end

    interp_value=[fval_old;gradient_old*x_base;fval;gradient*x_base];
    [x_inter_rel,coefficient_cubic]=minCubicInterp(interp_matrix,interp_value);
    x_inter=x_inter_rel*x_base;

    if DRAW_FLAG
        x_draw=1:direction*draw_interval:direction*draw_range;
        x_draw=x_draw/x_base;
        line(x_draw*x_base,coefficient_cubic(1)*x_draw.^3+coefficient_cubic(2)*x_draw.^2+...
            coefficient_cubic(3)*x_draw+coefficient_cubic(4));
    end

    % limit search space, process constraints
    if x_inter > up_bou
        x_inter=up_bou;
    elseif x_inter < low_bou
        x_inter=low_bou;
    end

    [fval_inter,gradient_inter]=obj_fcn(x_inter);NFE=NFE+1;

    % only work for one best(convex)
    % three situation discuss
    if gradient < 0
        x_old=x;
        fval_old=fval;
        gradient_old=gradient;
    else
        if gradient_inter < 0
            x_old=x;
            fval_old=fval;
            gradient_old=gradient;
        end
    end

    x=x_inter;
    fval=fval_inter;
    gradient=gradient_inter;

    quit_flag=judgeQuit...
        (x,x_old,fval,fval_old,gradient,obj_torl,iter,iter_max);
    if quit_flag
        done=1;
        x_best=x;
        fval_best=fval;
    end

    result_x_list=[result_x_list;x];
    result_obj_list=[result_obj_list;fval];
    iter=iter+1;
end
output.result_x_list=result_x_list;
output.result_obj_list=result_obj_list;

    function [lamada,coefficient_cubic]=minCubicInterp(interp_matrix,interp_value)
        % calculate min cubic curve
        %
        coefficient_cubic=interp_matrix\interp_value;

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

    function quit_flag=judgeQuit...
            (x,x_old,fval,fval_old,gradient,torlance,iteration,iteration_max)
        quit_flag=0;
        if abs(fval-fval_old)/fval_old < torlance
            quit_flag=1;
        end
        if abs(gradient) < torlance
            quit_flag=1;
        end
        if abs(x-x_old) < 1e-5
            quit_flag=1;
        end
        if iteration >= iteration_max
            quit_flag=1;
        end
    end
end
