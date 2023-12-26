function [x,fval,NFE,output]=optimalPenaltyMethod...
    (object_function,x0,A,b,Aeq,beq,lb,ub,nonlcon_function,...
    c,r1,r2)
% optimal funtion PenaltyMethod version 1
% x0,b,beq,lb,ub,c,ceq MUST BE column vector!
% this penalty method use own fminunc
%

% initialization of parameters
if nargin < 10
    c=1.421400430223882;
    r1=9.989069410863966e+02;
    r2=1;
    d=0.9;
end

iteration_max=100;
tolerance_max=1e-6;

% input check
if isempty(A)
    A=0;
end
if isempty(b)
    b=0;
end
if isempty(Aeq)
    Aeq=0;
end
if isempty(beq)
    beq=0;
end
if isempty(lb)
    lb=-inf;
end
if isempty(ub)
    ub=inf;
end

% record para
NFE=0;
iter_times=0;

% main
fval0=object_function(x0);
penalty_function=@(x) penaltyFunction(object_function,x,A,b,Aeq,beq,lb,ub,nonlcon_function,r1,r2);
[x1,fval1,NFE_QN]=optimalQuasiNewtonMethod(penalty_function,x0);NFE=NFE+NFE_QN;

while ~judgeQuit(x1,x0,fval1,fval0)&&iter_times<=iteration_max
    % did not meet exit condition to enter loop
    x0=x1;
    fval0=fval1;
    if judgeTolerance(x0,A,b,Aeq,beq,lb,ub,nonlcon_function,tolerance_max)
        r1=r1*d;
        r2=r2*d;
    else
        r1=c*r1;
        r2=c*r2;
    end
    penalty_function=@(x) penaltyFunction(object_function,x,A,b,Aeq,beq,lb,ub,nonlcon_function,r1,r2);
    [x1,fval1,NFE_QN]=optimalQuasiNewtonMethod(penalty_function,x0);NFE=NFE+NFE_QN;
    
    iter_times=iter_times+1;
end
x=x1;
fval=object_function(x);

output.iter_times=iter_times;
if iter_times>iteration_max
    exitflag=-1;
else
    exitflag=0;
end
output.exitflag=exitflag+1;
end

function fval=penaltyFunction(func,x,A,b,Aeq,beq,lb,ub,nonlcon,r1,r2)
% The two equality constraints are directly squared, and the rest are compared with 0;
%
line_con=max(A*x-b,0);
equa_con=Aeq*x-beq;
lbou_con=max(lb-x,0);
ubou_con=max(x-ub,0);
if isempty(nonlcon)
    c=0;ceq=0;
else
    [c,ceq]=nonlcon(x);
    if isempty(c)
        c=0;
    end
    if isempty(ceq)
        ceq=0;
    end
end
nonl_con=max(c,0);
none_con=ceq;

fval=func(x)+r1*(line_con'*line_con+lbou_con'*lbou_con+ubou_con'*ubou_con+nonl_con'*nonl_con)+...%linear constrain
    r2*(equa_con'*equa_con+none_con'*none_con);%eqaution constrain
end
function flag=judgeQuit(x1,x0,fval1,fval0)
epslion_x=1e-6;
epslion_f=1e-6;
delta_x=norm(x1-x0);
if fval0==0
    delta_f=abs((fval1-fval0));
else
    delta_f=abs((fval1-fval0)/norm(fval0));
end
if delta_x<=epslion_x
    if delta_f<=epslion_f
        flag=1;
    else
        flag=0;
    end
else
    flag=0;
end
end
function flag=judgeTolerance(x0,A,b,Aeq,beq,lb,ub,nonlcon,tolerance_max)
line_con=max(A*x0-b,0);
equa_con=Aeq*x0-beq;
lbou_con=max(lb-x0,0);
ubou_con=max(x0-ub,0);
if isempty(nonlcon)
    c=0;ceq=0;
else
    [c,ceq]=nonlcon(x0);
    if isempty(c)
        c=0;
    end
    if isempty(ceq)
        ceq=0;
    end
end
nonl_con=max(c,0);
none_con=ceq;
f=max([line_con;equa_con;lbou_con;ubou_con;none_con;nonl_con]);
if f<=tolerance_max
    flag=1;
else
    flag=0;
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
NFE=0;
done=0;
iteration=0;
information_flag=0; % whether show information
hess_update_flag=2;
if information_flag
    fprintf('%8s %8s  %12s    %8s\n','iteration','NFE','fval','lamada');
end
information_format='%6d    %8d %16.8g  %8.4g\n';

x_list=[x_initial'];
if hess_update_flag==1
    hessian_function=@(x_delta,gradient_delta,H) hessianDFPMethod(x_delta,gradient_delta,H);
elseif hess_update_flag==2
    hessian_function=@(x_delta,gradient_delta,H) hessianBFGSMethod(x_delta,gradient_delta,H);
end

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
    
    x_list=[x_list;x_new'];
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
    
    iteration=iteration+1;
end
output.x_list=x_list;
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
            disp('hessianDFPMethod: nan!')
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
% quadratic interpolation is used to get midpoint between out two point
% by three point's fval and gradient initial, cubic interpolation was used
% to search accurate step/lamada
% direction is search line, direction*gradient is one dimension differ
% do not move x_initial, fval_initial, gradient_initial
% just change lamada to move x change fval and gradient
% _old is record of last time
% search max range default is 1e6
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

draw_information=0; % whether draw information
draw_range=0.00001;
draw_interval=draw_range*0.02;

if draw_information
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
    if rcond(interp_matrix)<1e-16
        disp('stop');
    end
    coefficient_quad=interp_matrix\interp_value;
    
    if draw_information
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
    if dir*interp_x2 < dir*interp_x1
        disp('stop');
    end
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
    
    if draw_information
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

if isempty(gradient) || isnan(gradient(1))
    disp('stop');
end

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
