clc;
clear;
close all hidden;

object_function=@(x) objectFunction(x);
A=[1,2];
B=1;
Aeq=[2,1];
Beq=1;
low_bou=ones(2,1)*-2;
up_bou=ones(2,1)*2;
x_initial=[-1;2];

% compare
options = optimoptions('fmincon','Display','iter-detailed','Algorithm','sqp');
[x,fval,exitflag,output,grad,hessian]=fmincon(object_function,x_initial,A,B,Aeq,Beq,[],[],[],options);
disp(x);

% optimal
[x_best,fval_best,NFE,output]=...
    optimalSequentialQuadraticProgramming(object_function,x_initial,A,B,Aeq,Beq);
disp(['x_best:']);disp(x_best);
disp(['fval_best:',num2str(fval_best)]);
disp(['NFE:',num2str(NFE)]);
disp(['iteration:',num2str(output.iteration)]);
result_x_best=output.result_x_best;

% draw
draw_regin_x=[-3,3];
draw_regin_y=[-3,3];
draw_d=0.1;
[X_draw,Y_draw]=meshgrid(draw_regin_x(1):draw_d:draw_regin_x(2),draw_regin_y(1):draw_d:draw_regin_y(2));
Z=zeros(size(Y_draw,1),size(X_draw,2));
for x_index=1:size(X_draw,2)
    for y_index=1:size(Y_draw,1)
        Z(y_index,x_index)=object_function([X_draw(1,x_index);Y_draw(y_index,1)]);
    end
end
figure(1);
surf(X_draw,Y_draw,Z);
figure(2);
contour(X_draw,Y_draw,Z);
line(result_x_best(:,1),result_x_best(:,2));
text(result_x_best(:,1),result_x_best(:,2),num2str(linspace(1,size(result_x_best,1),size(result_x_best,1))'-1));

function [x_best,fval_best,NFE,output]=optimalSequentialQuadraticProgramming...
    (object_function,x_initial,A,B,Aeq,Beq,low_bou,up_bou,nonlcon_function,....
    NFE_max,iteration_max,torlance,nonlcon_torlance)
% optimal funtion Sequential Quadratic Programming version 0
% unimportant function had been include in main function
% lineSearch method use wolfe Guidelines to decrease NFE
% do not test robustness!!!
%
% Copyright 2022 10 Adel
%
if nargin < 11 || isempty(nonlcon_torlance)
    nonlcon_torlance=1e-3;
    if nargin < 10 || isempty(torlance)
        torlance=1e-3;
        if nargin < 9
            iteration_max=[];
            if nargin < 8
                NFE_max=[];
            end
        end
    end
end

if nargin < 9
    nonlcon_function=[];
    if nargin < 8
        up_bou=[];
        if nargin < 7
            low_bou=[];
            if nargin < 6
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

INFORMATION_FLAG=1; % whether show information
HESS_UPDATA_FLAG=2; % hessian matrix updata flag, 1 is DFP, 2 is BFGS
GRADIENT_FLAG=0; % whether use object function gradient
GRADIENT_NONLCON_FLAG=0; % whether use nonlcon function gradient

variable_number=size(x_initial,1);

if isempty(iteration_max)
iteration_max=10*variable_number;
end
if isempty(NFE_max)
    NFE_max=100*variable_number;
end

if INFORMATION_FLAG
    fprintf('%8s %8s  %12s %16s   %8s\n','iteration','NFE','fval','Feasibility','lamada');
end
information_format='%6d    %8d %16.8g %12.8g  %8.4g\n';

if HESS_UPDATA_FLAG==1
    hessian_function=@(x_delta,gradient_delta,H) hessianDFPMethod(x_delta,gradient_delta,H);
elseif HESS_UPDATA_FLAG==2
    hessian_function=@(x_delta,gradient_delta,H) hessianBFGSMethod(x_delta,gradient_delta,H);
end

NFE=0;done=0;iteration=0;
result_x_best=zeros(iteration_max,variable_number);
result_fval_best=zeros(iteration_max,1);

x=x_initial;
[fval,gradient,NFE_fg]=getFvalGradient...
    (object_function,x,[],torlance,...
    GRADIENT_FLAG);NFE=NFE+NFE_fg;

gradient_initial=gradient;

if norm(gradient,Inf)<=torlance
    done=1;
    x_best=x;
    fval_best=fval;
    gradient_best=gradient;
end

if INFORMATION_FLAG
    fprintf('%6d    %8d %16.8g\n',iteration,NFE,fval);
end

iteration=iteration+1;
H=eye(size(x_initial,1));
% main loop
%
while ~done
    % Quadratic Programming to obtain line search direction
    [direction,lamda,lamdaeq]=programQuadratic(x,gradient,H,...
        A,B,Aeq,Beq,low_bou,up_bou,nonlcon_function,nonlcon_torlance);
    
    lamada=1;
    %     if iteration==1
    %         lamada=min(1/norm(gradient,Inf),1);
    %     end
    
    % line search best step
    [x_new,fval_new,gradient_new,lamada_best,NFC_line]=lineSearch...
        (object_function,direction,x_initial,fval_initial,gradient_initial,lamada,...
        A,B,Aeq,Beq,low_bou,up_bou,nonlcon_function,...
        GRADIENT_FLAG,GRADIENT_NONLCON_FLAG);
    NFE=NFE+NFC_line;
    
    if judgeQuit(torlance,x,x_new,gradient_initial,...
            gradient_new,iteration,iteration_max)
        done=1;
        x_best=x;
        fval_best=fval;
        gradient_best=gradient;
    end
    
    x_delta=x_new-x;
    gradient_delta=gradient_new-gradient;
    %     if iteration==1
    %         H=x_delta'*gradient_delta/(gradient_delta'*gradient_delta)*H;
    %     end
    H=hessian_function(x_delta,gradient_delta,H);
    
    x=x_new;
    fval=fval_new;
    gradient=gradient_new;
    
    if INFORMATION_FLAG
        fprintf(information_format,iteration,NFE,fval,abs(Aeq*x-Beq),lamada_best);
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
            (torlance,x,x_new,gradient_initial,gradient_new,iteration,iteration_max)
        judgement=0;
        if (norm((x-x_new)./(1+abs(x)),2)<=torlance)
            judgement=1;
        end
        if (norm(gradient_new,2)<=torlance*(norm(gradient_initial,2)+1))
            judgement=1;
        end
        if iteration>iteration_max
            judgement=1;
        end
    end
    function H=hessianDFPMethod(x_delta,g_delta,H)
        % updata matrix H DFP method
        x_delta_g_delta=(x_delta'*g_delta);
        g_delta_H_g_delta=(g_delta'*H*g_delta);
        if x_delta_g_delta ~= 0 && ~isnan(x_delta_g_delta) &&...
                g_delta_H_g_delta ~= 0 && ~isnan(g_delta_H_g_delta)
            A__=x_delta*x_delta'/x_delta_g_delta;
            B__=H*g_delta*(H*g_delta)'/g_delta_H_g_delta;
        else
            A__=0;B__=0;
        end
        H=H+A__-B__;
    end
    function H=hessianBFGSMethod(x_delta,g_delta,H)
        % updata matrix H BFGS method
        x_delta_g_delta=(x_delta'*g_delta);
        if x_delta_g_delta ~= 0 && ~isnan(x_delta_g_delta)
            A__=(1+g_delta'*H*g_delta/x_delta_g_delta)*x_delta*x_delta'/x_delta_g_delta;
            B__=(x_delta*(H*g_delta)'+H*g_delta*x_delta')/x_delta_g_delta;
        else
            A__=0;B__=0;
        end
        if isnan(A__(1,1)) || isnan(B__(1,1))
            A__(find(isnan(A__)))=0;
            %             disp('hessianDFPMethod: nan!')
        end
        H=H+A__-B__;
    end
end
function [direction,lamda_lin,lamdaeq_lin,lamda,lamdaeq]=programQuadratic(x,gradient,H,...
    A,B,Aeq,Beq,low_bou,up_bou,nonlcon_function,...
    con,coneq,gradient_con,gradient_coneq,nonlcon_torlance)
% solve quadratic programming problem to obtian direction
%
variable_number=size(H,1);
inequal_lin_number=0;
equal_lin_number=0;
inequal_number=0;
equal_number=0;

% process lin inequal constraints
A_lin_matrix=[];
B_lin_colume=[];
if ~isempty(A)
    active_con_lin_index=[]; % active constraint equation index
    if isempty(B)
        B=zeros(size(A,1),1);
    end
    con_lin=A*x-B;
    for con_lin_index=1:size(con_lin,1)
       if con_lin(con_lin_index) <= nonlcon_torlance
           active_con_lin_index=[active_con_lin_index;con_lin_index];
           inequal_lin_number=inequal_lin_number+1;
       end
    end
    A_lin_matrix=A(active_con_lin_index,:);
    B_lin_colume=B(active_con_lin_index,1);
end

% process lin equal constraints
Aeq_lin_matrix=[];
Beq_lin_colume=[];
if ~isempty(Aeq)
    if isempty(Beq)
        Beq=zeros(size(Aeq,1),1);
    end
    Aeq_lin_matrix=Aeq;
    Beq_lin_colume=Beq;
    equal_lin_number=size(Beq,1);
end

% process nonl inequal constraints
A_matrix=[];
B_colume=[];
if ~isempty(con)
    active_con_index=[]; % active not linear constraint equation index
    for con_index=1:size(con,1)
       if con(con_index) <= nonlcon_torlance
           active_con_index=[active_con_index;con_index];
           inequal_number=inequal_number+1;
       end
    end
    A_matrix=gradient_con(active_con_lin_index,:);
    B_colume=con(active_con_lin_index,1);
end

% process nonl equal constraints
Aeq_matrix=[];
Beq_colume=[];
if ~isempty(coneq)
    Aeq_matrix=gradient_coneq;
    Beq_colume=coneq;
    equal_number=size(con_eq,1);
end

% merge into matrix
if equal_lin_number > 0 && inequal_lin_number > 0
    matrix=[
        H,A_lin_matrix',Aeq_lin_matrix';
        A_lin_matrix,zeros(inequal_lin_number),zeros(inequal_lin_number,equal_lin_number);
        Aeq_lin_matrix,zeros(equal_lin_number,inequal_lin_number),zeros(equal_lin_number);
        ];
    colume=[
        -gradient;
        -B_lin_colume;
        -Beq_lin_colume;
        ];
elseif equal_lin_number == 0 && inequal_lin_number > 0
    matrix=[
        H,A_lin_matrix';
        A_lin_matrix,zeros(inequal_lin_number);
        ];
    colume=[
        -gradient;
        -B_lin_colume;
        ];
elseif equal_lin_number > 0 && inequal_lin_number == 0
    matrix=[
        H,Aeq_lin_matrix';
        Aeq_lin_matrix,zeros(equal_lin_number);
        ];
    colume=[
        -gradient;
        -Beq_lin_colume;
        ];
else
    matrix=H;
    colume=-gradient;
end

% process solve answer
solution=matrix\colume;
direction=solution(1:variable_number,1);
lamda_lin=solution(variable_number+1:variable_number+inequal_lin_number,1);
lamdaeq_lin=solution(variable_number+inequal_lin_number+1:variable_number+inequal_lin_number+equal_lin_number,1);
end
function [x_best,fval_best,gradient_best,lamada_best,NFE]=lineSearch...
    (object_function,direction,x_initial,fval_initial,gradient_initial,lamada,...
    A,B,Aeq,Beq,low_bou,up_bou,nonlcon_function,...
    GRADIENT_FLAG,GRADIENT_NONLCON_FLAG)
% quadratic/cubic interpolation method version 13
% add gradient and constraints support
% quadratic/cubic interpolation optimization was used
% direction is search line, direction*gradient is one dimension differ
% do not move x_initial, fval_initial, gradient_initial
% just change lamada to move x change fval and gradient
% _old is record of last time
% search range default is 1e-6 to 1e6
% each time's fval should be recond if it's a rand search
% gradient only can be variable_number x 1 matrix or nan, can not be []
%
% Copyright 2022 10 Adel
%
cubic_done=0;
quad_done=0;
iteration_max=length(x_initial)+10;
c1=0.01;
c2=0.9;
NFE=0;
tolorance=1e-1;
max_change=1e-1;
serach_range=[1e-9,1e9];

dir=1; % search direction, means forward or backward
if direction'*gradient_initial > 0
    lamada=-lamada;
    dir=-1;
end

draw_range=0.00001;
draw_interval=draw_range*0.02;
DRAW_FLAG=0;
if DRAW_FLAG
    close all hidden;
    figure(1);
    for draw_lamada=0:dir*draw_interval:dir*draw_range
        line(draw_lamada,object_function(x_initial+draw_lamada*direction),'Marker','o');
    end
    %     axis([-dra+w_range,draw_range,-5,20]);
end

% if isempty(Aeq)
    x=x_initial+lamada*direction;
    if ~GRADIENT_FLAG
        fval=object_function(x);NFE=NFE+1;
        gradient=NaN;
    else
        [fval,gradient]=object_function(x);NFE=NFE+1;
    end
    
    data_list=[0,fval_initial,gradient_initial';
        lamada,fval,nan(1,length(gradient_initial))];
    
    if (fval<=(fval_initial+c1*lamada*gradient_initial'*direction))
        if isnan(gradient(1))
            [~,gradient,NFE_fg]=getFvalGradient...
                (object_function,x,fval,[],GRADIENT_FLAG);NFE=NFE+NFE_fg;
        end
        data_list(2,3:end)=gradient';
        if (abs(gradient'*direction)<=c2*abs(gradient_initial'*direction)) % whether x satisfy torlance/volfe
            quad_done=1;
            cubic_done=1;
            x_best=x;
            fval_best=fval;
            gradient_best=gradient;
            lamada_best=lamada;
        end
    end
    
% else
%     lamada=(Beq-Aeq*x_initial)/(Aeq*direction);
%     x=x_initial+lamada*direction;
%     [fval,gradient,NFE_fg]=getFvalGradient...
%                 (object_function,x,[],[],GRADIENT_FLAG);NFE=NFE+NFE_fg;
%     
%     quad_done=1;
%     cubic_done=1;
%     x_best=x;
%     fval_best=fval;
%     gradient_best=gradient;
%     lamada_best=lamada;
% end

iteration=1;

interp_x0=0;
interp_fval0=fval_initial;

% quadratic interpolation optimal
% quadratic interpolation is to find another point for cubic interpolation
% so out point fval should large than mid point
% 0 < 1 < 2, f(1) < f(2), f(1) < f(0)
interp_x1=lamada;
interp_fval1=fval;
interp_gradient1=gradient;
while ~quad_done
    if interp_x1 > serach_range(2)
        error('lineSearch: can not find min in search range');
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
    
    if DRAW_FLAG
        x_draw=0:dir*draw_interval:dir*draw_range;
        x_draw=x_draw/interp_base;
        line(x_draw*interp_base,coefficient_quad(1)*x_draw.^2+coefficient_quad(2)*x_draw+coefficient_quad(3));
    end
    
    if coefficient_quad(1) <= 0
        % this mean function still decline, need to add more
        interp_x1=interp_x1*2;
        if ~GRADIENT_FLAG
            interp_fval1=object_function(x_initial+interp_x1*direction);NFE=NFE+1;
            interp_gradient1=NaN;
        else
            [interp_fval1,interp_gradient1]=object_function(x_initial+interp_x1*direction);NFE=NFE+1;
        end
        
        % updata interp_x1 into data_list
        data_list=dataListUpdata(data_list,interp_x1,interp_fval1,interp_gradient1',dir);
    else
        % this mean function have min fval
        interp_xp_rel=-coefficient_quad(2)/coefficient_quad(1)/2;
        interp_xp=interp_base*interp_xp_rel;
        % limit the max change
        if abs(interp_xp/interp_x1) < max_change
            interp_xp=interp_x1/2;
        elseif abs(interp_x1/interp_xp) < max_change
            interp_xp=interp_x1*2;
        end
        if ~GRADIENT_FLAG
            interp_fvalp=object_function(x_initial+interp_xp*direction);NFE=NFE+1;
            interp_gradientp=NaN;
        else
            [interp_fvalp,interp_gradientp]=object_function(x_initial+interp_xp*direction);NFE=NFE+1;
        end
        
        % updata interp_xp into data_list
        data_list=dataListUpdata(data_list,interp_xp,interp_fvalp,interp_gradientp',dir);
        
        % there are five situations to discuss
        % interp_xp is new point to discuss
        if dir*interp_xp < dir*interp_x1
            
            if interp_fvalp < interp_fval0
                % go to cubic interpolate
                quad_done = 1;
                
                interp_x2=interp_x1;
                interp_fval2=interp_fval1;
                interp_gradient2=interp_gradient1;
                
                interp_x1=interp_xp;
                interp_fval1=interp_fvalp;
                interp_gradient1=interp_gradientp;
                
                lamada_old=lamada;
                fval_old=fval;
                gradient_old=gradient;
                
                % get into quit judgement
                lamada=interp_x1;
                fval=interp_fval1;
                gradient=interp_gradient1;
                x=x_initial+lamada*direction;
                
                % quit judgement, gradient maybe nan out
                [quit_flag,NFE,gradient]=judgeQuit...
                    (fval_initial,gradient_initial,object_function,x,lamada,fval,gradient,...
                    NFE,direction,c1,c2,...
                    iteration,iteration_max,tolorance,lamada_old,...
                    GRADIENT_FLAG);
                
                if quit_flag
                    if isnan(gradient(1))
                        [~,gradient,NFE_fg]=getFvalGradient...
                            (object_function,x,fval,[],GRADIENT_FLAG);NFE=NFE+NFE_fg;
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
                interp_gradient1=interp_gradientp;
            end
        else
            while interp_fval1 > interp_fvalp
                % interp_fvalp is low than interp_fval1
                % this mean fval can decrease if move forward
                
                %                 if interp_x1>serach_range(2)
                %                     error('lineSearch: can not find min in search range');
                %                 end
                
                interp_x1=interp_xp;
                interp_fval1=interp_fvalp;
                interp_gradient1=interp_gradientp;
                
                lamada_old=lamada;
                fval_old=fval;
                gradient_old=gradient;
                
                % get into quit judgement
                lamada=interp_x1;
                fval=interp_fval1;
                gradient=interp_gradient1;
                x=x_initial+lamada*direction;
                
                % quit judgement, gradient maybe nan out
                [quit_flag,NFE,gradient]=judgeQuit...
                    (fval_initial,gradient_initial,object_function,x,lamada,fval,gradient,...
                    NFE,direction,c1,c2,...
                    iteration,iteration_max,tolorance,lamada_old,...
                    GRADIENT_FLAG);
                
                if quit_flag
                    if isnan(gradient(1))
                        [~,gradient,NFE_fg]=getFvalGradient...
                            (object_function,x,fval,[],GRADIENT_FLAG);NFE=NFE+NFE_fg;
                    end
                    cubic_done=1;
                    x_best=x;
                    fval_best=fval;
                    gradient_best=gradient;
                    lamada_best=lamada;
                end
                
                % move forward interp_xp
                interp_xp=2*interp_xp;
                if ~GRADIENT_FLAG
                    interp_fvalp=object_function(x_initial+interp_xp*direction);NFE=NFE+1;
                    interp_gradientp=NaN;
                else
                    [interp_fvalp,interp_gradientp]=object_function(x_initial+interp_xp*direction);NFE=NFE+1;
                end
                
                % updata interp_xp into data_list
                data_list=dataListUpdata(data_list,interp_xp,interp_fvalp,interp_gradientp',dir);
            end
            
            % go to cubic interpolate
            quad_done = 1;
            
            interp_x2=interp_xp;
            interp_fval2=interp_fvalp;
            interp_gradient2=interp_gradientp;
        end
    end
    
    iteration=iteration+1;
    
    % quit check
    if iteration >= iteration_max
        quad_done=1;
        cubic_done=1;
        x_best=x;
        fval_best=fval;
        if isnan(gradient(1))
            [~,gradient,NFE_fg]=getFvalGradient...
                (object_function,x,fval,[],GRADIENT_FLAG);NFE=NFE+NFE_fg;
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
    if isnan(interp_x1) || isnan(interp_x2)
        disp('nan!')
    end
    
    if abs(interp_x1/interp_x2-1) < 0.001
        interp_x1=interp_x1/2;
        
        if ~GRADIENT_FLAG
            interp_fval1=object_function(x_initial+interp_x1*direction);NFE=NFE+1;
            interp_gradient1=NaN;
        else
            [interp_fval1,interp_gradient1]=object_function(x_initial+interp_x1*direction);NFE=NFE+1;
        end
        
        % updata interp_xp into data_list
        data_list=dataListUpdata(data_list,interp_x1,interp_fval1,interp_gradient1',dir);
    end
    
    % cube interpolation
    interp_base=interp_x2;
    interp_x1_relative=interp_x1/interp_x2;
    interp_matrix=[0,0,0,1;
        0,0,1,0;
        interp_x1_relative^3,interp_x1_relative^2,interp_x1_relative,1;
        1,1,1,1;];
    
    interp_value=[fval_initial;direction'*gradient_initial*interp_base;interp_fval1;interp_fval2];
    [interp_p_rel,coefficient_cubic]=minCubicInterpolate(interp_matrix,interp_value);
    interp_xp=interp_p_rel*interp_base;
    
    if DRAW_FLAG
        x_draw=0:dir*draw_interval:dir*draw_range;
        x_draw=x_draw/interp_base;
        line(x_draw*interp_base,coefficient_cubic(1)*x_draw.^3+coefficient_cubic(2)*x_draw.^2+...
            coefficient_cubic(3)*x_draw+coefficient_cubic(4));
    end
    
    % limit the max change
    if abs(interp_xp/interp_x1) < max_change
        interp_xp=dir*interp_x1/2;
    elseif abs(interp_x1/interp_xp) < max_change
        interp_xp=dir*interp_x1*2;
    elseif abs(interp_xp-interp_x1)/interp_x1 < max_change
        interp_xp=interp_x2/2;
    end
    
    % evaluate and updata interp_xp into data_list
    if ~GRADIENT_FLAG
        interp_fvalp=object_function(x_initial+interp_xp*direction);NFE=NFE+1;
        interp_gradientp=NaN;
    else
        [interp_fvalp,interp_gradientp]=object_function(x_initial+interp_xp*direction);NFE=NFE+1;
    end
    data_list=dataListUpdata(data_list,interp_xp,interp_fvalp,interp_gradientp',dir);
    
    % there are six situations to discuss
    if dir*interp_xp ==  dir*interp_x1
        interp_xp=(interp_x1+interp_x2)/2; % interp_xp > interp_x1
        
        % evaluate and updata interp_xp into data_list
        if ~GRADIENT_FLAG
            interp_fvalp=object_function(x_initial+interp_xp*direction);NFE=NFE+1;
            interp_gradientp=NaN;
        else
            [interp_fvalp,interp_gradientp]=object_function(x_initial+interp_xp*direction);NFE=NFE+1;
        end
        data_list=dataListUpdata(data_list,interp_xp,interp_fvalp,interp_gradientp',dir);
    end
    if dir*interp_xp <  dir*interp_x1
        if interp_fvalp < interp_fval1
            interp_x2=interp_x1;
            interp_fval2=interp_fval1;
            interp_gradient2=interp_gradient1;
            
            interp_x1=interp_xp;
            interp_fval1=interp_fvalp;
            interp_gradient1=interp_gradientp;
            
            lamada_old=lamada;
            fval_old=fval;
            gradient_old=gradient;
            
            % get into quit judgement
            lamada=interp_x1;
            fval=interp_fval1;
            gradient=interp_gradient1;
            x=x_initial+lamada*direction;
            
            % quit judgement, gradient maybe nan out
            [quit_flag,NFE,gradient]=judgeQuit...
                (fval_initial,gradient_initial,object_function,x,lamada,fval,gradient,...
                NFE,direction,c1,c2,...
                iteration,iteration_max,tolorance,lamada_old,...
                GRADIENT_FLAG);
            
            if quit_flag
                if isnan(gradient(1))
                    [~,gradient,NFE_fg]=getFvalGradient...
                        (object_function,x,fval,[],GRADIENT_FLAG);NFE=NFE+NFE_fg;
                end
                cubic_done=1;
                x_best=x;
                fval_best=fval;
                gradient_best=gradient;
                lamada_best=lamada;
            end
        else
            % interp_fvalp > interp_fval1
            [~,interp_fval1,gradient_x1]=dataListLoad(data_list,interp_x1);
            
            if isnan(gradient_x1)
                [~,gradient_x1,NFE_fg]=getFvalGradient...
                    (object_function,x_initial+direction*interp_x1,interp_fval1,[],GRADIENT_FLAG);NFE=NFE+NFE_fg;
            end
            
            if dir*gradient_x1 > 0
                interp_x2=interp_x1;
                interp_fval2=interp_fval1;
                interp_gradient2=interp_gradient1;
                
                interp_x1=interp_xp;
                interp_fval1=interp_fvalp;
                interp_gradient1=interp_gradientp;
            else
                % interp_xp > interp_x1
                interp_xp=(interp_x1+interp_x2)/2;
                
                % evaluate and updata interp_xp into data_list
                if ~GRADIENT_FLAG
                    interp_fvalp=object_function(x_initial+interp_xp*direction);NFE=NFE+1;
                    interp_gradientp=NaN;
                else
                    [interp_fvalp,interp_gradientp]=object_function(x_initial+interp_xp*direction);NFE=NFE+1;
                end
                data_list=dataListUpdata(data_list,interp_xp,interp_fvalp,interp_gradientp',dir);
            end
        end
    end
    if dir*interp_x1 < dir*interp_xp
        if interp_fval1 < interp_fvalp
            interp_x2=interp_xp;
            interp_fval2=interp_fvalp;
            interp_gradient2=interp_gradientp;
        else
            lamada_old=lamada;
            fval_old=fval;
            gradient_old=gradient;
            
            lamada=interp_xp;
            fval=interp_fvalp;
            gradient=interp_gradientp;
            x=x_initial+lamada*direction;
            
            % quit judgement, gradient maybe nan out
            [quit_flag,NFE,gradient]=judgeQuit...
                (fval_initial,gradient_initial,object_function,x,lamada,fval,gradient,...
                NFE,direction,c1,c2,...
                iteration,iteration_max,tolorance,lamada_old,...
                GRADIENT_FLAG);
            
            if quit_flag
                if isnan(gradient(1))
                    [~,gradient,NFE_fg]=getFvalGradient...
                        (object_function,x,fval,[],GRADIENT_FLAG);NFE=NFE+NFE_fg;
                end
                cubic_done=1;
                x_best=x;
                fval_best=fval;
                gradient_best=gradient;
                lamada_best=lamada;
            end
            
            [~,interp_fvalp,gradient_xp]=dataListLoad(data_list,interp_xp);
            if isnan(gradient_xp)
                [~,gradient_xp,NFE_fg]=getFvalGradient...
                    (object_function,x_initial+direction*interp_xp,interp_fvalp,[],GRADIENT_FLAG);NFE=NFE+NFE_fg;
            end
            
            if dir*gradient_xp > 0
                interp_x2=interp_xp;
                interp_fval2=interp_fvalp;
                interp_gradient2=interp_gradientp;
            else
                interp_x1=interp_xp;
                interp_fval1=interp_fvalp;
                interp_gradient1=interp_gradientp;
            end
        end
    end
    iteration=iteration+1;
    if iteration >= iteration_max
        cubic_done=1;
        x_best=x;
        fval_best=fval;
        if isnan(gradient(1))
            [~,gradient,NFE_fg]=getFvalGradient...
                    (object_function,x,fval,[],GRADIENT_FLAG);NFE=NFE+NFE_fg;
        end
        gradient_best=gradient;
        lamada_best=lamada;
    end
end

% if isempty(gradient) || isnan(gradient(1))
%    disp('stop');
% end

    function [lamada,coefficient_cubic]=minCubicInterpolate(interpolate_matrix,interpolate_value)
        % calculate min cubic curve
        %
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
    function data_list=dataListUpdata(data_list,interpx,fval,gradient,dir)
        % updata gradient into data_list
        % data_list is [x_list,fval_list,gradient_list]
        % x_number x 2+variable_number matrix
        %
        if isempty(find(data_list(:,1)==interpx))
            index=find(dir*data_list(:,1)>dir*interpx);
            if isempty(index)
                index=size(data_list,1)+1;
            end
            if isnan(gradient(1))
                gradient=nan(1,size(data_list,2)-2);
            end
            data_list=[data_list(1:(index(1)-1),:);[interpx,fval,gradient];data_list((index(1)):end,:)];
        end
    end
    function [index,fval,gradient]=dataListLoad(data_list,interpx)
        % load gradient from data_list
        % data_list is [interpx_list,fval_list,dirgrad_list]
        % x_number x 2+variable_number matrix
        %
        index=find((data_list(:,1)==interpx));
        fval=data_list(index,2);
        gradient=data_list(index,3:end);
    end
    function [judgement,NFE,gradient]=judgeQuit...
            (fval_initial,gradient_initial,object_function,x,lamada,fval,gradient,...
            NFE,direction,c1,c2,...
            iteration,iteration_max,tolorance,lamada_old,...
            GRADIENT_FLAG)
        % judge quit function
        % main function realize wolfe guidelines
        %
        judgement=0;
        if (fval<=(fval_initial+c1*lamada*gradient_initial'*direction))
            if isnan(gradient(1))
                if ~GRADIENT_FLAG
                    [~,gradient,NFE_fgjq]=getFvalGradient...
                        (object_function,x,fval,[],GRADIENT_FLAG);NFE=NFE+NFE_fgjq;
                else
                    [~,gradient]=object_function(x);NFE=NFE+1;
                end
            end
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
function [fval,gradient,NFE]=getFvalGradient...
    (object_function,x,fval,step,...
    GRADIENT_FLAG)
% difference to find the gradient, forward difference
% default diff step is 1e-6
% if object function provide gradient, use object function gradient
%
NFE=0;
if nargin < 4
    step=[];
    if nargin < 3
        fval=[];
        if nargin < 2
            error('gradient:lack x initial');
        end
    end
end

if isempty(fval)
    fval=object_function(x);NFE=NFE+1;
end

if isempty(step)
    step=1e-6;
end

if ~GRADIENT_FLAG
    fval_origin=fval;
    gradient=zeros(size(x,1),size(x,2));
    for x_index=1:size(x,1)
        x_forward=x;
        x_forward(x_index)=x_forward(x_index)+step;
        gradient(x_index,1)=(object_function(x_forward)-fval_origin)/step;NFE=NFE+1;
    end
else
    [~,gradient]=object_function(x);
    NFE=NFE+1;
end
end

function [fval]=objectFunction(x)
% objectFunction
%
% fval=0.5*x'*[5,1;1,5]*x+[5,2]*x+3;

fval=100*(x(2)-x(1)^2)^2 + (1-x(1))^2;

% if nargout > 1
%     gradient=[5,1;1,5]*x+[5;2];
% end
end