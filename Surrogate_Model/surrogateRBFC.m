clc;
clear;
close all hidden;

load('PK.mat');
[predict_function,radialbasis_model]=interpRadialBasisPreModel(X,Y);
figure_handle=figure(1);
interpVisualize(radialbasis_model,low_bou,up_bou,figure_handle)

function [predict_function,radialbasis_model]=interpRadialBasisPreModel...
    (X,Y)
% radial basis function interp pre model function
% input initial data X, Y, which are real data
% X, Y are x_number x variable_number matrix
% aver_X,stdD_X is 1 x x_number matrix
% reference [1] to optimal hyper parameter of radial basis function
% optimal function is cubic interpolation optimization
% basis function is gaussian function
%
% reference: [1] RIPPA S. An algorithm for selecting a good value for the
% parameter c in radial basis function interpolation [J]. Advances in
% Computational Mathematics, 1999, 11(2-3): 193-210.
%
% Copyright 2023 Adel
%
if nargin < 3
    basis_function=[];
end

[x_number,variable_number]=size(X);

% normalize data
aver_X=mean(X);
stdD_X=std(X);
aver_Y=mean(Y);
stdD_Y=std(Y);
index__=find(stdD_X == 0);
if ~isempty(index__),stdD_X(index__)=1;end
index__=find(stdD_Y == 0);
if ~isempty(index__),stdD_Y(index__)=1;end
X_nomlz=(X-aver_X)./stdD_X;
Y_nomlz=(Y-aver_Y)./stdD_Y;

% initialization distance of all X
X_dis=zeros(x_number,x_number);
for variable_index=1:variable_number
    X_dis=X_dis+(X_nomlz(:,variable_index)-X_nomlz(:,variable_index)').^2;
end
X_dis=sqrt(X_dis);

c=(prod(max(X_nomlz)-min(Y_nomlz))/x_number)^(1/variable_number);

% triple kernal function
basis_function_gauss=@(r,c) exp(-c*r.^2);
dRM_dc_function=@(x_number,X_dis,rdibas_matrix,c) -X_dis.^2.*rdibas_matrix;
object_function=@(c) objectFunctionRadiabasis....
    (X_dis,Y_nomlz,x_number,basis_function_gauss,c,dRM_dc_function);
[c,~,NFE,~]=optimalCubicInterp...
    (object_function,c,1e-2,1e2,1e-3);
basis_function=@(r) basis_function_gauss(r,c);

[beta,rdibas_matrix]=interpRadialBasis...
    (X_dis,Y_nomlz,basis_function,x_number);

% initialization predict function
predict_function=@(X_predict) interpRadialBasisPredictor...
    (X_predict,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_number,variable_number,beta,basis_function);

radialbasis_model.X=X;
radialbasis_model.Y=Y;
radialbasis_model.radialbasis_matrix=rdibas_matrix;
radialbasis_model.beta=beta;

radialbasis_model.aver_X=aver_X;
radialbasis_model.stdD_X=stdD_X;
radialbasis_model.aver_Y=aver_Y;
radialbasis_model.stdD_Y=stdD_Y;
radialbasis_model.basis_function=basis_function;

radialbasis_model.predict_function=predict_function;

% abbreviation:
% num: number, pred: predict, vari: variable
    function [fval,gradient]=objectFunctionRadiabasis....
            (X_dis,Y,x_number,basis_function,c,dRM_dc_function)
        % MSE_CV function, simple approximation to RMS
        % basis_function input is c and x_sq
        %
        basis_function=@(r) basis_function(r,c);
        [beta__,rdibas_matrix__,inv_rdibas_matrix__]=interpRadialBasis...
            (X_dis,Y,basis_function,x_number);
        U=beta__./diag(inv_rdibas_matrix__);
        fval=sum(U.^2);
        
        % calculate gradient
        if nargout > 1
            inv_rdibas_matrix_gradient=-inv_rdibas_matrix__*...
                dRM_dc_function...
                (x_number,X_dis,rdibas_matrix__,c)*inv_rdibas_matrix__;
            U_gradient=zeros(x_number,1);
            I=eye(x_number);
            for x_index=1:x_number
                U_gradient(x_index)=(I(x_index,:)*inv_rdibas_matrix_gradient*Y)/...
                    inv_rdibas_matrix__(x_index,x_index)-...
                    beta__(x_index)*(I(x_index,:)*inv_rdibas_matrix_gradient*I(:,x_index))/...
                    inv_rdibas_matrix__(x_index,x_index)^2;
            end
            
            gradient=2*sum(U.*U_gradient);
        end
    end

    function [beta,rdibas_matrix,inv_rdibas_matrix]=interpRadialBasis...
            (X_dis,Y,basis_function,x_number)
        % interp polynomial responed surface core function
        % calculation beta
        %
        % Copyright 2022 Adel
        %
        rdibas_matrix=basis_function(X_dis);
        
        % stabilize matrix
        rdibas_matrix=rdibas_matrix+eye(x_number)*1e-6;
        
        % solve beta
        inv_rdibas_matrix=rdibas_matrix\eye(x_number);
        beta=inv_rdibas_matrix*Y;
    end

    function [x_best,favl_best,NFE,output]=optimalCubicInterp...
            (object_function,x_initial,low_bou,up_bou,torlance,iteration_max)
        % cubic interp optimization, should provide fval and gradient
        % only work for one best(convex)
        %
        if nargin < 6
            iteration_max=[];
            if nargin < 5
                torlance=[];
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
        
        INFORMATION_FLAG=0; % whether show information
        
        draw_range=0.001;
        draw_interval=draw_range*0.02;
        DRAW_FLAG=0;
        
        if isempty(iteration_max)
            iteration_max=10*length(x_initial);
        end
        
        if isempty(torlance)
            torlance=1e-6;
        end
        
        x=x_initial;
        done=0;
        iteration=0;
        NFE=0;
        result_x_list=[];
        result_fval_list=[];
        
        % decide which turn to search
        [fval,gradient]=object_function(x);NFE=NFE+1;
        result_x_list=[result_x_list;x];
        result_fval_list=[result_fval_list;fval];
        if gradient < -torlance
            direction=1;
        elseif gradient > torlance
            direction=-1;
        else
            done=1;
            x_best=x;
            favl_best=fval;
        end
        
        x_old=x;
        fval_old=fval;
        gradient_old=gradient;
        iteration=iteration+1;
        
        % move forward to first point
        if ~done
            x=x_old+direction*0.01;
            if x > up_bou
                x=up_bou;
            elseif x < low_bou
                x=low_bou;
            end
            [fval,gradient]=object_function(x);NFE=NFE+1;
            result_x_list=[result_x_list;x];
            result_fval_list=[result_fval_list;fval];
            quit_flag=judgeQuit...
                (x,x_old,fval,fval_old,gradient,torlance,iteration,iteration_max);
            if quit_flag
                done=1;
                x_best=x;
                favl_best=fval;
            end
            iteration=iteration+1;
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
            [x_inter_rel,coefficient_cubic]=minCubicInterpolate(interp_matrix,interp_value);
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
            
            [fval_inter,gradient_inter]=object_function(x_inter);NFE=NFE+1;
            
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
                (x,x_old,fval,fval_old,gradient,torlance,iteration,iteration_max);
            if quit_flag
                done=1;
                x_best=x;
                favl_best=fval;
            end
            
            result_x_list=[result_x_list;x];
            result_fval_list=[result_fval_list;fval];
            iteration=iteration+1;
        end
        output.result_x_list=result_x_list;
        output.result_fval_list=result_fval_list;
        
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

    function [Y_pred]=interpRadialBasisPredictor...
            (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,beta,basis_function)
        % radial basis function interpolation predict function
        %
        [x_pred_num,~]=size(X_pred);

        % normalize data
        X_pred_nomlz=(X_pred-aver_X)./stdD_X;
        
        % calculate distance
        X_dis_pred=zeros(x_pred_num,x_num);
        for vari_index=1:vari_num
            X_dis_pred=X_dis_pred+...
                (X_pred_nomlz(:,vari_index)-X_nomlz(:,vari_index)').^2;
        end
        X_dis_pred=sqrt(X_dis_pred);
        
        % predict variance
        Y_pred=basis_function(X_dis_pred)*beta;
        
        % normalize data
        Y_pred=Y_pred*stdD_Y+aver_Y;
    end

end
