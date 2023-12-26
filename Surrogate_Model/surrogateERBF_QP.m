clc;
clear;
close all hidden;

load('PK.mat');
[predict_function,ensemleradialbasis_model]=interpEnsemleRadialBasisPreModel...
    (X,Y);
figure_handle=figure(1);
interpVisualize(ensemleradialbasis_model,low_bou,up_bou,figure_handle)

function [predict_function,ensemleradialbasis_model]=interpEnsemleRadialBasisPreModel...
    (X,Y)
% get ensemle radial basis function interpolation model function
% using quadratic programming to calculate weigth of each sub model
% using cubic interpolation optimal to decrese time use
% input initial data X, Y, which are real data
% X, Y are x_number x variable_number matrix
% aver_X,stdD_X is 1 x x_number matrix
% output is a radial basis model, include X, Y, base_function
% and predict_function
%
% reference: [1] SHI R, LIU L, LONG T, et al. An efficient ensemble of
% radial basis functions method based on quadratic programming [J].
% Engineering Optimization, 2016, 48(1202 - 25.
%
% Copyright 2023 Adel
%
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

c_initial=(prod(max(X_nomlz)-min(Y_nomlz))/x_number)^(1/variable_number);

% initialization distance of all X
X_dis=zeros(x_number,x_number);
for variable_index=1:variable_number
    X_dis=X_dis+(X_nomlz(:,variable_index)-X_nomlz(:,variable_index)').^2;
end
X_dis=sqrt(X_dis);

% linear kernal function
basis_function_linear=@(r,c) r+c;
dRM_dc_function=@(x_number,X_dis,rdibas_matrix,c) ones(x_number,x_number);
object_function=@(c) objectFunctionRadiabasis....
    (X_dis,Y_nomlz,x_number,basis_function_linear,c,dRM_dc_function);

[linear_c_backward,fval_backward,~]=optimalCubicInterp...
    (object_function,-1e2,-1e2,1e2,1e-3);
[linear_c_forward,fval_forward,~]=optimalCubicInterp...
    (object_function,1e2,-1e2,1e2,1e-3);
if fval_forward < fval_backward
    c_linear=linear_c_forward;
else
    c_linear=linear_c_backward;
end

% gauss kernal function
basis_function_gauss=@(r,c) exp(-c*r.^2);
dRM_dc_function=@(x_number,X_dis,rdibas_matrix,c) -X_dis.^2.*rdibas_matrix;
object_function=@(c) objectFunctionRadiabasis....
    (X_dis,Y_nomlz,x_number,basis_function_gauss,c,dRM_dc_function);
[c_gauss,~,~,~]=optimalCubicInterp...
    (object_function,c_initial,1e-2,1e2,1e-3);

% spline kernal function
basis_function_spline=@(r,c) r.^2.*log(r.^2*c+1e-3);
dRM_dc_function=@(x_number,X_dis,rdibas_matrix,c) X_dis.^4./(X_dis.^2*c+1e-3);
object_function=@(c) objectFunctionRadiabasis....
    (X_dis,Y_nomlz,x_number,basis_function_spline,c,dRM_dc_function);
[c_spline,~,~,~]=optimalCubicInterp...
    (object_function,c_initial,1e-2,1e2,1e-3);

% triple kernal function
basis_function_triple=@(r,c) (r+c).^3;
dRM_dc_function=@(x_number,X_dis,rdibas_matrix,c) 3*(X_dis+c).^3;
object_function=@(c) objectFunctionRadiabasis....
    (X_dis,Y_nomlz,x_number,basis_function_triple,c,dRM_dc_function);
[c_triple,~,~,~]=optimalCubicInterp...
    (object_function,c_initial,1e-2,1e2,1e-3);

% multiquadric kernal function
basis_function_multiquadric=@(r,c) sqrt(r+c);
dRM_dc_function=@(x_number,X_dis,rdibas_matrix,c) 0.5./rdibas_matrix;
object_function=@(c) objectFunctionRadiabasis....
    (X_dis,Y_nomlz,x_number,basis_function_multiquadric,c,dRM_dc_function);
[c_binomial,~,~,~]=optimalCubicInterp...
    (object_function,c_initial,1e-2,1e2,1e-3);

% inverse multiquadric kernal function
basis_function_inverse_multiquadric=@(r,c) 1./sqrt(r+c);
dRM_dc_function=@(x_number,X_dis,rdibas_matrix,c) -0.5*rdibas_matrix.^3;
object_function=@(c) objectFunctionRadiabasis....
    (X_dis,Y_nomlz,x_number,basis_function_inverse_multiquadric,c,dRM_dc_function);

% c_initial=1;
% drawFunction(object_function,1e-1,10);

[c_inverse_binomial,~,~,~]=optimalCubicInterp...
    (object_function,c_initial,1e-2,1e2,1e-3);

% generate total model
basis_function_list={
    @(r) basis_function_linear(r,c_linear);
    @(r) basis_function_gauss(r,c_gauss);
    @(r) basis_function_spline(r,c_spline);
    @(r) basis_function_triple(r,c_triple);
    @(r) basis_function_multiquadric(r,c_binomial);
    @(r) basis_function_inverse_multiquadric(r,c_inverse_binomial);};
c_list=[c_linear;c_gauss;c_spline;c_triple;c_binomial;c_inverse_binomial];

model_number=size(basis_function_list,1);
beta_list=zeros(x_number,model_number);
rdibas_matrix_list=zeros(x_number,x_number,model_number);
inv_rdibas_matrix_list=zeros(x_number,x_number,model_number);

% calculate model matrix and error
model_error_list=zeros(model_number,x_number);
for model_index=1:model_number
    basis_function=basis_function_list{model_index};
    [beta,rdibas_matrix,inv_rdibas_matrix]=interpRadialBasis...
        (X_dis,Y_nomlz,basis_function,x_number);
    beta_list(:,model_index)=beta;
    rdibas_matrix_list(:,:,model_index)=rdibas_matrix;
    inv_rdibas_matrix_list(:,:,model_index)=inv_rdibas_matrix;
    
    model_error_list(model_index,:)=(beta./...
        diag(inv_rdibas_matrix))';
end

% calculate weight of each model
C=model_error_list*model_error_list';
eta=trace(C)/x_number;
I_model=eye(model_number);
one_model=ones(model_number,1);
w=(C+eta*I_model)\one_model/...
    (one_model'*((C+eta*I_model)\one_model));
while min(w) < -0.05
    % minimal weight cannot less than zero too much
    eta=eta*10;
    w=(C+eta*I_model)\one_model/...
        (one_model'*((C+eta*I_model)\one_model));
end

% initialization predict function
predict_function=@(X_predict) interpEnsemleRadialBasisPredictor...
    (X_predict,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_number,variable_number,model_number,beta_list,basis_function_list,w);

ensemleradialbasis_model.X=X;
ensemleradialbasis_model.Y=Y;
ensemleradialbasis_model.aver_X=aver_X;
ensemleradialbasis_model.stdD_X=stdD_X;
ensemleradialbasis_model.aver_Y=aver_Y;
ensemleradialbasis_model.stdD_Y=stdD_Y;

ensemleradialbasis_model.basis_function_list=basis_function_list;
ensemleradialbasis_model.c_list=c_list;
ensemleradialbasis_model.beta_list=beta_list;
ensemleradialbasis_model.rdibas_matrix_list=rdibas_matrix_list;
ensemleradialbasis_model.inv_rdibas_matrix_list=inv_rdibas_matrix_list;
ensemleradialbasis_model.model_error_list=model_error_list;
ensemleradialbasis_model.w=w;

ensemleradialbasis_model.predict_function=predict_function;

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

    function Y_pred=interpEnsemleRadialBasisPredictor...
            (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,model_number,beta_list,basis_function_list,w)
        % ensemle radial basis function interpolation predict function
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

        % calculate each sub model predict fval and get predict_y
        y_pred_nomlz_list=zeros(x_pred_num,model_number);
        for model_index__=1:model_number
            basis_function__=basis_function_list{model_index__};
            beta__=beta_list(:,model_index__);
            y_pred_nomlz_list(:,model_index__)=basis_function__(X_dis_pred)*beta__;
        end
        Y_pred_nomlz=y_pred_nomlz_list*w;
        
        % normalize data
        Y_pred=Y_pred_nomlz*stdD_Y+aver_Y;
    end

end
