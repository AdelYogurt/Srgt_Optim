function model_RBFQP=srgtRBFQdPg(X,Y)
% generate ensemle radial basis function surrogate model
% input data will be normalize by average and standard deviation of data
% using quadratic programming to calculate weigth of each sub model
% using cubic interpolation optimal to decrese time use
%
% input:
% X (matrix): x_num x vari_num matrix
% Y (matrix): x_num x 1 matrix
%
% output:
% model_RBFQP(a ensemle radial basis function model)
%
% abbreviation:
% num: number, pred: predict, vari: variable
%
% reference: [1] SHI R, LIU L, LONG T, et al. An efficient ensemble of
% radial basis functions method based on quadratic programming [J].
% Engineering Optimization, 2016, 48(1202 - 25.
%
% Copyright 2023.2 Adel
%

% normalize data
[x_num,vari_num]=size(X);
aver_X=mean(X);
stdD_X=std(X);stdD_X(stdD_X == 0)=1;
aver_Y=mean(Y);
stdD_Y=std(Y);stdD_Y(stdD_Y == 0)=1;
X_nomlz=(X-aver_X)./stdD_X;
Y_nomlz=(Y-aver_Y)./stdD_Y;

c_initial=(prod(max(X_nomlz)-min(Y_nomlz))/x_num)^(1/vari_num);

% initialization distance of all X
X_dis=zeros(x_num,x_num);
for vari_idx=1:vari_num
    X_dis=X_dis+(X_nomlz(:,vari_idx)-X_nomlz(:,vari_idx)').^2;
end
X_dis=sqrt(X_dis);

% linear kernal function
basis_fcn_linear=@(r,c) r+c;
dRM_dc_fcn=@(x_number,X_dis,RBF_matrix,c) ones(x_number,x_number);
obj_fcn=@(c) objRBF(X_dis,Y_nomlz,x_num,basis_fcn_linear,c,dRM_dc_fcn);

[linear_c_backward,fval_backward,~]=optimCubicInterp(obj_fcn,-1e2,-1e2,1e2,1e-3);
[linear_c_forward,fval_forward,~]=optimCubicInterp(obj_fcn,1e2,-1e2,1e2,1e-3);
if fval_forward < fval_backward
    c_linear=linear_c_forward;
else
    c_linear=linear_c_backward;
end

% gauss kernal function
basis_fcn_gauss=@(r,c) exp(-c*r.^2);
dRM_dc_fcn=@(x_number,X_dis,RBF_matrix,c) -X_dis.^2.*RBF_matrix;
obj_fcn=@(c) objRBF(X_dis,Y_nomlz,x_num,basis_fcn_gauss,c,dRM_dc_fcn);
[c_gauss,~,~,~]=optimCubicInterp(obj_fcn,c_initial,1e-2,1e2,1e-3);

% spline kernal function
basis_fcn_spline=@(r,c) r.^2.*log(r.^2*c+1e-3);
dRM_dc_fcn=@(x_number,X_dis,RBF_matrix,c) X_dis.^4./(X_dis.^2*c+1e-3);
obj_fcn=@(c) objRBF(X_dis,Y_nomlz,x_num,basis_fcn_spline,c,dRM_dc_fcn);
[c_spline,~,~,~]=optimCubicInterp(obj_fcn,c_initial,1e-2,1e2,1e-3);

% triple kernal function
basis_fcn_triple=@(r,c) (r+c).^3;
dRM_dc_fcn=@(x_number,X_dis,RBF_matrix,c) 3*(X_dis+c).^3;
obj_fcn=@(c) objRBF(X_dis,Y_nomlz,x_num,basis_fcn_triple,c,dRM_dc_fcn);
[c_triple,~,~,~]=optimCubicInterp(obj_fcn,c_initial,1e-2,1e2,1e-3);

% multiquadric kernal function
basis_fcn_multiquadric=@(r,c) sqrt(r+c);
dRM_dc_fcn=@(x_number,X_dis,RBF_matrix,c) 0.5./RBF_matrix;
obj_fcn=@(c) objRBF(X_dis,Y_nomlz,x_num,basis_fcn_multiquadric,c,dRM_dc_fcn);
[c_binomial,~,~,~]=optimCubicInterp(obj_fcn,c_initial,1e-2,1e2,1e-3);

% inverse multiquadric kernal function
basis_fcn_inverse_multiquadric=@(r,c) 1./sqrt(r+c);
dRM_dc_fcn=@(x_number,X_dis,RBF_matrix,c) -0.5*RBF_matrix.^3;
obj_fcn=@(c) objRBF(X_dis,Y_nomlz,x_num,basis_fcn_inverse_multiquadric,c,dRM_dc_fcn);
[c_inverse_binomial,~,~,~]=optimCubicInterp(obj_fcn,c_initial,1e-2,1e2,1e-3);

% c_initial=1;
% drawFcn(obj_fcn,1e-1,10);

% generate total model
basis_fcn_list={
    @(r) basis_fcn_linear(r,c_linear);
    @(r) basis_fcn_gauss(r,c_gauss);
    @(r) basis_fcn_spline(r,c_spline);
    @(r) basis_fcn_triple(r,c_triple);
    @(r) basis_fcn_multiquadric(r,c_binomial);
    @(r) basis_fcn_inverse_multiquadric(r,c_inverse_binomial);};
c_list=[c_linear;c_gauss;c_spline;c_triple;c_binomial;c_inverse_binomial];

model_num=size(basis_fcn_list,1);
beta_list=zeros(x_num,model_num);
RBF_matrix_list=zeros(x_num,x_num,model_num);
inv_RBF_matrix_list=zeros(x_num,x_num,model_num);

% calculate model matrix and error
Err_Model_nomlz=zeros(x_num,model_num);
for model_index=1:model_num
    basis_fcn=basis_fcn_list{model_index};
    [beta,RBF_matrix,inv_RBF_matrix]=calRBF...
        (X_dis,Y_nomlz,basis_fcn,x_num);
    beta_list(:,model_index)=beta;
    RBF_matrix_list(:,:,model_index)=RBF_matrix;
    inv_RBF_matrix_list(:,:,model_index)=inv_RBF_matrix;

    Err_Model_nomlz(:,model_index)=(beta./diag(inv_RBF_matrix));
end

% calculate weight of each model
C=Err_Model_nomlz'*Err_Model_nomlz;
eta=trace(C)/x_num;
I_model=eye(model_num);
one_model=ones(model_num,1);
weight=(C+eta*I_model)\one_model/...
    (one_model'*((C+eta*I_model)\one_model));
while min(weight) < -0.05
    % minimal weight cannot less than zero too much
    eta=eta*10;
    weight=(C+eta*I_model)\one_model/...
        (one_model'*((C+eta*I_model)\one_model));
end

% initialization predict function
pred_fcn=@(X_predict) predictRBFQdPg...
    (X_predict,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_num,vari_num,model_num,beta_list,basis_fcn_list,weight);

model_RBFQP.X=X;
model_RBFQP.Y=Y;

model_RBFQP.model_num=model_num;
model_RBFQP.basis_fcn_list=basis_fcn_list;
model_RBFQP.c_list=c_list;

model_RBFQP.beta_list=beta_list;
model_RBFQP.RBF_matrix_list=RBF_matrix_list;
model_RBFQP.inv_RBF_matrix_list=inv_RBF_matrix_list;
model_RBFQP.Err_Model_nomlz=Err_Model_nomlz;
model_RBFQP.weight=weight;

model_RBFQP.predict=pred_fcn;
model_RBFQP.Err=@() calErrRBFQdPg(stdD_Y,Err_Model_nomlz,weight);

    function [fval,gradient]=objRBF....
            (X_dis,Y,x_num,basis_fcn,c,dRM_dc_fcn)
        % MSE_CV function, simple approximation to RMS
        % basis_fcn input is c and x_sq
        %
        basis_fcn=@(r) basis_fcn(r,c);
        [beta__,RBF_matrix__,inv_RBF_matrix__]=calRBF...
            (X_dis,Y,basis_fcn,x_num);
        U=beta__./diag(inv_RBF_matrix__);
        fval=sum(U.^2);

        % calculate gradient
        if nargout > 1
            inv_RBF_matrix_gradient=-inv_RBF_matrix__*...
                dRM_dc_fcn...
                (x_num,X_dis,RBF_matrix__,c)*inv_RBF_matrix__;
            U_gradient=zeros(x_num,1);
            I=eye(x_num);
            for x_index=1:x_num
                U_gradient(x_index)=(I(x_index,:)*inv_RBF_matrix_gradient*Y)/...
                    inv_RBF_matrix__(x_index,x_index)-...
                    beta__(x_index)*(I(x_index,:)*inv_RBF_matrix_gradient*I(:,x_index))/...
                    inv_RBF_matrix__(x_index,x_index)^2;
            end

            gradient=2*sum(U.*U_gradient);
        end
    end

    function [beta,RBF_matrix,inv_RBF_matrix]=calRBF...
            (X_dis,Y,basis_fcn,x_num)
        % interp polynomial responed surface core function
        % calculation beta
        %
        % Copyright 2022 Adel
        %
        RBF_matrix=basis_fcn(X_dis);

        % stabilize matrix
        RBF_matrix=RBF_matrix+eye(x_num)*(1000+x_num)*eps;

        % solve beta
        inv_RBF_matrix=RBF_matrix\eye(x_num);
        beta=inv_RBF_matrix*Y;
    end

    function Y_pred=predictRBFQdPg...
            (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,model_num,beta_list,basis_fcn_list,w)
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
        y_pred_nomlz_list=zeros(x_pred_num,model_num);
        for model_idx=1:model_num
            y_pred_nomlz_list(:,model_idx)=basis_fcn_list{model_idx}(X_dis_pred)*beta_list(:,model_idx);
        end
        Y_pred_nomlz=y_pred_nomlz_list*w;

        % normalize data
        Y_pred=Y_pred_nomlz*stdD_Y+aver_Y;
    end

    function Error_pred=calErrRBFQdPg(stdD_Y,Error_Model_nomlz,weight)
        % analysis method to quickly calculate R^2 of RBF surrogate model
        %
        % reference: [1] Rippa S. An Algorithm for Selecting a Good Value
        % for the Parameter c in Radial Basis Function Interpolation [J].
        % Advances in Computational Mathematics, 1999, 11(2-3): 193-210.
        %
        Error_pred=sum(Error_Model_nomlz*stdD_Y.*weight',2);
    end

end
