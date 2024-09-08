function model_RBFOpt=srgtRBFOpt(X,Y,basis_fcn,dRM_dc_fcn)
% generate radial basis function surrogate model
% input data will be normalize by average and standard deviation of data
% reference [1] to optimal hyper parameter of radial basis function
% optimal function is cubic interpolation optimization
%
% input:
% X (matrix): x_num x vari_num matrix
% Y (matrix): x_num x 1 matrix
% basis_fcn (function handle): optional input, default is (r+c).^3
% dRM_dc_fcn (function handle): optional input, derivative of basis_fcn with respect ot c
%
% output:
% model_RBFOpt(struct): a radial basis function model
%
% abbreviation:
% num: number, pred: predict, vari: variable
%
% reference: [1] RIPPA S. An algorithm for selecting a good value for the
% parameter c in radial basis function interpolation [J]. Advances in
% Computational Mathematics, 1999, 11(2-3): 193-210
%
% Copyright 2023.2 Adel
%
if nargin < 4
    basis_fcn=[];
    dRM_dc_fcn=[];
end

% normalize data
[x_num,vari_num]=size(X);
aver_X=mean(X);
stdD_X=std(X);stdD_X(stdD_X == 0)=1;
aver_Y=mean(Y);
stdD_Y=std(Y);stdD_Y(stdD_Y == 0)=1;
X_nomlz=(X-aver_X)./stdD_X;
Y_nomlz=(Y-aver_Y)./stdD_Y;

% initialization distance of all X
X_dis=zeros(x_num,x_num);
for vari_index=1:vari_num
    X_dis=X_dis+(X_nomlz(:,vari_index)-X_nomlz(:,vari_index)').^2;
end
X_dis=sqrt(X_dis);

% triple kernal function
if isempty(basis_fcn),basis_fcn=@(r,c) (r+c).^3;end
if isempty(dRM_dc_fcn),dRM_dc_fcn=@(x_number,X_dis,RBF_matrix,c) 3*(X_dis+c).^2;end
obj_fcn=@(c) objRBF(X_dis,Y_nomlz,x_num,basis_fcn,c,dRM_dc_fcn);
[c,~,~,~]=fminbnd(obj_fcn,1e-2,1e2,optimset('Display','none'));
basis_fcn=@(r) basis_fcn(r,c);

[beta,RBF_matrix,inv_RBF_matrix]=calRBF(X_dis,Y_nomlz,basis_fcn,x_num);

% initialization predict function
pred_fcn=@(X_predict) predictRBF...
    (X_predict,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_num,vari_num,beta,basis_fcn);

model_RBFOpt.X=X;
model_RBFOpt.Y=Y;

model_RBFOpt.basis_fcn=basis_fcn;

model_RBFOpt.RBF_matrix=RBF_matrix;
model_RBFOpt.inv_RBF_matrix=inv_RBF_matrix;
model_RBFOpt.beta=beta;

model_RBFOpt.predict=pred_fcn;
model_RBFOpt.Err=@() calErrRBF(stdD_Y,inv_RBF_matrix,beta);

    function [fval,grad]=objRBF....
            (X_dis,Y,x_number,basis_fcn,c,dRM_dc_fcn)
        % simple approximation to coefficient of multiple correlation
        % basis_function input is c and x_sq
        %
        basis_fcn=@(r) basis_fcn(r,c);
        [beta__,RBF_matrix__,inv_RBF_matrix__]=calRBF...
            (X_dis,Y,basis_fcn,x_number);
        EPN=beta__./diag(inv_RBF_matrix__);
        fval=sum(EPN.^2);

        % calculate gradient
        if nargout > 1
            inv_RBF_matrix_gradient=-inv_RBF_matrix__*...
                dRM_dc_fcn...
                (x_number,X_dis,RBF_matrix__,c)*inv_RBF_matrix__;
            EPN_grad=zeros(x_number,1);
            I=eye(x_number);
            for x_index=1:x_number
                EPN_grad(x_index)=(I(x_index,:)*inv_RBF_matrix_gradient*Y)/...
                    inv_RBF_matrix__(x_index,x_index)-...
                    beta__(x_index)*(I(x_index,:)*inv_RBF_matrix_gradient*I(:,x_index))/...
                    inv_RBF_matrix__(x_index,x_index)^2;
            end

            grad=2*sum(EPN.*EPN_grad);
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
        RBF_matrix=RBF_matrix+eye(x_num)*((1000+x_num)*eps);

        % solve beta
        if rcond(RBF_matrix) < eps
            beta=lsqminnorm(RBF_matrix,Y);
        else
            beta=RBF_matrix\Y;
        end

        if nargout > 2
            inv_RBF_matrix=RBF_matrix\eye(x_num);
        end
    end

    function [Y_pred]=predictRBF...
            (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,beta,basis_function)
        % radial basis function interpolation predict function
        %
        [x_pred_num,~]=size(X_pred);

        % normalize data
        X_pred_nomlz=(X_pred-aver_X)./stdD_X;

        % calculate distance
        X_dis_pred=zeros(x_pred_num,x_num);
        for vari_idx=1:vari_num
            X_dis_pred=X_dis_pred+...
                (X_pred_nomlz(:,vari_idx)-X_nomlz(:,vari_idx)').^2;
        end
        X_dis_pred=sqrt(X_dis_pred);

        % predict variance
        Y_pred=basis_function(X_dis_pred)*beta;

        % normalize data
        Y_pred=Y_pred*stdD_Y+aver_Y;
    end

    function Err_pred=calErrRBF(stdD_Y,inv_RBF_matrix,beta)
        % analysis method to quickly calculate LOO of RBF surrogate model
        %
        % reference: [1] Rippa S. An Algorithm for Selecting a Good Value
        % for the Parameter c in Radial Basis Function Interpolation [J].
        % Advances in Computational Mathematics, 1999, 11(2-3): 193-210.
        %
        Err_pred=beta*stdD_Y./diag(inv_RBF_matrix);
    end
end
