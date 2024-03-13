function model_RBF=srgtRBF(X,Y,basis_fcn)
% generate radial basis function surrogate model
% input data will be normalize by average and standard deviation of data
%
% input:
% X (matrix): x_num x vari_num matrix
% Y (matrix): x_num x 1 matrix
% basis_fcn (function handle): optional input, default is r.^3
%
% output:
% model_RBF(struct): a radial basis function model
%
% abbreviation:
% num: number, pred: predict, vari: variable
%
% Copyright 2023.2 Adel
%
if nargin < 3
    basis_fcn=[];
end

% normalize data
[x_num,vari_num]=size(X);
aver_X=mean(X);
stdD_X=std(X);stdD_X(stdD_X == 0)=1;
aver_Y=mean(Y);
stdD_Y=std(Y);stdD_Y(stdD_Y == 0)=1;
X_nomlz=(X-aver_X)./stdD_X;
Y_nomlz=(Y-aver_Y)./stdD_Y;

if isempty(basis_fcn)
    basis_fcn=@(r) r.^3;
end

% initialization distance of all X
X_dis=zeros(x_num,x_num);
for vari_idx=1:vari_num
    X_dis=X_dis+(X_nomlz(:,vari_idx)-X_nomlz(:,vari_idx)').^2;
end
X_dis=sqrt(X_dis);

[beta,RBF_matrix]=calRBF(X_dis,Y_nomlz,basis_fcn,x_num);

% initialization predict function
pred_fcn=@(X_predict) predictRBF...
    (X_predict,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_num,vari_num,beta,basis_fcn,RBF_matrix);

model_RBF.X=X;
model_RBF.Y=Y;

model_RBF.basis_fcn=basis_fcn;
model_RBF.beta=beta;

model_RBF.predict=pred_fcn;
model_RBF.Err=@() calErrRBF(stdD_Y,RBF_matrix,beta);

    function [beta,RBF_matrix]=calRBF(X_dis,Y,basis_fcn,x_num)
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
    end

    function [Y_pred,var_pred]=predictRBF...
            (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,beta,basis_fcn,RBF_matrix)
        % radial basis function interpolation predict function
        %
        x_pred_num=size(X_pred,1);

        % normalize data
        X_pred_nomlz=(X_pred-aver_X)./stdD_X;
        
        % calculate distance
        X_dis_pred=zeros(x_pred_num,x_num);
        for vari_i=1:vari_num
            X_dis_pred=X_dis_pred+...
                (X_pred_nomlz(:,vari_i)-X_nomlz(:,vari_i)').^2;
        end
        X_dis_pred=sqrt(X_dis_pred);
        
        % predict variance
        basis=basis_fcn(X_dis_pred);
        Y_pred=basis*beta;
        
        % normalize data
        Y_pred=Y_pred*stdD_Y+aver_Y;

        if nargout > 1
            var_pred=-basis/RBF_matrix*basis';
            var_pred=diag(var_pred);
        end
    end

    function Err_pred=calErrRBF(stdD_Y,RBF_matrix,beta)
        % analysis method to quickly calculate LOO of RBF surrogate model
        %
        % reference: [1] Rippa S. An Algorithm for Selecting a Good Value
        % for the Parameter c in Radial Basis Function Interpolation [J].
        % Advances in Computational Mathematics, 1999, 11(2-3): 193-210.
        %
        inv_RBF_matrix=RBF_matrix\eye(size(RBF_matrix));
        Err_pred=beta*stdD_Y./diag(inv_RBF_matrix);
    end
end
