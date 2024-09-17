function srgt=srgtsfPRS(X,Y,poly_param)
% generarte polynomial response surface surrogate model
% input data will be normalize by average and standard deviation of data
%
% input:
% X (matrix): x_num x vari_num matrix
% Y (vector): x_num x 1 matrix
% poly_param (double/function handle): optional input, polynomial order or function
%
% output include:
% srgt(struct): a polynomial response surface model
%
% Copyright 2023.2 Adel
%
if nargin < 3,poly_param=[];end

% initialize data

% normalize data
[x_num,vari_num]=size(X);
aver_X=mean(X);
stdD_X=std(X);stdD_X(stdD_X == 0)=1;
aver_Y=mean(Y);
stdD_Y=std(Y);stdD_Y(stdD_Y == 0)=1;

% polynomial function define
if isempty(poly_param),poly_param=2;end
if isnumeric(poly_param)
    switch poly_param
        case 0
            poly_fcn=@(X)ones(size(X,1),1).*stdD_Y+aver_Y;
        case 1
            poly_fcn=@(X)[ones(size(X,1),1),X-aver_X].*stdD_Y+aver_Y;
        case 2
            poly_fcn=@(X)polyFcnQuad(X);
        otherwise
            error('srgtsfPRS: poly_param larger than 2');
    end
elseif isa(poly_param,'function_handle'),poly_fcn=poly_param;
else,error('srgtsfPRS: error input of poly_param');end

% calculate PRS
beta=calPRS(X,Y,poly_fcn);

% initialization predict function
pred_fcn=@(X_pred)predictPRS(X_pred,beta,poly_fcn);

srgt.X=X;
srgt.Y=Y;
srgt.poly_param=poly_param;
srgt.poly_fcn=poly_fcn;
srgt.beta=beta;
srgt.predict=pred_fcn;

    function beta=calPRS(X,Y,poly_fcn)
        % PRS train function, calculate beta
        % y(x)=f(x)
        %
        % input:
        % X (matrix): trained X, x_num x vari_num
        % Y (vector): trained Y, x_num x 1
        % poly_fcn (function handle): polynomial function
        %
        psi=poly_fcn(X);
        Y_norm=(Y-aver_Y)./stdD_Y;
        beta=psi\Y_norm;
    end

    function Y_pred=predictPRS(X_pred,beta,poly_fcn)
        % KRG predict function
        %
        % input:
        % X_pred (matrix): predict X, x_pred_num x vari_num
        % beta (function handle): regression coefficient vector
        % poly_fcn (function handle): polynomial function
        %
        % output:
        % Y_pred (vector): predict Y, x_pred_num x 1
        %
        psi_pred=poly_fcn(X_pred);
        Y_pred=(psi_pred*beta)*stdD_Y+aver_Y;
    end

    function phi=polyFcnQuad(X)
        % quadratic basis polynomial function
        %
        X_norm=(X-aver_X);
        phi_crs=zeros(size(X,1),(vari_num-1)*vari_num/2);

        crs_idx=1;
        for i=1:vari_num
            batch=vari_num-i;
            phi_crs(:,crs_idx:(crs_idx+batch-1))=X_norm(:,i).*X_norm(:,i+1:end);
            crs_idx=crs_idx+batch;
        end
        phi=[ones(size(X,1),1),X_norm,X_norm.^2,phi_crs].*stdD_Y+aver_Y;
    end
end
