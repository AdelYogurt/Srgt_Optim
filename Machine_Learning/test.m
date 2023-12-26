clc;
clear;
close all hidden;

load('CMF_4_2.mat');

X={XHF,XLF};
F=ones(6,1);

object_function=@(cov) objectFunction(cov,X,F);
cov=log(rand(1,7));
[fval,gradient]=object_function(cov)
[gradient_diff]=differ(object_function,cov)

function [fval,gradient]=objectFunction(cov,X,F)
if nargout < 2
    K=calCovMF(cov,X);
    fval=F'*K*F;
elseif nargout < 3
    [K,dK_dvar]=calCovMF(cov,X);
    fval=F'*K*F;
    gradient=zeros(7,1);
    for gradient_index=1:7
        gradient(gradient_index)=F'*dK_dvar{gradient_index}*F;
    end
end

end

function [K,dK_dvar]=calCovMF(cov,X,Z)
% obtain covariance of x
%
XHF=X{1};
XLF=X{2};
[xH_num,variable_number]=size(XHF);
[xL_num,~]=size(XLF);
x_num=xH_num+xL_num;
X=[XHF;XLF];

lenH=exp(cov(1:variable_number));
etaH=exp(cov(variable_number+1));
lenL=exp(cov(variable_number+1+(1:variable_number)));
etaL=exp(cov(2*(variable_number+1)));
rho=exp(cov(end));

% predict
if nargin > 2 && nargout < 2 && ~isempty(Z)
    if strcmp(Z,'diag')
        K=rho*rho*etaL+etaH;
    else
        [z_num,variable_number]=size(Z);
        % initializate square of X inner distance
        sq_dis=zeros(x_num,z_num,variable_number);
        for len_index=1:variable_number
            sq_dis(:,:,len_index)=(X(:,len_index)-Z(:,len_index)').^2;
        end

        % exp of x__x with H
        exp_disH=zeros(xH_num,variable_number);
        for len_index=1:variable_number
            exp_dis=exp_dis+...
                sq_dis(1:xH_num,:,len_index)/2/lenH(len_index)^2;
        end

        % exp of x__x with L
        exp_disL=zeros(x_num,variable_number);
        for len_index=1:variable_number
            exp_disL=exp_dis+...
                sq_dis(1:x_num,:,len_index)/2/lenL(len_index)^2;
        end

        K=etaL*exp_disL;
        K(1:xH_num,:)=rho*rho*K(1:xH_num,:)+etaH*exp_disH;
        K(xH_num+1:end,:)=rho*K(xH_num+1:end,:);
    end
else
    % initializate square of X inner distance
    sq_dis=zeros(x_num,x_num,variable_number);
    for len_index=1:variable_number
        sq_dis(:,:,len_index)=(X(:,len_index)-X(:,len_index)').^2;
    end

    % exp of x__x with H
    exp_disH=zeros(xH_num);
    for len_index=1:variable_number
        exp_disH=exp_disH+...
            sq_dis(1:xH_num,1:xH_num,len_index)/2/lenH(len_index)^2;
    end
    exp_disH=exp(-exp_disH);
    KH=etaH*exp_disH;

    % exp of x__x with L
    exp_disL=zeros(x_num);
    for len_index=1:variable_number
        exp_disL=exp_disL+...
            sq_dis(1:end,1:end,len_index)/2/lenL(len_index)^2;
    end
    exp_disL=exp(-exp_disL);
    % times rho: HH to rho2, HL to rho, LL to 1
    rho_exp_disL=exp_disL;
    rho_exp_disL(1:xH_num,1:xH_num)=...
        (rho*rho)*exp_disL(1:xH_num,1:xH_num);
    rho_exp_disL((xH_num+1):xH_num,1:xH_num)=...
        rho*exp_disL((xH_num+1):xH_num,1:xH_num);
    rho_exp_disL(1:xH_num,(xH_num+1):xH_num)=...
        rho_exp_disL((xH_num+1):xH_num,1:xH_num)';

    KL=rho_exp_disL*etaL;
    K=KL;
    K(1:xH_num,1:xH_num)=K(1:xH_num,1:xH_num)+KH;

    if nargout >= 2
        dK_dvar=cell(1,2*variable_number+3);

        % len H
        for len_index=1:variable_number
            dK_dlenH=zeros(x_num);
            dK_dlenH(1:xH_num,1:xH_num)=KH.*...
                sq_dis(1:xH_num,1:xH_num,len_index)/lenH(len_index)^2;
            dK_dvar{len_index}=dK_dlenH;
        end

        % eta H
        dK_detaH=zeros(x_num);
        dK_detaH(1:xH_num,1:xH_num)=KH;
        dK_dvar{variable_number+1}=dK_detaH;

        % len L
        for len_index=1:variable_number
            dK_dlenL=KL.*sq_dis(:,:,len_index)/lenL(len_index)^2;
            dK_dvar{(variable_number+1)+len_index}=dK_dlenL;
        end

        % eta H
        dK_dvar{2*(variable_number+1)}=KL;

        % rho
        dK_drho=zeros(x_num);
        dK_drho(1:xH_num,1:xH_num)=...
            2*rho*rho*etaL*exp_disL(1:xH_num,1:xH_num);
        dK_drho((xH_num+1):xH_num,1:xH_num)=...
            rho*etaL*exp_disL((xH_num+1):xH_num,1:xH_num);
        dK_drho(1:xH_num,(xH_num+1):xH_num)=...
            dK_drho((xH_num+1):xH_num,1:xH_num)';
        dK_dvar{end}=dK_drho;
    end
end

end
