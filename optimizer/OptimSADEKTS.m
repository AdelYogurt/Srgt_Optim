classdef OptimSADEKTS < OptimKRGCDE
    % SADE-KTS optimization algorithm
    % adding knowledge-transfer-based sampling to original SADE algorithm
    %
    % referance: [1] LONG T, YE N, SHI R, et al. Surrogate-Assisted
    % Differential Evolution Using Knowledge-Transfer-Based Sampling for
    % Expensive Optimization Problems [J]. AIAA Journal, 2021, 60(1-16.
    %
    % Copyright 2022 Adel
    %
    % abbreviation:
    % obj: objective, con: constraint, iter: iteration, tor: torlance
    % fcn: function, srgt: surrogate
    % lib: library, init: initial, rst: restart, potl: potential
    % pred: predict, var:variance, vari: variable, num: number
    %
    properties
        elite_rate=0.5;
        correction_factor=0.1;
        penalty_SVM=100; % SVM parameter
    end

    % main function
    methods
        function self=OptimSADEKTS(NFE_max,iter_max,obj_torl,con_torl,dataoptim_filestr)
            % initialize optimization
            %
            if nargin < 5
                dataoptim_filestr='';
                if nargin < 4
                    con_torl=[];
                    if nargin < 3
                        obj_torl=[];
                        if nargin < 2
                            iter_max=[];
                            if nargin < 1
                                NFE_max=[];
                            end
                        end
                    end
                end
            end

            if isempty(con_torl)
                con_torl=1e-3;
            end
            if isempty(obj_torl)
                obj_torl=1e-6;
            end

            self@OptimKRGCDE(NFE_max,iter_max,obj_torl,con_torl,dataoptim_filestr);
        end

        function X_init=dataKTS(self,datalib_source,pop_num)
            % Knowledge-Transfer-Based Sampling Method
            %
            [x_num,vari_num]=size(datalib_source.X);
            if nargin < 3
                pop_num=min(100,10*vari_num);
            end

            % generate initial latin hypercubic which will be corrected
            % X_source=lhsdesign(pop_num,vari_num).*(up_bou-low_bou)+low_bou;

            % import data from data library to rank data
            [X,Obj,Con,Coneq,~,~]=datalibLoad(datalib_source);
            if pop_num <= x_num
                idx_list=randperm(x_num,pop_num);
                X_source=X(idx_list,:);
            else
                X_source=[X;lhsdesign((pop_num-size(X,1)),vari_num).*(up_bou-low_bou)+low_bou];
            end

            % KTS
            %     line(X_initial(:,1),X_initial(:,2),'lineStyle','none','Marker','o','Color','b')

            % rank x_list data
            [X,~,~,~]=self.rankData(X,Obj,Con,Coneq,[],self.con_torl);

            N_elite=round(x_num*self.elite_rate);

            % generate SVM model, elite will be 1
            Class=[ones(N_elite,1);-ones(x_num-N_elite,1)];
            model_SVM=self.classifySVM(X,Class,self.penalty_SVM);

            % get predict value by SVM, if equal to 1 is elite
            Bool=model_SVM.predict(X_source) == 1;

            while all(~Bool)
                X_source=lhsdesign(pop_num,vari_num).*(up_bou-low_bou)+low_bou;

                % get predict value by SVM
                Bool=model_SVM.predict(X_source) == 1;
            end

            % move X to nearest X_superior
            X_superior=X_source(Bool,:);
            X_inferior=X_source;
            X_inferior(Bool,:)=[];
            for x_index=1:size(X_inferior,1)
                x=X_inferior(x_index,:);

                distance=sqrt(sum((x-X_superior).^2,2));
                [~,index]=min(distance);
                x_superior=X_superior(index,:); % nearest x_superior
                x=x+self.correction_factor*(x_superior-x);
                X_inferior(x_index,:)=x;
            end

            X_init=[X_inferior;X_superior];
            %     line(X_updata(:,1),X_updata(:,2),'lineStyle','none','Marker','.','Color','r')
        end
    end

    % machine learning
    methods(Static)
        function model_SVM=classifySVM(X,Y,model_option)
            % generate support vector machine model
            % use fmincon to get alpha
            %
            % input:
            % X(x_num x vari_num matrix), Y(x_num x 1 matrix),...
            % model_option(optional, include: box_con, kernel_fcn, optimize_option)
            %
            % output:
            % model_SVM(a support vector machine model)
            %
            % abbreviation:
            % num: number, pred: predict, vari: variable
            % nomlz: normalization, var: variance, fcn: function
            %
            if nargin < 3,model_option=struct();end
            if ~isfield(model_option,'box_con'), model_option.('box_con')=1;end
            if ~isfield(model_option,'kernel_fcn'), model_option.('kernel_fcn')=[];end
            if ~isfield(model_option,'optimize_option'), model_option.('optimize_option')=optimoptions...
                    ('quadprog','Display','none');end

            % normalization data
            [x_num,vari_num]=size(X);
            aver_X=mean(X);
            stdD_X=std(X);stdD_X(stdD_X == 0)=1;
            X_nomlz=(X-aver_X)./stdD_X;
            Y(Y == 0)=-1;

            % default kernal function
            kernel_fcn=model_option.kernel_fcn;
            if isempty(kernel_fcn)
                % notice after standard normal distribution normalize
                sigma=1/vari_num;
                kernel_fcn=@(U,V) kernelGaussian(U,V,sigma);
            end

            % initialization kernal function process X_cov
            K=kernel_fcn(X_nomlz,X_nomlz);
            box_con=model_option.box_con;
            alpha=solveAlpha(K,Y,x_num,box_con,model_option);

            % obtain other paramter
            alpha_Y=alpha.*Y;
            w=sum(alpha_Y.*X_nomlz);
            bool_support=(alpha > 1e-6); % support vector
            alpha_Y_cov=K*alpha_Y;
            b=sum(Y(bool_support)-alpha_Y_cov(bool_support))/length(bool_support);

            % generate predict function
            pred_fcn=@(X_pred) predictSVM(X_pred,X_nomlz,alpha_Y,b,aver_X,stdD_X,kernel_fcn);

            % output model
            model_SVM.X=X;
            model_SVM.Y=Y;
            model_SVM.aver_X=aver_X;
            model_SVM.stdD_X=stdD_X;

            model_SVM.alpha=alpha;
            model_SVM.bool_support=bool_support;
            model_SVM.w=w;
            model_SVM.b=b;

            model_SVM.box_con=box_con;
            model_SVM.kernel=kernel_fcn;
            model_SVM.predict=pred_fcn;

            function [Y_pred,Prob_pred]=predictSVM...
                    (X_pred,X_nomlz,alpha_Y,b,aver_X,stdD_X,kernel_fcn)
                % SVM predict function
                %
                [x_pred_num,~]=size(X_pred);
                X_pred_nomlz=(X_pred-aver_X)./stdD_X;

                K_pred=kernel_fcn(X_pred_nomlz,X_nomlz);
                Prob_pred=K_pred*alpha_Y+b;

                Prob_pred=1./(1+exp(-Prob_pred));
                Y_pred=ones(x_pred_num,1);
                Y_pred(Prob_pred < 0.5)=0;
            end

            function K=kernelGaussian(U,V,sigma)
                % gaussian kernal function
                %
                K=zeros(size(U,1),size(V,1));
                for vari_index=1:size(U,2)
                    K=K+(U(:,vari_index)-V(:,vari_index)').^2;
                end
                K=exp(-K*sigma);
            end

            function alpha=solveAlpha(K,Y,x_num,box_con,model_option)
                % min SVM object function to get alpha
                %

                % solve alpha need to minimize followed equation
                % obj=sum(alpha)-(alpha.*Y)'*K*(alpha.*Y)/2;

                alpha=ones(x_num,1)*0.5;
                low_bou_A=0*ones(x_num,1);
                if isempty(box_con) || box_con==0
                    up_bou_A=[];
                else
                    up_bou_A=box_con*ones(x_num,1);
                end
                Aeq=Y';
                hess=-K.*(Y*Y');
                grad=ones(x_num,1);
                alpha=quadprog(hess,grad,[],[],Aeq,0,low_bou_A,up_bou_A,alpha,model_option.optimize_option);
            end

        end

    end
end

%% data library function
function [X,Obj,Con,Coneq,Vio]=datalibLoad(datalib,low_bou,up_bou)
% updata data to exist data lib
%
if nargin < 3,up_bou=[];if nargin < 2,low_bou=[];end,end
if isempty(up_bou), up_bou=realmax;end
if isempty(low_bou), low_bou=-realmax;end

idx=[];
for x_idx=1:size(datalib.X,1)
    x=datalib.X(x_idx,:);
    if all(x > low_bou) && all(x < up_bou), idx=[idx;x_idx];end
end

X=datalib.X(idx,:);
Obj=datalib.Obj(idx,:);
if ~isempty(datalib.Con),Con=datalib.Con(idx,:);
else,Con=[];end
if ~isempty(datalib.Coneq),Coneq=datalib.Coneq(idx,:);
else,Coneq=[];end
if ~isempty(datalib.Vio),Vio=datalib.Vio(idx,:);
else,Vio=[];end

end
