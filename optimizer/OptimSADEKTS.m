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
        function self=OptimSADEKTS(NFE_max,iter_max,obj_tol,con_tol)
            % initialize optimization
            %
            if nargin < 4
                con_tol=[];
                if nargin < 3
                    obj_tol=[];
                    if nargin < 2
                        iter_max=[];
                        if nargin < 1
                            NFE_max=[];
                        end
                    end
                end
            end

            if isempty(con_tol)
                con_tol=1e-3;
            end
            if isempty(obj_tol)
                obj_tol=1e-6;
            end

            self@OptimKRGCDE(NFE_max,iter_max,obj_tol,con_tol);
        end

        function X_init=dataKTS(self,datalib_source,vari_num,low_bou,up_bou,pop_num,init_type)
            % Knowledge-Transfer-Based Sampling Method
            %
            if nargin < 7
                init_type='';
                if nargin < 6
                    pop_num=[];
                end
            end
            [x_src_num,vari_src_num]=size(datalib_source.X);
            if vari_src_num ~= vari_num
                error('OptimSADEKTS: SVM-KTS donot support inequal variable number');
            end

            if isempty(init_type), init_type='inherit';end
            if isempty(pop_num), pop_num=min(100,10*vari_num);end

            % import data from source data library and rank data
            [X_src,Obj_src,Con_src,Coneq_src,Vio_src]=datalibLoad(datalib_source);
            [X_src,~,~,~]=self.rankData(X_src,Obj_src,Con_src,Coneq_src,Vio_src);
            X_src_norm=(X_src-datalib_source.low_bou)./(datalib_source.up_bou-datalib_source.low_bou);

            % generate initial latin hypercubic which will be corrected
            switch init_type
                case 'restart'
                    X_tar_norm=lhsdesign(pop_num,vari_num);
                case 'inherit'
                    % remove ovevrlap point
                    X_src_inherit_norm=X_src_norm;
                    x_idx=1;
                    while x_idx < size(X_src_inherit_norm,1)
                        x_src_norm=X_src_inherit_norm(x_idx,:);
                        idx_search=x_idx+1:size(X_src_inherit_norm,1);
                        dist=vecnorm(X_src_inherit_norm(idx_search,:)-x_src_norm,2,2);
                        idx_search(dist > self.add_tol)=[];
                        X_src_inherit_norm(idx_search,:)=[];
                        x_idx=x_idx+1;
                    end

                    if x_src_num < pop_num
                        X_tar_norm=[X_src_inherit_norm;lhsdesign(pop_num-x_src_num,vari_num)];
                    else
                        X_tar_norm=X_src_inherit_norm(randperm(size(X_src_inherit_norm,1),pop_num),:);
                    end
            end

            % KTS
            % line(X_src_norm(:,1),X_src_norm(:,2),'lineStyle','none','Marker','o','Color','b')
            % line(X_tar_norm(:,1),X_tar_norm(:,2),'lineStyle','none','Marker','^','Color','k')

            N_elite=round(x_src_num*self.elite_rate);

            % generate SVM model, elite will be 1
            Class_src=[true(N_elite,1);false(x_src_num-N_elite,1)];
            model_option.box_con=self.penalty_SVM;
            model_SVM=classifySVM(X_src_norm,Class_src,model_option);

            % get predict value by SVM, if equal to 1 is elite
            Bool_tar=model_SVM.predict(X_tar_norm) == 1;

            iter=0;
            while all(~Bool_tar) && iter < 10 % while all initial target point are not in superior
                X_tar_norm=lhsdesign(pop_num,vari_num);

                % get predict value by SVM
                Bool_tar=model_SVM.predict(X_tar_norm) == 1;
                iter=iter+1;
            end

            % move X to nearest X_superior
            X_superior_norm=X_tar_norm(Bool_tar,:);
            X_inferior_norm=X_tar_norm;
            X_inferior_norm(Bool_tar,:)=[];
            for x_idx=1:size(X_inferior_norm,1)
                x_src_norm=X_inferior_norm(x_idx,:);

                x_dist=sqrt(sum((x_src_norm-X_superior_norm).^2,2));
                [~,idx]=min(x_dist);
                x_superior=X_superior_norm(idx,:); % nearest x_superior
                x_src_norm=x_src_norm+self.correction_factor*(x_superior-x_src_norm);
                X_inferior_norm(x_idx,:)=x_src_norm;
            end

            % assembly
            X_init_norm=[X_inferior_norm;X_superior_norm];
            X_init=X_init_norm.*(up_bou-low_bou)+low_bou;
            self.X_init=X_init;

            % KTS
            % line(X_tar_norm(:,1),X_tar_norm(:,2),'lineStyle','none','Marker','^','Color','k')
            % line(X_init_norm(:,1),X_init_norm(:,2),'lineStyle','none','Marker','v','Color','r')
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
