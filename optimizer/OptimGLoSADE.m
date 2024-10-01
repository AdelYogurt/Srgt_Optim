classdef OptimGLoSADE < handle
    % KRG-CDE optimization algorithm
    %
    % referance:
    % [1] Wang Y, Yin D Q, Yang S, et al. Global and local
    % surrogate-assisted differential evolution for expensive constrained
    % optimization problems with inequality constraints[J]. IEEE
    % Transactions on Cybernetics, 2019, 49: 1642-56.
    %
    % Copyright 2022 Adel
    %
    % abbreviation:
    % obj: objective, con: constraint, iter: iteration, tor: torlance
    % fcn: function, srgt: surrogate
    % lib: library, init: initial, rst: restart, potl: potential
    % pred: predict, var:variance, vari: variable, num: number
    %

    % basic parameter
    properties
        NFE_max;
        iter_max;
        obj_tol;
        con_tol;

        datalib; % X, Obj, Con, Coneq, Vio
        dataoptim; % NFE, Add_idx, Iter

        obj_fcn_srgt;
        con_fcn_srgt;

        Srgt_obj;
        Srgt_con;
        Srgt_coneq;

        Srgt_obj_local;
        Srgt_con_local;
        Srgt_coneq_local;
    end

    % problem parameter
    properties
        FLAG_CON;
        FLAG_MULTI_OBJ;
        FLAG_MULTI_FIDELITY;
        FLAG_DiISCRETE_VARI;

        FLAG_DRAW_FIGURE=0; % whether draw data
        FLAG_INFORMATION=1; % whether print data
        FLAG_CONV_JUDGE=0; % whether judgment convergence

        datalib_filestr=''; % datalib save mat name
        dataoptim_filestr=''; % optimize save mat namename

        add_tol=1000*eps; % surrogate add point protect range
        X_init=[];

        % hyper parameter
        sample_num_init=[];
        pop_num=[];
        RBF_num=[];
        lambda=[];

        % differ evoluation parameter
        scaling_factor=0.8;
        cross_rate=0.8;

        KRG_option=struct('simplify_flag',true);
    end

    % main function
    methods
        function self=OptimGLoSADE(NFE_max,iter_max,obj_tol,con_tol)
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

            self.NFE_max=NFE_max;
            self.iter_max=iter_max;
            self.obj_tol=obj_tol;
            self.con_tol=con_tol;
        end

        function [x_best,obj_best,NFE,output,con_best,coneq_best,vio_best]=optimize(self,varargin)
            % main optimize function
            %

            % step 1, initialize problem
            if length(varargin) == 1
                % input is struct or object
                problem=varargin{1};
                if isstruct(problem)
                    prob_field=fieldnames(problem);
                    if ~contains(prob_field,'objcon_fcn'), error('OptimGLoSADE.optimize: input problem lack objcon_fcn'); end
                    objcon_fcn=problem.objcon_fcn;
                    if ~contains(prob_field,'vari_num'), error('OptimGLoSADE.optimize: input problem lack vari_num'); end
                    if ~contains(prob_field,'low_bou'), error('OptimGLoSADE.optimize: input problem lack low_bou'); end
                    if ~contains(prob_field,'up_bou'), error('OptimGLoSADE.optimize: input problem lack up_bou'); end
                else
                    prob_method=methods(problem);
                    if ~contains(prob_method,'objcon_fcn'), error('OptimGLoSADE.optimize: input problem lack objcon_fcn'); end
                    objcon_fcn=@(x) problem.objcon_fcn(x);
                    prob_prop=properties(problem);
                    if ~contains(prob_prop,'vari_num'), error('OptimGLoSADE.optimize: input problem lack vari_num'); end
                    if ~contains(prob_prop,'low_bou'), error('OptimGLoSADE.optimize: input problem lack low_bou'); end
                    if ~contains(prob_prop,'up_bou'), error('OptimGLoSADE.optimize: input problem lack up_bou'); end
                end
                vari_num=problem.vari_num;
                low_bou=problem.low_bou;
                up_bou=problem.up_bou;
            else
                % multi input
                varargin=[varargin,repmat({[]},1,4-length(varargin))];
                [objcon_fcn,vari_num,low_bou,up_bou]=varargin{:};
            end

            % hyper parameter
            if isempty(self.pop_num),self.pop_num=min(100,10*vari_num);end
            if isempty(self.RBF_num),self.RBF_num=max(100,(vari_num+1)*(vari_num+2)/2);end
            self.sample_num_init=self.pop_num;
            self.lambda=100;

            % NFE and iteration setting
            if isempty(self.NFE_max),self.NFE_max=10+10*vari_num;end
            if isempty(self.iter_max),self.iter_max=20+20*vari_num;end

            if isempty(self.dataoptim)
                self.dataoptim.NFE=0;
                self.dataoptim.Add_idx=[];
                self.dataoptim.iter=0;
                self.dataoptim.done=false;
            end

            % step 2, generate initial data library
            self.sampleInit(objcon_fcn,vari_num,low_bou,up_bou);

            % step 3-7, adaptive samlping base on optimize strategy

            % initialize all data to begin optimize
            [X,Obj,Con,Coneq,Vio]=self.datalibLoad(self.datalib);
            obj_num=size(Obj,2);con_num=size(Con,2);coneq_num=size(Coneq,2);vio_num=size(Vio,2);
            result_X=zeros(self.iter_max,vari_num);
            result_Obj=zeros(self.iter_max,1);
            if con_num,result_Con=zeros(self.iter_max,con_num);
            else,result_Con=[];end
            if coneq_num,result_Coneq=zeros(self.iter_max,coneq_num);
            else,result_Coneq=[];end
            if vio_num,result_Vio=zeros(self.iter_max,vio_num);
            else,result_Vio=[];end
            con_best=[];coneq_best=[];vio_best=[];
            pop_X=X;pop_Obj=Obj;pop_Vio=Vio; % initial population

            self.dataoptim.iter=self.dataoptim.iter+1;
            search_mode='G'; % 'G' is global search,'L' is local search
            pop_prcs_idx=1;
            while ~self.dataoptim.done
                if search_mode == 'G' && pop_prcs_idx == 1
                    % construct GRNN
                    [self.obj_fcn_srgt,self.con_fcn_srgt,self.Srgt_obj,self.Srgt_con,self.Srgt_coneq]=self.getSrgtFcnGRNN...
                        (X,Obj,Con,Coneq);
                end

                if search_mode == 'G'
                    % global search
                    x_infill=self.searchGlobal(X,Obj,pop_X,low_bou,up_bou,pop_X(pop_prcs_idx,:));
                    obj_pred=self.obj_fcn_srgt(x_infill);
                elseif search_mode == 'L'
                    % local search
                    x_infill=self.searchLocal(X,Obj,Con,Coneq,Vio,...
                        vari_num,low_bou,up_bou,pop_X(pop_prcs_idx,:));
                    obj_pred=self.obj_fcn_srgt(x_infill);
                else
                    error('OptimGLoSADE.optimize: unknown search mode')
                end

                % updata infill point
                [self.datalib,x_infill,obj_infill,con_infill,coneq_infill,vio_infill,repeat_idx]=self.sample(self.datalib,objcon_fcn,x_infill);
                [X,Obj,Con,Coneq,Vio]=self.datalibLoad(self.datalib);

                fprintf('mode:    %c    real_obj:    %f    pred_obj:    %f\n',search_mode,obj_pred,obj_infill);

                % whether updata population
                if vio_infill < pop_Vio(pop_prcs_idx, :) && pop_Vio(pop_prcs_idx, :) > 0
                    pop_X(pop_prcs_idx, :)=x_infill;
                    pop_Obj(pop_prcs_idx,:)=obj_infill;
                    pop_Vio(pop_prcs_idx,:)=vio_infill;
                elseif obj_infill < pop_Obj(pop_prcs_idx, :) && pop_Vio(pop_prcs_idx, :) <= 0 && vio_infill <= 0
                    pop_X(pop_prcs_idx, :)=x_infill;
                    pop_Obj(pop_prcs_idx,:)=obj_infill;
                    pop_Vio(pop_prcs_idx,:)=vio_infill;
                end
                pop_prcs_idx=pop_prcs_idx+1;

                if pop_prcs_idx > self.pop_num
                    switch search_mode
                        case 'L'
                            search_mode='G';
                        case 'G'
                            search_mode='L';
                    end
                    pop_prcs_idx=1;
                end

                % find best result to record
                best_idx=self.datalib.Best_idx(end);
                x_best=X(best_idx,:);
                result_X(self.dataoptim.iter,:)=x_best;
                obj_best=Obj(best_idx,:);
                result_Obj(self.dataoptim.iter,:)=obj_best;
                if con_num,result_Con(self.dataoptim.iter,:)=Con(best_idx,:);end
                if coneq_num,result_Coneq(self.dataoptim.iter,:)=Coneq(best_idx,:);end
                if vio_num,vio_best=Vio(best_idx,:);result_Vio(self.dataoptim.iter,:)=vio_best;end
                self.dataoptim.iter=self.dataoptim.iter+1;

                % information
                if self.FLAG_DRAW_FIGURE && vari_num < 3
                    displaySrgt([],self.Srgt_obj{1},low_bou,up_bou);
                    line(x_infill(1),x_infill(2),obj_infill,'Marker','o','color','r');
                end

                if self.FLAG_INFORMATION
                    fprintf('obj:    %f    vio:    %f    NFE:    %-3d\n',obj_best,vio_best,self.dataoptim.NFE);
                end

                % forced interrupt
                if self.dataoptim.iter > self.iter_max || self.dataoptim.NFE >= self.NFE_max
                    self.dataoptim.done=true;
                end

                % convergence judgment
                if self.FLAG_CONV_JUDGE && self.dataoptim.iter > 2
                    if ( abs((obj_infill-obj_infill_old)/obj_infill_old) < self.obj_tol && ...
                            ((~isempty(vio_infill) && vio_infill == 0) || isempty(vio_infill)) )
                        self.dataoptim.done=true;
                    end
                end
                obj_infill_old=obj_infill;

                % save iteration
                if ~isempty(self.dataoptim_filestr)
                    datalib=self.datalib;
                    dataoptim=self.dataoptim;
                    save(self.dataoptim_filestr,'datalib','dataoptim');
                end
            end

            % cut result
            result_X(self.dataoptim.iter:end,:)=[];
            result_Obj(self.dataoptim.iter:end,:)=[];
            if con_num,result_Con(self.dataoptim.iter:end,:)=[];end
            if coneq_num,result_Coneq(self.dataoptim.iter:end,:)=[];end
            if vio_num,result_Vio(self.dataoptim.iter:end,:)=[];end

            x_best=result_X(end,:);
            obj_best=result_Obj(end,:);
            NFE=self.dataoptim.NFE;
            if ~isempty(result_Con),con_best=result_Con(end,:);end
            if ~isempty(result_Coneq),coneq_best=result_Coneq(end,:);end
            if ~isempty(result_Vio),vio_best=result_Vio(end,:);end

            output.result_x_best=result_X;
            output.result_obj_best=result_Obj;
            output.result_con_best=result_Con;
            output.result_coneq_best=result_Coneq;
            output.result_vio_best=result_Vio;
            output.NFE=self.dataoptim.NFE;
            output.Add_idx=self.dataoptim.Add_idx;
            output.datalib=self.datalib;
            output.dataoptim=self.dataoptim;
        end

        function sampleInit(self,objcon_fcn,vari_num,low_bou,up_bou)
            % initial latin hypercube sample
            %

            % obtain datalib
            if isempty(self.datalib)
                self.datalib=self.datalibGet(vari_num,low_bou,up_bou,self.con_tol,self.datalib_filestr);
            else
                self.datalib.low_bou=low_bou;
                self.datalib.up_bou=up_bou;
                self.datalib.filestr=self.datalib_filestr;
            end

            if size(self.datalib.X,1) < self.sample_num_init
                if isempty(self.X_init)
                    % use latin hypercube method to get enough initial sample x_list
                    sample_num=min(self.sample_num_init-size(self.datalib.X,1),self.NFE_max-self.dataoptim.NFE);
                    self.X_init=lhsdesign(sample_num,vari_num,'iterations',50,'criterion','maximin').*(up_bou-low_bou)+low_bou;
                end

                % updata data lib by x_list
                self.datalib=self.sample(self.datalib,objcon_fcn,self.X_init);
            end

            % detech expensive constraints
            if ~isempty(self.datalib.Vio)
                self.FLAG_CON=true(1);
            else
                self.FLAG_CON=false(1);
            end
        end

        function [datalib,X,Obj,Con,Coneq,Vio,repeat_idx]=sample...
                (self,datalib,objcon_fcn,X_add,cost,fidelity)
            % sample objcon_fcn warpper, will record best sample
            %
            if nargin < 6,fidelity=[];if nargin < 5,cost=[];end,end
            if isempty(fidelity), fidelity=1;end
            if isempty(cost), cost=1;end

            x_num_add=size(X_add,1);
            datalib_idx=zeros(x_num_add,1);
            repeat_idx=zeros(x_num_add,1);

            for x_idx=1:x_num_add
                x_add=X_add(x_idx,:);

                % check add x range
                if isempty(datalib.X)
                    dist=realmax;
                else
                    dist=vecnorm(datalib.X-x_add,2,2);
                end
                if any(dist < self.add_tol)
                    overlap_idx=find(dist < self.add_tol,1);
                    repeat_idx(x_idx)=overlap_idx;
                    datalib_idx(x_idx)=overlap_idx;
                else
                    datalib=self.datalibAdd(datalib,objcon_fcn,x_add);

                    % recode add information
                    self.dataoptim.NFE=self.dataoptim.NFE+cost;
                    self.dataoptim.Add_idx=[self.dataoptim.Add_idx;fidelity,size(datalib.X,1)];
                    datalib_idx(x_idx)=size(datalib.X,1);
                end
            end

            X=datalib.X(datalib_idx,:);
            Obj=datalib.Obj(datalib_idx,:);
            if ~isempty(datalib.Con),Con=datalib.Con(datalib_idx,:);
            else,Con=[];end
            if ~isempty(datalib.Coneq),Coneq=datalib.Coneq(datalib_idx,:);
            else,Coneq=[];end
            if ~isempty(datalib.Vio),Vio=datalib.Vio(datalib_idx,:);
            else,Vio=[];end
        end

    end

    % strategy function
    methods
        function x_infill=searchGlobal(self,X,Obj,pop_X,low_bou,up_bou,x_current)
            % find global infill point function
            %
            if rand < 0.5
                % generate trial population
                pop_trial=DERand1(low_bou,up_bou,pop_X,self.scaling_factor,self.pop_num);
                pop_trial=DECrossover(low_bou,up_bou,pop_X,pop_trial,self.cross_rate);

                obj_pred=self.obj_fcn_srgt(pop_trial);
                con_pred=self.con_fcn_srgt(pop_trial);
                vio_pred=max(max(con_pred-self.con_tol,0),2);

                feasi_idx=find(vio_pred <= 0);
                if isempty(feasi_idx)
                    [~,idx]=min(vio_pred);
                elseif length(feasi_idx)==self.pop_num
                    [~,idx]=min(obj_pred);
                else
                    [~,idx]=min(obj_pred(feasi_idx, :));
                    idx=feasi_idx(idx);
                end

                x_infill=pop_trial(idx(1),:);
            else
                % generate trial population
                pop_trial=DECurrentRand1(low_bou,up_bou,pop_X,self.scaling_factor,self.pop_num,x_current);
                pop_trial=DECrossover(low_bou,up_bou,pop_X,pop_trial,self.cross_rate);

                % select nearest point to construct RBF
                select_num=min(self.RBF_num,size(X,1));
                dist=sum(((x_current-X)./(up_bou-low_bou)).^2,2);
                [~,idx_list]=sort(dist);
                idx_list=idx_list(1:select_num);
                X_RBF=X(idx_list,:);
                Obj_RBF=Obj(idx_list,:);

                % find most uncertain point base on RBF
                srgt_obj=srgtsfRBF(X_RBF,Obj_RBF);
                [~,var_pred]=srgt_obj.predict(pop_trial);

                [~, idx]=max(var_pred);
                x_infill=pop_trial(idx(1), :);
            end

            function X_new=DERand1(low_bou,up_bou,X,F,new_num)
                % generate rand index
                DE_idx=zeros(new_num,3);
                for new_idx=1:new_num
                    DE_idx(new_idx,:)=randperm(size(X,1),3);
                end
                X_new=X(DE_idx(:,1),:)+F*(X(DE_idx(:,2),:)-X(DE_idx(:,3),:));
                X_new=max(X_new,low_bou);
                X_new=min(X_new,up_bou);
            end

            function X_new=DECurrentRand1(low_bou,up_bou,X,F,new_num,x_current)
                % generate rand index
                DE_idx=zeros(new_num,3);
                for new_idx=1:new_num
                    DE_idx(new_idx,:)=randperm(size(X,1),3);
                end
                X_new=x_current+F*(X(DE_idx(:,1),:)-x_current)+F*(X(DE_idx(:,2),:)-X(DE_idx(:,3),:));
                X_new=max(X_new,low_bou);
                X_new=min(X_new,up_bou);
            end

            function X=DECrossover(low_bou,up_bou,X,V,C_R)
                crs_idx=find(rand(size(X)) < C_R);
                X(crs_idx)=V(crs_idx);
                X=max(X,low_bou);
                X=min(X,up_bou);
            end

        end

        function x_infill=searchLocal(self,X,Obj,Con,Coneq,Vio,...
                vari_num,low_bou,up_bou,x_current)
            % find local infill point function
            %
            fmincon_option=optimoptions('fmincon','Display','none','Algorithm','sqp','ConstraintTolerance',self.con_tol);
            bou_min=0.001;

            % sort X potential by Obj
            [X,Obj,Con,Coneq,Vio]=self.rankData(X,Obj,Con,Coneq,Vio);

            % select nearest point to construct RBF
            select_num=min(self.RBF_num,size(X,1));
            dist=sum(((x_current-X)./(up_bou-low_bou)).^2,2);
            [~,idx_list]=sort(dist);
            idx_list=idx_list(1:select_num);
            X_RBF=X(idx_list,:);
            Obj_RBF=Obj(idx_list,:);
            if ~isempty(Con),Con_RBF=Con(idx_list,:);
            else,Con_RBF=[];end
            if ~isempty(Coneq),Coneq_RBF=Coneq(idx_list,:);
            else,Coneq_RBF=[];end

            % modify
            % get RBF model and function
            [obj_fcn_local,con_fcn_local,self.Srgt_obj_local,self.Srgt_con_local,self.Srgt_coneq_local]=self.getSrgtFcnRBF...
                (X_RBF,Obj_RBF,Con_RBF,Coneq_RBF);

            % get local infill point
            % obtian total constraint function
            low_bou_local=min(X_RBF,[],1);
            up_bou_local=max(X_RBF,[],1);

            % boundary protect
            bou_nomlz=(up_bou_local-low_bou_local)./(up_bou-low_bou)/2;
            bou_nomlz=max(bou_nomlz,bou_min);
            bou_center=(low_bou_local+up_bou_local)/2;
            low_bou_local=bou_center-bou_nomlz.*(up_bou-low_bou);
            up_bou_local=bou_center+bou_nomlz.*(up_bou-low_bou);
            low_bou_local=max(low_bou_local,low_bou);
            up_bou_local=min(up_bou_local,up_bou);

            [x_infill,~,exit_flag,output_fmincon]=fmincon(obj_fcn_local,x_current,[],[],[],[],...
                low_bou_local,up_bou_local,con_fcn_local,fmincon_option);

            x_infill=max(x_infill,low_bou);
            x_infill=min(x_infill,up_bou);
        end

        function BestX = FeasiRuleGRNN(TrialU, GRNNnet, eta)
            % find best feasible point base on GRNN
            %
            rho = size(TrialU,1);

            TrialU_Obj =  sim(GRNNnet.Obj,TrialU')';
            TrialU_Con = sim(GRNNnet.Con,TrialU')';

            TrialU_Vio = sum(max(TrialU_Con,0),2);

            feasi_idx = find(TrialU_Vio<=eta);
            if length(feasi_idx) == 0
                [~,idx] = min(TrialU_Vio);
            elseif length(feasi_idx) == rho
                [~,idx] = min(TrialU_Obj);
            else
                [~,idx] = min(TrialU_Obj(feasi_idx, :));
                idx = feasi_idx(idx);
            end

            BestX = TrialU(idx,:);
        end

    end

    % common function
    methods(Static)
        function [obj_fcn_srgt,con_fcn_srgt,Srgt_obj,Srgt_con,Srgt_coneq]=getSrgtFcnRBF...
                (x_list,obj_list,con_list,coneq_list)
            % generate surrogate function of objective and constraints
            %
            % output:
            % obj_fcn_srgt(output is obj_pred),...
            % con_fcn_srgt(output is con_pred, coneq_pred)
            %

            % generate obj surrogate
            Srgt_obj=cell(size(obj_list,2),1);
            for obj_idx=1:size(obj_list,2)
                Srgt_obj{obj_idx}=srgtsfRBF(x_list,obj_list(:,obj_idx));
            end

            % generate con surrogate
            if ~isempty(con_list)
                Srgt_con=cell(size(con_list,2),1);
                for con_idx=1:size(con_list,2)
                    Srgt_con{con_idx}=srgtsfRBF(x_list,con_list(:,con_idx));
                end
            else
                Srgt_con=[];
            end

            % generate coneq surrogate
            if ~isempty(coneq_list)
                Srgt_coneq=cell(size(coneq_list,2),1);
                for coneq_idx=1:size(coneq_list,2)
                    Srgt_coneq{coneq_idx}=srgtsfRBF(x_list,coneq_list(:,coneq_idx));
                end
            else
                Srgt_coneq=[];
            end

            obj_fcn_srgt=@(X_pred) objFcnSurr(X_pred,Srgt_obj);
            if isempty(Srgt_con) && isempty(Srgt_coneq)
                con_fcn_srgt=[];
            else
                con_fcn_srgt=@(X_pred) conFcnSurr(X_pred,Srgt_con,Srgt_coneq);
            end

            function [Obj_pred]=objFcnSurr(X_pred,Srgt_obj)
                % connect all predict obj
                %
                Obj_pred=zeros(size(X_pred,1),length(Srgt_obj));
                for con_i=1:length(Srgt_obj)
                    [Obj_pred(:,con_i)]=Srgt_obj{con_i}.predict(X_pred);
                end
            end

            function [Con_pred,Coneq_pred]=conFcnSurr(X_pred,Srgt_con,Srgt_coneq)
                % connect all predict con and coneq
                %
                if isempty(Srgt_con)
                    Con_pred=[];
                else
                    Con_pred=zeros(size(X_pred,1),length(Srgt_con));
                    for con_i=1:length(Srgt_con)
                        [Con_pred(:,con_i)]=Srgt_con{con_i}.predict(X_pred);
                    end
                end
                if isempty(Srgt_coneq)
                    Coneq_pred=[];
                else
                    Coneq_pred=zeros(size(X_pred,1),length(Srgt_coneq));
                    for coneq_i=1:length(Srgt_coneq)
                        [Coneq_pred(:,coneq_i)]=Srgt_coneq{coneq_i}.predict(X_pred);
                    end
                end
            end
        end

        function [obj_fcn_srgt,con_fcn_srgt,Srgt_obj,Srgt_con,Srgt_coneq]=getSrgtFcnGRNN...
                (x_list,obj_list,con_list,coneq_list)
            % generate surrogate function of objective and constraints
            %
            % output:
            % obj_fcn_srgt(output is obj_pred),...
            % con_fcn_srgt(output is con_pred, coneq_pred)
            %

            % generate obj surrogate
            Srgt_obj=newgrnn(x_list',obj_list');

            % generate con surrogate
            if ~isempty(con_list)
                Srgt_con=newgrnn(x_list',con_list');
            else
                Srgt_con=[];
            end

            % generate coneq surrogate
            if ~isempty(coneq_list)
                Srgt_coneq=newgrnn(x_list',coneq_list');
            else
                Srgt_coneq=[];
            end

            obj_fcn_srgt=@(X_pred) objFcnSurr(X_pred,Srgt_obj);
            if isempty(Srgt_con) && isempty(Srgt_coneq)
                con_fcn_srgt=[];
            else
                con_fcn_srgt=@(X_pred) conFcnSurr(X_pred,Srgt_con,Srgt_coneq);
            end

            function [Obj_pred]=objFcnSurr(X_pred,Srgt_obj)
                % connect all predict obj
                %
                Obj_pred=sim(Srgt_obj,X_pred')';
            end

            function [Con_pred,Coneq_pred]=conFcnSurr(X_pred,Srgt_con,Srgt_coneq)
                % connect all predict con and coneq
                %
                if isempty(Srgt_con)
                    Con_pred=[];
                else
                    Con_pred=sim(Srgt_con,X_pred')';
                end
                if isempty(Srgt_coneq)
                    Coneq_pred=[];
                else
                    Con_pred=sim(Srgt_coneq,X_pred')';
                end
            end
        end

        function [X,Obj,Con,Coneq,Vio]=rankData(X,Obj,Con,Coneq,Vio)
            % rank data base on feasibility rule
            % infeasible is rank by sum of constraint
            %
            [x_num,~]=size(X);

            % rank data
            % infeasible data rank by violation, feasible data rank by obj
            if ~isempty(Vio),Bool_feas=Vio <= 0;
            else,Bool_feas=true(x_num,1);end
            all=1:x_num;
            feasi_index_list=all(Bool_feas);
            infeasi_index_list=all(~Bool_feas);
            [~,sort_idx]=sort(Obj(feasi_index_list));
            feasi_index_list=feasi_index_list(sort_idx);
            [~,sort_idx]=sort(Vio(infeasi_index_list));
            infeasi_index_list=infeasi_index_list(sort_idx);
            sort_idx=[feasi_index_list,infeasi_index_list];

            % rank by index_list
            X=X(sort_idx,:);
            Obj=Obj(sort_idx);
            if ~isempty(Con),Con=Con(sort_idx,:);end
            if ~isempty(Coneq),Coneq=Coneq(sort_idx,:);end
            if ~isempty(Vio),Vio=Vio(sort_idx);end
        end

    end

    % data library function
    methods(Static)
        function datalib=datalibGet(vari_num,low_bou,up_bou,con_tol,datalib_filestr)
            % generate data library object
            %
            if nargin < 5
                datalib_filestr=[];
                if nargin < 4 || isempty(con_tol)
                    con_tol=0;
                end
            end

            datalib=struct();
            datalib.vari_num=vari_num;
            datalib.low_bou=low_bou;
            datalib.up_bou=up_bou;
            datalib.con_tol=con_tol;
            datalib.filestr=datalib_filestr;

            datalib.X=[];
            datalib.Obj=[];
            datalib.Con=[];
            datalib.Coneq=[];
            datalib.Vio=[];
            datalib.Best_idx=[];
        end

        function [datalib,x,obj,con,coneq,vio]=datalibAdd(datalib,objcon_fcn,x,obj,con,coneq)
            % add new x into data lib
            %
            if ~isempty(objcon_fcn)
                [obj,con,coneq]=objcon_fcn(x); % eval value
            end
            vio=[];

            % calculate vio
            if ~isempty(con),vio=[vio,max(max(con-datalib.con_tol,0),[],2)];end
            if ~isempty(coneq),vio=[vio,max(max(abs(coneq)-datalib.con_tol,0),[],2)];end
            vio=max(vio,[],2);

            datalib.X=[datalib.X;x];
            datalib.Obj=[datalib.Obj;obj];
            datalib.Con=[datalib.Con;con];
            datalib.Coneq=[datalib.Coneq;coneq];
            datalib.Vio=[datalib.Vio;vio];

            % recode best index of data library
            Best_idx=datalib.Best_idx;
            if isempty(Best_idx)
                Best_idx=1;
            else
                if isempty(datalib.Vio)
                    if obj <= datalib.Obj(Best_idx(end))
                        Best_idx=[Best_idx;size(datalib.X,1)];
                    else
                        Best_idx=[Best_idx;Best_idx(end)];
                    end
                else
                    if vio < datalib.Vio(Best_idx(end)) || (obj <= datalib.Obj(Best_idx(end)) && vio == 0)
                        Best_idx=[Best_idx;size(datalib.X,1)];
                    else
                        Best_idx=[Best_idx;Best_idx(end)];
                    end
                end
            end
            datalib.Best_idx=Best_idx;

            if ~isempty(datalib.filestr),save(datalib.filestr,'datalib');end
        end

        function [X,Obj,Con,Coneq,Vio]=datalibLoad(datalib,low_bou,up_bou)
            % updata data to exist data lib
            %
            if nargin < 3,up_bou=[];if nargin < 2,low_bou=[];end,end
            if isempty(up_bou), up_bou=realmax;end
            if isempty(low_bou), low_bou=-realmax;end

            Bool_inside=[datalib.X  >= low_bou,datalib.X <= up_bou];
            Bool_inside=all(Bool_inside,2);

            X=datalib.X(Bool_inside,:);
            Obj=datalib.Obj(Bool_inside,:);
            if ~isempty(datalib.Con),Con=datalib.Con(Bool_inside,:);
            else,Con=[];end
            if ~isempty(datalib.Coneq),Coneq=datalib.Coneq(Bool_inside,:);
            else,Coneq=[];end
            if ~isempty(datalib.Vio),Vio=datalib.Vio(Bool_inside,:);
            else,Vio=[];end
        end
    end
end

%% surrogate function

function model_RBF=srgtsfRBF(X,Y,basis_fcn)
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
