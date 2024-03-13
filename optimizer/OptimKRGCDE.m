classdef OptimKRGCDE < handle
    % KRG-CDE optimization algorithm
    %
    % referance: [1] 叶年辉,龙腾,武宇飞,等.
    % 基于Kriging代理模型的约束差分进化算法 [J]. 航空学报,2021,42(6): 13.
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
        obj_torl;
        con_torl;

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

        add_torl=1000*eps; % surrogate add point protect range
        X_init=[];

        % hyper parameter
        sample_num_init=[];
        pop_num=[];
        RBF_num=[];

        % differ evoluation parameter
        scaling_factor=0.8;
        cross_rate=0.8;

        KRG_option=struct('simplify_flag',true);
    end

    % main function
    methods
        function self=OptimKRGCDE(NFE_max,iter_max,obj_torl,con_torl)
            % initialize optimization
            %
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

            if isempty(con_torl)
                con_torl=1e-3;
            end
            if isempty(obj_torl)
                obj_torl=1e-6;
            end

            self.NFE_max=NFE_max;
            self.iter_max=iter_max;
            self.obj_torl=obj_torl;
            self.con_torl=con_torl;
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
                    if ~contains(prob_field,'objcon_fcn'), error('OptimKRGCDE.optimize: input problem lack objcon_fcn'); end
                    objcon_fcn=problem.objcon_fcn;
                    if ~contains(prob_field,'vari_num'), error('OptimKRGCDE.optimize: input problem lack vari_num'); end
                    if ~contains(prob_field,'low_bou'), error('OptimKRGCDE.optimize: input problem lack low_bou'); end
                    if ~contains(prob_field,'up_bou'), error('OptimKRGCDE.optimize: input problem lack up_bou'); end
                else
                    prob_method=methods(problem);
                    if ~contains(prob_method,'objcon_fcn'), error('OptimKRGCDE.optimize: input problem lack objcon_fcn'); end
                    objcon_fcn=@(x) problem.objcon_fcn(x);
                    prob_pro=properties(problem);
                    if ~contains(prob_pro,'vari_num'), error('OptimKRGCDE.optimize: input problem lack vari_num'); end
                    if ~contains(prob_pro,'low_bou'), error('OptimKRGCDE.optimize: input problem lack low_bou'); end
                    if ~contains(prob_pro,'up_bou'), error('OptimKRGCDE.optimize: input problem lack up_bou'); end
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

            self.Srgt_obj=repmat({self.KRG_option},obj_num,1);
            self.Srgt_con=repmat({self.KRG_option},con_num,1);
            self.Srgt_coneq=repmat({self.KRG_option},coneq_num,1);

            self.dataoptim.iter=self.dataoptim.iter+1;
            next_search_mode='G'; % 'G' is global search,'l' is local search
            while ~self.dataoptim.done
                search_mode=next_search_mode;

                if search_mode == 'G'
                    % global search
                    x_infill=self.searchGlobal(X,Obj,Con,Coneq,Vio,...
                        vari_num,low_bou,up_bou,self.pop_num);
                else
                    % local search
                    x_infill=self.searchLocal(X,Obj,Con,Coneq,Vio,...
                        vari_num,low_bou,up_bou,self.pop_num,self.RBF_num);
                end

                % updata infill point
                [self.datalib,x_infill,obj_infill,con_infill,coneq_infill,vio_infill,repeat_idx]=self.sample(self.datalib,objcon_fcn,x_infill);
                [X,Obj,Con,Coneq,Vio]=self.datalibLoad(self.datalib);

                improve_flag=false;
                if self.FLAG_CON
                    min_vio=min(Vio);Bool_feas=Vio == 0;
                    min_obj=min(Obj([Bool_feas(1:end-1);false]),[],1);

                    % if all point is infeasible,violation of point infilled is
                    % less than min violation of all point means improve.if
                    % feasible point exist,obj of point infilled is less than min
                    % obj means improve
                    if vio_infill < min_vio
                        if ~isempty(min_obj)
                            if obj_infill < min_obj
                                % improve, continue local search
                                improve_flag=true;
                            end
                        else
                            % improve, continue local search
                            improve_flag=true;
                        end
                    end
                else
                    min_obj=min(Obj(1:end-1));

                    % obj of point infilled is less than min obj means improve
                    if obj_infill < min_obj
                        % imporve, continue local search
                        improve_flag=true;
                    end
                end

                % if no imporve begin local search or global search
                if ~improve_flag
                    switch next_search_mode
                        case 'L'
                            next_search_mode='G';
                        case 'G'
                            next_search_mode='L';
                    end
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
                    surrogateVisualize(self.Srgt_obj{1},low_bou,up_bou);
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
                    if ( abs((obj_infill-obj_infill_old)/obj_infill_old) < self.obj_torl && ...
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
                self.datalib=self.datalibGet(vari_num,low_bou,up_bou,self.con_torl,self.datalib_filestr);
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
            % find best result to record
            %
            if nargin < 6, fidelity=[];if nargin < 5, cost=[];end,end
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
                if any(dist < self.add_torl)
                    overlap_idx=find(dist < self.add_torl,1);
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
        function x_infill=searchGlobal(self,X,Obj,Con,Coneq,Vio,...
                vari_num,low_bou,up_bou,pop_num)
            % find global infill point function
            %

            % step 5
            % rank x_list data
            X_rank=self.rankData(X,Obj,Con,Coneq,Vio);

            % step 6
            % only the first population_number will be use
            X_best_pop=X_rank(1:pop_num,:);

            % differ evolution mutations
            X_new_R1=DERand...
                (low_bou,up_bou,X_best_pop,self.scaling_factor,pop_num,1);
            X_new_R2=DERand...
                (low_bou,up_bou,X_best_pop,self.scaling_factor,pop_num,2);
            X_new_CR=DECurrentRand...
                (low_bou,up_bou,X_best_pop,self.scaling_factor);
            X_new_CB=DECurrentBest...
                (low_bou,up_bou,X_best_pop,self.scaling_factor,1);

            % differ evolution crossover
            X_new_R1=DECrossover...
                (low_bou,up_bou,X_best_pop,X_new_R1,self.cross_rate);
            X_new_R2=DECrossover...
                (low_bou,up_bou,X_best_pop,X_new_R2,self.cross_rate);
            X_new_CR=DECrossover...
                (low_bou,up_bou,X_best_pop,X_new_CR,self.cross_rate);
            X_new_CB=DECrossover...
                (low_bou,up_bou,X_best_pop,X_new_CB,self.cross_rate);

            % find global infill point base kriging model from offspring X
            X_DE=[X_new_R1;X_new_R2;X_new_CR;X_new_CB];

            % step 4
            % updata kriging model and function
            [self.obj_fcn_srgt,self.con_fcn_srgt,...
                self.Srgt_obj,self.Srgt_con,self.Srgt_coneq]=self.getSrgtFcnKRG...
                (X,Obj,Con,Coneq,self.Srgt_obj,self.Srgt_con,self.Srgt_coneq);

            % evaluate each x_offspring obj and constraints
            [obj_pred_DE_list,obj_var_DE_list]=self.obj_fcn_srgt(X_DE);
            if self.FLAG_CON
                if ~isempty(self.con_fcn_srgt)
                    [con_pred_DE_list,coneq_pred_DE_list,con_var_DE_list,coneq_var_DE_list]=self.con_fcn_srgt(X_DE);
                end

                vio_DE_list=zeros(4*pop_num,1);
                if ~isempty(Con)
                    vio_DE_list=vio_DE_list+sum(max(con_pred_DE_list-self.con_torl,0),2);
                end
                if ~isempty(Coneq)
                    vio_DE_list=vio_DE_list+sum((abs(coneq_pred_DE_list)-self.con_torl),2);
                end

                feasi_boolean_DE_list=vio_DE_list == 0;
            else
                feasi_boolean_DE_list=true(ones(1,4*pop_num));
            end

            % if have feasiable_idx_list,only use feasiable to choose
            if all(~feasi_boolean_DE_list)
                % base on constaints improve select global infill
                % lack process of equal constraints
                con_nomlz_base=max(min(Con,[],1),0);
                Con_impove_prob=sum(...
                    normcdf((con_nomlz_base-con_pred_DE_list)./sqrt(con_var_DE_list)),2);
                [~,con_best_idx]=max(Con_impove_prob);
                con_best_idx=con_best_idx(1);
                x_infill=X_DE(con_best_idx,:);
            else
                % base on fitness DE point to select global infill
                if self.FLAG_CON
                    X_DE=X_DE(feasi_boolean_DE_list,:);
                    obj_pred_DE_list=obj_pred_DE_list(feasi_boolean_DE_list);
                    obj_var_DE_list=obj_var_DE_list(feasi_boolean_DE_list);
                end

                obj_DE_min=min(obj_pred_DE_list,[],1);
                obj_DE_max=max(obj_pred_DE_list,[],1);

                DE_fitness_list=-(obj_pred_DE_list-obj_DE_min)/(obj_DE_max-obj_DE_min)+...
                    obj_var_DE_list;
                [~,fitness_best_idx]=max(DE_fitness_list);
                fitness_best_idx=fitness_best_idx(1);
                x_infill=X_DE(fitness_best_idx,:);
            end

            function X_new = DERand(low_bou,up_bou,X,F,x_num,rand_num)
                [x_number__,variable_number__] = size(X);
                X_new = zeros(x_num,variable_number__);
                for x_idx__ = 1:x_num
                    idx__ = randi(x_number__,2*rand_num+1,1);
                    X_new(x_idx__,:) = X(idx__(1),:);
                    for rand_idx__ = 1:rand_num
                        X_new(x_idx__,:) = X_new(x_idx__,:)+...
                            F*(X(idx__(2*rand_idx__),:)-X(idx__(2*rand_idx__+1),:));
                        X_new(x_idx__,:) = max(X_new(x_idx__,:),low_bou);
                        X_new(x_idx__,:) = min(X_new(x_idx__,:),up_bou);
                    end
                end
            end

            function X_new = DECurrentRand(low_bou,up_bou,X,F)
                [x_number__,variable_number__] = size(X);
                X_new = zeros(x_number__,variable_number__);
                for x_idx__ = 1:x_number__
                    idx__ = randi(x_number__,3,1);
                    X_new(x_idx__,:) = X(x_idx__,:)+...
                        F*(X(idx__(1),:)-X(x_idx__,:)+...
                        X(idx__(2),:)-X(idx__(3),:));
                    X_new(x_idx__,:) = max(X_new(x_idx__,:),low_bou);
                    X_new(x_idx__,:) = min(X_new(x_idx__,:),up_bou);
                end
            end

            function X_new = DECurrentBest(low_bou,up_bou,X,F,x_best_idx)
                [x_number__,variable_number__] = size(X);
                X_new = zeros(x_number__,variable_number__);
                for x_idx__ = 1:x_number__
                    idx__ = randi(x_number__,2,1);
                    X_new(x_idx__,:) = X(x_idx__,:)+...
                        F*(X(x_best_idx,:)-X(x_idx__,:)+...
                        X(idx__(1),:)-X(idx__(2),:));
                    X_new(x_idx__,:) = max(X_new(x_idx__,:),low_bou);
                    X_new(x_idx__,:) = min(X_new(x_idx__,:),up_bou);
                end
            end

            function X_new = DECrossover(low_bou,up_bou,X,V,C_R)
                [x_number__,variable_number__] = size(X);
                X_new = X;
                rand_number = rand(x_number__,variable_number__);
                idx__ = find(rand_number < C_R);
                X_new(idx__) = V(idx__);
                for x_idx__ = 1:x_number__
                    X_new(x_idx__,:) = max(X_new(x_idx__,:),low_bou);
                    X_new(x_idx__,:) = min(X_new(x_idx__,:),up_bou);
                end
            end

        end

        function x_infill=searchLocal(self,X,Obj,Con,Coneq,Vio,...
                vari_num,low_bou,up_bou,pop_num,RBF_num)
            % find local infill point function
            %
            fmincon_option=optimoptions('fmincon','Display','none','Algorithm','sqp','ConstraintTolerance',self.con_torl);
            bou_min=0.001;

            % sort X potential by Obj
            [X,Obj,Con,Coneq,Vio]=self.rankData(X,Obj,Con,Coneq,Vio);

            % step 8
            % rand select initial local point from x_list
            x_idx=randi(pop_num);
            x_init=X(x_idx,:);

            % select nearest point to construct RBF
            RBF_num=min(RBF_num,size(X,1));
            dist=sum(((x_init-X)./(up_bou-low_bou)).^2,2);
            [~,idx_list]=sort(dist);
            idx_list=idx_list(1:RBF_num);
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

            [x_infill,~,exit_flag,output_fmincon]=fmincon(obj_fcn_local,x_init,[],[],[],[],...
                low_bou_local,up_bou_local,con_fcn_local,fmincon_option);

            x_infill=max(x_infill,low_bou);
            x_infill=min(x_infill,up_bou);
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
                Srgt_obj{obj_idx}=srgtRBF(x_list,obj_list(:,obj_idx));
            end

            % generate con surrogate
            if ~isempty(con_list)
                Srgt_con=cell(size(con_list,2),1);
                for con_idx=1:size(con_list,2)
                    Srgt_con{con_idx}=srgtRBF(x_list,con_list(:,con_idx));
                end
            else
                Srgt_con=[];
            end

            % generate coneq surrogate
            if ~isempty(coneq_list)
                Srgt_coneq=cell(size(coneq_list,2),1);
                for coneq_idx=1:size(coneq_list,2)
                    Srgt_coneq{coneq_idx}=srgtRBF(x_list,coneq_list(:,coneq_idx));
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

                if any(isnan([Obj_pred]))
                    disp('nan');
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

                if any(isnan([Con_pred,Coneq_pred]))
                    disp('nan');
                end
            end
        end

        function [obj_fcn_srgt,con_fcn_srgt,Srgt_obj,Srgt_con,Srgt_coneq]=getSrgtFcnKRG...
                (x_list,obj_list,con_list,coneq_list,Srgt_obj,Srgt_con,Srgt_coneq)
            % generate surrogate function of objective and constraints
            %
            % output:
            % obj_fcn_srgt(output is obj_pred),...
            % con_fcn_srgt(output is con_pred, coneq_pred)
            %
            if nargin < 7
                Srgt_coneq=[];
                if nargin < 6
                    Srgt_con=[];
                    if nargin < 5
                        Srgt_obj=[];
                    end
                end
            end
            
            % generate obj surrogate
            if isempty(Srgt_obj),Srgt_obj=cell(size(obj_list,2),1);end
            for obj_idx=1:size(obj_list,2)
                Srgt_obj{obj_idx}=srgtKRG(x_list,obj_list(:,obj_idx),Srgt_obj{obj_idx});
            end

            % generate con surrogate
            if ~isempty(con_list)
                if isempty(Srgt_con),Srgt_con=cell(size(con_list,2),1);end
                for con_idx=1:size(con_list,2)
                    Srgt_con{con_idx}=srgtKRG(x_list,con_list(:,con_idx),Srgt_con{con_idx});
                end
            else
                Srgt_con=[];
            end

            % generate coneq surrogate
            if ~isempty(coneq_list)
                if isempty(Srgt_coneq),Srgt_coneq=cell(size(coneq_list,2),1);end
                for coneq_idx=1:size(coneq_list,2)
                    Srgt_coneq{coneq_idx}=srgtKRG(x_list,coneq_list(:,coneq_idx),Srgt_coneq{coneq_idx});
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

            function [Obj_pred,Obj_var]=objFcnSurr(X_pred,Srgt_obj)
                % connect all predict obj
                %
                Obj_pred=zeros(size(X_pred,1),length(Srgt_obj));
                Obj_var=zeros(size(X_pred,1),length(Srgt_obj));
                for con_i=1:length(Srgt_obj)
                    [Obj_pred(:,con_i),Obj_var(:,con_i)]=Srgt_obj{con_i}.predict(X_pred);
                end
            end

            function [Con_pred,Coneq_pred,Con_var,Coneq_var]=conFcnSurr(X_pred,Srgt_con,Srgt_coneq)
                % connect all predict con and coneq
                %
                if isempty(Srgt_con)
                    Con_pred=[];
                    Con_var=[];
                else
                    Con_pred=zeros(size(X_pred,1),length(Srgt_con));
                    Con_var=zeros(size(X_pred,1),length(Srgt_con));
                    for con_i=1:length(Srgt_con)
                        [Con_pred(:,con_i),Con_var(:,con_i)]=Srgt_con{con_i}.predict(X_pred);
                    end
                end
                if isempty(Srgt_coneq)
                    Coneq_pred=[];
                    Coneq_var=[];
                else
                    Coneq_pred=zeros(size(X_pred,1),length(Srgt_coneq));
                    Coneq_var=zeros(size(X_pred,1),length(Srgt_coneq));
                    for coneq_i=1:length(Srgt_coneq)
                        [Coneq_pred(:,coneq_i),Coneq_var(:,coneq_i)]=Srgt_coneq{coneq_i}.predict(X_pred);
                    end
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
        function datalib=datalibGet(vari_num,low_bou,up_bou,con_torl,datalib_filestr)
            % generate data library object
            %
            if nargin < 5
                datalib_filestr=[];
                if nargin < 4 || isempty(con_torl)
                    con_torl=0;
                end
            end

            datalib=struct();
            datalib.vari_num=vari_num;
            datalib.low_bou=low_bou;
            datalib.up_bou=up_bou;
            datalib.con_torl=con_torl;
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
            if ~isempty(con),vio=[vio,max(max(con-datalib.con_torl,0),[],2)];end
            if ~isempty(coneq),vio=[vio,max(abs(coneq-datalib.con_torl),[],2)];end
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
