classdef OptimSKO < handle
    % Kriging model base efficient global optimization
    % include MSP, EI, PI, MSE, LCB infill criteria
    %
    % referance:
    % [1] HAN Zhonghua. Kriging surrogate model and its application to
    % design optimization: A review of recent progress[J]. ACTA
    % AERONAUTICAET ASTRONAUTICA SINICA, 2016, 37(11): 3197-3225.
    %
    % Copyright 2023 Adel
    %
    % abbreviation:
    % obj: objective, con: constraint, iter: iteration, torl: torlance
    % fcn: function, srgt: surrogate
    % lib: library, init: initial, rst: restart, potl: potential
    %

    % basic parameter
    properties
        predict;
        obj_fcn_srgt;
        con_fcn_srgt;

        Srgt_obj;
        Srgt_con;
        Srgt_coneq;

        option_optim;

        FLAG_CON;
        FLAG_MULTI_OBJ;
        FLAG_MULTI_FIDELITY;
        FLAG_DiISCRETE_VARI;

        datalib;

        NFE_max;
        iter_max;
        obj_torl;
        con_torl;
        NFE;
        Add_idx;

        FLAG_DRAW_FIGURE=0; % whether draw data
        FLAG_INFORMATION=1; % whether print data
        FLAG_CONV_JUDGE=0; % whether judgment convergence
    end

    % main function
    methods
        function self=OptimSKO(NFE_max,iter_max,obj_torl,con_torl)
            % initialize self
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

            if isempty(con_torl),con_torl=1e-3;end
            if isempty(obj_torl),obj_torl=1e-6;end

            self.NFE_max=NFE_max;
            self.iter_max=iter_max;
            self.con_torl=con_torl;
            self.obj_torl=obj_torl;

            % surrogate information
            option_optim.KRG_option=struct();
            option_optim.adapt_hyp=false;

            % strategy parameter option
            option_optim.sample_num_init=[];

            % optimize process option
            option_optim.datalib_filestr=''; % datalib save mat name
            option_optim.dataoptim_filestr=''; % optimize save mat name
            option_optim.nomlz_value=100; % max obj when normalize obj,con,coneq
            option_optim.add_torl=1000*eps; % surrogate add point protect range
            option_optim.X_init=[]; % initial sample point
            option_optim.criteria='EI'; % infill criteria
            option_optim.constraint='auto'; % constraint process method
            option_optim.fmincon_option=optimoptions...
                ('fmincon','Display','none','Algorithm','sqp','ConstraintTolerance',0,'FiniteDifferenceStepSize',1e-5);
            option_optim.ga_option=optimoptions...
                ('ga','Display','none','HybridFcn','fmincon','ConstraintTolerance',0,'FunctionTolerance',1e-3);
            option_optim.expert_fcn=[]; % function handle, @(x)

            self.option_optim=option_optim;
        end

        function [x_best,obj_best,NFE,output,con_best,coneq_best,vio_best]=optimize(self,varargin)
            % optimize driver
            %

            % step 1, initialize problem
            if length(varargin) == 1
                % input is struct or object
                problem=varargin{1};
                if isstruct(problem)
                    prob_field=fieldnames(problem);
                    if ~contains(prob_field,'objcon_fcn'), error('OptimSKO.optimize: input problem lack objcon_fcn'); end
                    objcon_fcn=problem.objcon_fcn;
                    if ~contains(prob_field,'vari_num'), error('OptimSKO.optimize: input problem lack vari_num'); end
                    if ~contains(prob_field,'low_bou'), error('OptimSKO.optimize: input problem lack low_bou'); end
                    if ~contains(prob_field,'up_bou'), error('OptimSKO.optimize: input problem lack up_bou'); end
                else
                    prob_method=methods(problem);
                    if ~contains(prob_method,'objcon_fcn'), error('OptimSKO.optimize: input problem lack objcon_fcn'); end
                    objcon_fcn=@(x) problem.objcon_fcn(x);
                    prob_pro=properties(problem);
                    if ~contains(prob_pro,'vari_num'), error('OptimSKO.optimize: input problem lack vari_num'); end
                    if ~contains(prob_pro,'low_bou'), error('OptimSKO.optimize: input problem lack low_bou'); end
                    if ~contains(prob_pro,'up_bou'), error('OptimSKO.optimize: input problem lack up_bou'); end
                end
                vari_num=problem.vari_num;
                low_bou=problem.low_bou;
                up_bou=problem.up_bou;
            else
                % multi input
                varargin=[varargin,repmat({[]},1,4-length(varargin))];
                [objcon_fcn,vari_num,low_bou,up_bou]=varargin{:};
            end

            sample_num_init=self.option_optim.sample_num_init;
            if isempty(sample_num_init),sample_num_init=6*vari_num+3;end
            self.option_optim.sample_num_init=sample_num_init;

            con_best=[];coneq_best=[];vio_best=[];
            self.NFE=0;
            self.Add_idx=[];

            % step 2, latin hypercube sample
            self.sampleInit(objcon_fcn,vari_num,low_bou,up_bou,sample_num_init);

            % step 3, adaptive sample
            [result_X,result_Obj,result_Con,result_Coneq,result_Vio,result_NFE]=self.sampleAdapt...
                (objcon_fcn,vari_num,low_bou,up_bou);

            x_best=result_X(end,:);
            obj_best=result_Obj(end,:);
            NFE=self.NFE;
            if ~isempty(result_Con),con_best=result_Con(end,:);end
            if ~isempty(result_Coneq),coneq_best=result_Coneq(end,:);end
            if ~isempty(result_Vio),vio_best=result_Vio(end,:);end

            output.result_NFE=result_NFE;
            output.result_x_best=result_X;
            output.result_obj_best=result_Obj;
            output.result_con_best=result_Con;
            output.result_coneq_best=result_Coneq;
            output.result_vio_best=result_Vio;
            output.NFE=self.NFE;
            output.Add_idx=self.Add_idx;
            output.datalib=self.datalib;
        end

        function sampleInit(self,objcon_fcn,vari_num,low_bou,up_bou,sample_num_init)
            % initial latin hypercube sample
            %
            X_init=self.option_optim.X_init;
            datalib_filestr=self.option_optim.datalib_filestr;

            % obtain datalib
            if isempty(self.datalib)
                self.datalib=self.datalibGet(vari_num,low_bou,up_bou,self.con_torl,datalib_filestr);
            end

            if size(self.datalib.X,1) < sample_num_init
                if isempty(X_init)
                    % use latin hypercube method to get enough initial sample x_list
                    sample_num=min(sample_num_init-size(self.datalib.X,1),self.NFE_max-self.NFE);
                    X_init=lhsdesign(sample_num,vari_num,'iterations',50,'criterion','maximin').*(up_bou-low_bou)+low_bou;
                end

                % updata data lib by x_list
                self.datalib=self.sample(self.datalib,objcon_fcn,X_init);
            end

            % detech expensive constraints
            if ~isempty(self.datalib.Vio)
                self.FLAG_CON=true(1);
            else
                self.FLAG_CON=false(1);
            end
        end

        function [result_X,result_Obj,result_Con,result_Coneq,result_Vio,result_NFE]=sampleAdapt...
                (self,objcon_fcn,vari_num,low_bou,up_bou)
            % adapt sample to optimize best point
            %

            model_option=self.option_optim.KRG_option;

            % optimize process option
            dataoptim_filestr=self.option_optim.dataoptim_filestr;
            nomlz_value=self.option_optim.nomlz_value;

            iter=0;done=false;

            [X,Obj,Con,Coneq,Vio]=self.datalibLoad(self.datalib,low_bou,up_bou);
            obj_num=size(Obj,2);con_num=size(Con,2);coneq_num=size(Coneq,2);vio_num=size(Vio,2);
            result_NFE=zeros(self.iter_max,1);
            result_X=zeros(self.iter_max,vari_num);
            result_Obj=zeros(self.iter_max,1);
            if con_num,result_Con=zeros(self.iter_max,con_num);
            else,result_Con=[];end
            if coneq_num,result_Coneq=zeros(self.iter_max,coneq_num);
            else,result_Coneq=[];end
            if vio_num,result_Vio=zeros(self.iter_max,vio_num);
            else,result_Vio=[];end
            x_best=[];obj_best=[];vio_best=[];

            self.Srgt_obj=repmat({model_option},obj_num,1);
            self.Srgt_con=repmat({model_option},con_num,1);
            self.Srgt_coneq=repmat({model_option},coneq_num,1);

            iter=iter+1;
            while ~done
                % construct kriging model
                [self.obj_fcn_srgt,self.con_fcn_srgt,...
                    self.Srgt_obj,self.Srgt_con,self.Srgt_coneq]=self.getSrgtFcnKRG...
                    (X,Obj,Con,Coneq,...
                    self.Srgt_obj,self.Srgt_con,self.Srgt_coneq);
                best_idx=self.datalib.Best_idx(end);
                x_min=X(best_idx,:);
                obj_min=Obj(best_idx,:);
                if ~isempty(Vio),vio_min=Vio(best_idx,:);
                else,vio_min=[];end

                % add infill point
                [x_infill,infill_fcn]=self.searchInfill(vari_num,low_bou,up_bou,x_min,obj_min,vio_min);
                [self.datalib,x_infill,obj_infill]=self.sample(self.datalib,objcon_fcn,x_infill);
                [X,Obj,Con,Coneq,Vio]=self.datalibLoad(self.datalib,low_bou,up_bou);

                best_idx=self.datalib.Best_idx(end);
                result_NFE(iter,:)=self.NFE;
                x_best=X(best_idx,:);
                result_X(iter,:)=x_best;
                obj_best=Obj(best_idx,:);
                result_Obj(iter,:)=obj_best;
                if con_num,result_Con(iter,:)=Con(best_idx,:);end
                if coneq_num,result_Coneq(iter,:)=Coneq(best_idx,:);end
                if vio_num,vio_best=Vio(best_idx,:);result_Vio(iter,:)=vio_best;end
                iter=iter+1;

                % information
                if self.FLAG_DRAW_FIGURE && vari_num < 3
                    surrogateVisualize(self.Srgt_obj{1},low_bou,up_bou);
                    if vari_num == 1
                        line(x_infill(1),obj_infill,'Marker','o','color','r');
                    else
                        line(x_infill(1),x_infill(2),obj_infill,'Marker','o','color','r');
                    end
                end

                if self.FLAG_INFORMATION
                    fprintf('obj:    %f    vio:    %f    NFE:    %-3d\n',obj_best,vio_best,self.NFE);
                end

                % forced interrupt
                if iter > self.iter_max || self.NFE >= self.NFE_max
                    done=1;
                end

                % convergence judgment
                if self.FLAG_CONV_JUDGE && iter > 2
                    obj_best_old=result_Obj(iter-2,:);
                    if ( abs((obj_best-obj_best_old)/obj_best_old) < self.obj_torl && ...
                            ((~isempty(vio_best) && vio_best == 0) || isempty(vio_best)) )
                        done=1;
                    end
                end
            end

            % cut result
            result_NFE(iter:end,:)=[];
            result_X(iter:end,:)=[];
            result_Obj(iter:end,:)=[];
            if con_num,result_Con(iter:end,:)=[];end
            if coneq_num,result_Coneq(iter:end,:)=[];end
            if vio_num,result_Vio(iter:end,:)=[];end
        end

        function [datalib,X,Obj,Con,Coneq,Vio,repeat_idx]=sample...
                (self,datalib,objcon_fcn,X_add,cost,fidelity)
            % find best result to record
            %
            add_torl=self.option_optim.add_torl;
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
                if any(dist < add_torl)
                    overlap_idx=find(dist < add_torl,1);
                    repeat_idx(x_idx)=overlap_idx;
                    datalib_idx(x_idx)=overlap_idx;
                else
                    datalib=self.datalibAdd(datalib,objcon_fcn,x_add);

                    % recode add information
                    self.NFE=self.NFE+cost;
                    self.Add_idx=[self.Add_idx;fidelity,size(datalib.X,1)];
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
        function [x_infill,obj_fcn]=searchInfill(self,vari_num,low_bou,up_bou,x_init,obj_min,vio_min)
            % MSP, EI, PI, MSE, LCB infill criteria
            %

            % basical infill function
            switch self.option_optim.criteria
                case 'MSP'
                    infill_fcn=@(x) self.obj_fcn_srgt(x);
                case 'EI'
                    w_l=0.5;
                    infill_fcn=@(x) -EIFcn(x,self.obj_fcn_srgt,obj_min,w_l);
                case 'PI'
                    infill_fcn=@(x) -PIFcn(x,self.obj_fcn_srgt,obj_min);
                case 'MSE'
                    infill_fcn=@(x) -MSEFcn(x,self.obj_fcn_srgt);
                case 'LCB'
                    infill_fcn=@(x) LCBFcn(x,self.obj_fcn_srgt);
            end

            % expert knowledge
            expert_fcn=self.option_optim.expert_fcn;
            if ~isempty(expert_fcn)
                infill_fcn=@(x) infill_fcn(x)*expert_fcn(x);
            end

            obj_fcn=infill_fcn;
            con_fcn=[];

            % constraint process
            if self.FLAG_CON
                switch self.option_optim.constraint
                    case 'auto'
                        if (vio_min > 0)
                            obj_fcn=@(x) -POFFcn(x,self.con_fcn_srgt);
                        else
                            POF_fcn=@(x) POFFcn(x,self.con_fcn_srgt);

                            obj_fcn=@(x) infill_fcn(x)*POF_fcn(x);
                        end
                    case 'POF'
                        POF_fcn=@(x) POFFcn(x,self.con_fcn_srgt);

                        obj_fcn=@(x) infill_fcn(x)*POF_fcn(x);
                    case 'direct'
                        obj_fcn=infill_fcn;
                        con_fcn=self.con_fcn_srgt;
                end
            end

            problem.solver='fmincon';
            problem.objective=obj_fcn;
            problem.x0=x_init;
            problem.lb=low_bou;
            problem.ub=up_bou;
            problem.nonlcon=con_fcn;
            problem.options=self.option_optim.fmincon_option;
            ms=MultiStart('Display','off');
            rs=CustomStartPointSet([x_init;lhsdesign(19,vari_num).*(up_bou-low_bou)+low_bou]);
            [x_infill,~,exit_flag]=run(ms,problem,rs);

            if (exit_flag == -2 && strcmp(self.option_optim.constraint,'auto')) ...
                    || norm(x_infill-x_init) < self.option_optim.add_torl
                problem.objective=infill_fcn;
                problem.nonlcon=self.con_fcn_srgt;
                rs=CustomStartPointSet([x_init;lhsdesign(19,vari_num).*(up_bou-low_bou)+low_bou]);
                [x_infill,~,~]=run(ms,problem,rs);
            end

            function EI=EIFcn(x,obj_fcn_srgt,obj_min_surr,w_l)
                w_g=1-w_l;
                [obj_pred,obj_var]=obj_fcn_srgt(x);
                obj_nomlz=(obj_min_surr-obj_pred)/sqrt(obj_var);
                EI_l=(obj_min_surr-obj_pred)*normcdf(obj_nomlz);
                EI_g=sqrt(obj_var)*normpdf(obj_nomlz);
                EI=w_l*EI_l+w_g*EI_g;
                EI(obj_var < eps)=0;
            end

            function PI=PIFcn(x,obj_fcn_srgt,obj_min_surr)
                [obj_pred,obj_var]=obj_fcn_srgt(x);
                obj_nomlz=(obj_min_surr-obj_pred)/sqrt(obj_var);
                PI=normcdf(obj_nomlz);
                PI(obj_var < eps)=0;
            end

            function MSE=MSEFcn(x,obj_fcn_srgt)
                [~,obj_var]=obj_fcn_srgt(x);
                MSE=obj_var;
            end

            function LCB=LCBFcn(x,obj_fcn_srgt)
                [obj_pred,obj_var]=obj_fcn_srgt(x);
                LCB=obj_pred-sqrt(obj_var);
            end

            function POF=POFFcn(x,con_fcn_srgt)
                % calculate feasible probability
                %
                POF=1;
                [con_pred,~,con_var,~]=con_fcn_srgt(x);
                con_var(con_var < eps)=eps;
                con_nomlz=con_pred./sqrt(con_var);
                if ~isempty(con_pred)
                    if all(con_pred > 0)
                        POF=0;
                        return;
                    end
                    con_feas_prob=normcdf(-con_nomlz);
                    POF=POF*prod(con_feas_prob);
                end
            end
        end
    end

    % auxiliary function
    methods(Static)
        function [X_surr,Obj_surr,Con_surr,Coneq_surr,Vio_surr,...
                Obj_max,Con_max,Coneq_max,Vio_max]=preSurrData...
                (X,Obj,Con,Coneq,Vio,nomlz_fval)
            % normalize data by max value in data
            %
            X_surr=X;
            Obj_max=max(abs(Obj),[],1);
            Obj_surr=Obj/Obj_max*nomlz_fval;
            if ~isempty(Con)
                Con_max=max(abs(Con),[],1);
                Con_surr=Con./Con_max*nomlz_fval;
            else
                Con_max=[];
                Con_surr=[];
            end
            if ~isempty(Coneq)
                Coneq_max=max(abs(Coneq),[],1);
                Coneq_surr=Coneq./Coneq_max*nomlz_fval;
            else
                Coneq_max=[];
                Coneq_surr=[];
            end
            if ~isempty(Vio)
                Vio_max=max(abs(Vio),[],1);
                Vio_surr=Vio./Vio_max*nomlz_fval;
            else
                Vio_max=[];
                Vio_surr=[];
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
                Srgt_obj{obj_idx}=srgtsfKRG(x_list,obj_list(:,obj_idx),Srgt_obj{obj_idx});
            end

            % generate con surrogate
            if ~isempty(con_list)
                if isempty(Srgt_con),Srgt_con=cell(size(con_list,2),1);end
                for con_idx=1:size(con_list,2)
                    Srgt_con{con_idx}=srgtsfKRG(x_list,con_list(:,con_idx),Srgt_con{con_idx});
                end
            else
                Srgt_con=[];
            end

            % generate coneq surrogate
            if ~isempty(coneq_list)
                if isempty(Srgt_coneq),Srgt_coneq=cell(size(coneq_list,2),1);end
                for coneq_idx=1:size(coneq_list,2)
                    Srgt_coneq{coneq_idx}=srgtsfKRG(x_list,coneq_list(:,coneq_idx),Srgt_coneq{coneq_idx});
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

        function [datalib,x,obj,con,coneq,vio]=datalibAdd(datalib,objcon_fcn,x)
            % add new x into data lib
            %
            [obj,con,coneq]=objcon_fcn(x);vio=[]; % eval value

            % calculate vio
            if ~isempty(con),vio=[vio,max(max(con-datalib.con_torl,0),[],2)];end
            if ~isempty(coneq),vio=[vio,max(max(abs(coneq)-datalib.con_torl,0),[],2)];end
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

%% surrogate

function model_KRG=srgtsfKRG(X,Y,model_option)
% generate Kriging surrogate model
% input data will be normalize by average and standard deviation of data
%
% input:
% X (matrix): x_num x vari_num matrix
% Y (matrix): x_num x 1 matrix
% model_option (struct): optional input
%
% model_option include:
% optimize_hyp, simplify_hyp, optimize_option, hyp, reg_fcn
%
% output:
% model_KRG (struct): a Kriging model
%
% abbreviation:
% num: number, pred: predict, vari: variable, hyp: hyper parameter
% NLL: negative log likelihood
%
% Copyright 2023.2 Adel
%
if nargin < 3,model_option=struct();end

% Kriging option
if ~isfield(model_option,'optimize_hyp'), model_option.('optimize_hyp')=true;end
if ~isfield(model_option,'simplify_hyp'), model_option.('simplify_hyp')=true;end
if model_option.('simplify_hyp'),FLAG_GRAD=false;else,FLAG_GRAD=true;end
if ~isfield(model_option,'optimize_option')
    model_option.('optimize_option')=optimoptions...
        ('fminunc','Display','none',...
        'OptimalityTolerance',1e-6,...
        'FiniteDifferenceStepSize',1e-5,...,
        'MaxIterations',20,'SpecifyObjectiveGradient',FLAG_GRAD);
end

if ~isfield(model_option,'hyp'), model_option.('hyp')=[];end
if ~isfield(model_option,'reg_fcn'), model_option.('reg_fcn')=[];end
if ~isfield(model_option,'cov_fcn'), model_option.('cov_fcn')=[];end

% normalize data
[x_num,vari_num]=size(X);
aver_Y=mean(Y);
stdD_Y=std(Y);stdD_Y(stdD_Y == 0)=1;
Y_nomlz=(Y-aver_Y)./stdD_Y;
aver_X=mean(X);
stdD_X=std(X);stdD_X(stdD_X == 0)=1;
X_nomlz=(X-aver_X)./stdD_X;

% covarianve function define
cov_fcn=model_option.('cov_fcn');
if isempty(cov_fcn)
    % initial X_dis_sq
    X_dis_sq=zeros(x_num,x_num,vari_num);
    for vari_idx=1:vari_num
        X_dis_sq(:,:,vari_idx)=(X_nomlz(:,vari_idx)-X_nomlz(:,vari_idx)').^2;
    end

    cov_fcn=@(X,X_pred,hyp) covExp(X,X_pred,hyp,aver_X,stdD_X,X_nomlz,X_dis_sq);
end

% regression function define
reg_fcn=model_option.('reg_fcn');
if isempty(reg_fcn)
        reg_fcn=@(X) ones(size(X,1),1);
%     if x_num < vari_num,reg_fcn=@(X) ones(size(X,1),1).*stdD_Y+aver_Y; % constant
%     else,reg_fcn=@(X) [ones(size(X,1),1),X-stdD_X].*stdD_Y+aver_Y;end % linear
end

% calculate reg
fval_reg=reg_fcn(X);
fval_reg_nomlz=(fval_reg-aver_Y)./stdD_Y;

hyp=model_option.('hyp');
% kernal function is exp(-X_sq/vari_num^2*exp(hyp))
if isempty(hyp), hyp=ones(1,vari_num);end
% if isempty(hyp), hyp=log(x_num^(1/vari_num)*vari_num)*ones(1,vari_num);end

% if optimize hyperparameter
if model_option.optimize_hyp
    simplify_hyp=model_option.('simplify_hyp');
    obj_fcn_hyp=@(hyp) probNLLKRG(X_nomlz,Y_nomlz,x_num,vari_num,cov_fcn,hyp,fval_reg_nomlz);

    if simplify_hyp
        hyp=mean(hyp);
        low_bou_hyp=-4;
        up_bou_hyp=4;
    else
        low_bou_hyp=-4*ones(1,vari_num);
        up_bou_hyp=4*ones(1,vari_num);
    end

    % [fval,gradient]=obj_fcn_hyp(hyp)
    % [~,gradient_differ]=differ(obj_fcn_hyp,hyp)
    % drawFcn(obj_fcn_hyp,low_bou_hyp,up_bou_hyp);

    [hyp,~,~,~]=fminunc(obj_fcn_hyp,hyp,model_option.('optimize_option'));
    hyp=min(hyp,up_bou_hyp);hyp=max(hyp,low_bou_hyp);

    if simplify_hyp, hyp=hyp*ones(1,vari_num);end
end

% get parameter
cov=cov_fcn(X_nomlz,[],hyp);
[L_cov,beta,sigma_sq,inv_L_F_reg,~,inv_L_U]=calKRG(cov,Y_nomlz,x_num,fval_reg_nomlz);
sigma_sq=sigma_sq*stdD_Y^2; % renormalize data
gamma=L_cov'\inv_L_U;
inv_FTcovF=(inv_L_F_reg'*inv_L_F_reg)\eye(size(fval_reg_nomlz,2));

% initialization predict function
pred_fcn=@(X_predict) predictKRG...
    (X_predict,X,aver_Y,stdD_Y,...
    reg_fcn,cov_fcn,hyp,...
    L_cov,beta,sigma_sq,gamma,inv_L_F_reg,inv_FTcovF);

model_KRG=model_option;

model_KRG.X=X;
model_KRG.Y=Y;

model_KRG.hyp=hyp;
model_KRG.beta=beta;
model_KRG.gamma=gamma;

model_KRG.predict=pred_fcn;

    function [fval,grad]=probNLLKRG(X,Y,x_num,vari_num,cov_fcn,hyp,F_reg)
        % function to minimize sigma_sq
        %
        hyp=min(hyp,up_bou_hyp);hyp=max(hyp,low_bou_hyp); % Prevent excessive hyp
        if simplify_hyp, hyp=hyp*ones(1,vari_num);end % extend hyp

        if nargout > 1
            % require gradient
            [R,dR_dhyp]=cov_fcn(X,[],hyp);
        else
            [R]=cov_fcn(X,[],hyp);
        end
        [L,Beta,sigma2,inv_L_F]=calKRG(R,Y,x_num,F_reg);

        % calculation negative log likelihood
        if sigma2 == 0
            fval=0;grad=zeros(vari_num,1);
            if simplify_hyp, grad=0;end
            return;
        end
        fval=x_num/2*log(sigma2)+sum(log(diag(L)));

        if nargout > 1
            % calculate gradient
            grad=zeros(1,vari_num);
            inv_R=L'\(L\eye(x_num));
            inv_FTRF=(inv_L_F'*inv_L_F)\eye(size(F_reg,2));
            Y_Fmiu=Y-F_reg*Beta;

            for vari_i=1:vari_num
                dinv_R_dtheta=...
                    -inv_R*dR_dhyp(:,:,vari_i)*inv_R;
                dinv_FTRF_dtheta=-inv_FTRF*...
                    (F_reg'*dinv_R_dtheta*F_reg)*...
                    inv_FTRF;
                dmiu_dtheta=dinv_FTRF_dtheta*(F_reg'*inv_R*Y)+...
                    inv_FTRF*(F_reg'*dinv_R_dtheta*Y);
                dU_dtheta=-F_reg*dmiu_dtheta;
                dsigma2_dtheta=(dU_dtheta'*inv_R*Y_Fmiu+...
                    Y_Fmiu'*dinv_R_dtheta*Y_Fmiu+...
                    Y_Fmiu'*inv_R*dU_dtheta)/x_num;
                dlnsigma2_dtheta=1/sigma2*dsigma2_dtheta;
                dlndetR=trace(inv_R*dR_dhyp(:,:,vari_i));

                grad(vari_i)=x_num/2*dlnsigma2_dtheta+0.5*dlndetR;
            end

            if simplify_hyp, grad=sum(grad);end
        end
    end

    function [L_cov,beta,sigma_sq,inv_L_F_reg,inv_L_Y,inv_L_U]=calKRG(cov,Y,x_num,F_reg)
        % kriging interpolation kernel function
        % y(x)=f(x)+z(x)
        %
        L_cov=chol(cov)'; % inv_cov=cov\eye(x_num);
        inv_L_F_reg=L_cov\F_reg;
        inv_L_Y=L_cov\Y; % inv_FTRF=(F_reg'*inv_cov*F_reg)\eye(size(F_reg,2));

        % basical bias
        beta=inv_L_F_reg\inv_L_Y; % beta=inv_FTRF*(F_reg'*inv_cov*Y);
        inv_L_U=inv_L_Y-inv_L_F_reg*beta; % U=Y-F_reg*beta;
        sigma_sq=(inv_L_U'*inv_L_U)/x_num; % sigma_sq=(U'*inv_cov*U)/x_num;
    end

    function [Y_pred,Var_pred]=predictKRG...
            (X_pred,X,aver_Y,stdD_Y,...
            reg_fcn,cov_fcn,hyp,...
            L_cov,beta,sigma_sq,gamma,inv_L_F_reg,inv_FTcovF)
        % Kriging surrogate predict function
        %
        % input:
        % X_pred (matrix): x_pred_num x vari_num matrix, predict X
        %
        % output:
        % Y_pred (matrix): x_pred_num x 1 matrix, value
        % Var_pred (matrix): x_pred_num x 1 matrix, variance
        %

        fval_reg_pred_nomlz=(reg_fcn(X_pred)-aver_Y)./stdD_Y; % regression value
        cov_pred=cov_fcn(X,X_pred,hyp); % predict covariance

        % predict base fval
        Y_pred=fval_reg_pred_nomlz*beta+cov_pred'*gamma;

        % predict variance
        inv_L_r=L_cov\cov_pred;
        u=(inv_L_F_reg)'*inv_L_r-fval_reg_pred_nomlz';
        Var_pred=sigma_sq*(1+u'*inv_FTcovF*u-inv_L_r'*inv_L_r);
        Var_pred=diag(Var_pred);

        % renormalize data
        Y_pred=Y_pred*stdD_Y+aver_Y;
    end

    function [cov,dcov_dhyp]=covExp(X,X_pred,hyp,aver_X,stdD_X,X_nomlz,X_dis_sq)
        % gaussian covariance
        %
        [x_n,vari_n]=size(X);
        theta=exp(hyp);
        if isempty(X_pred)
            % self covariance
            cov=zeros(x_n,x_n);
            for vari_i=1:vari_n
                cov=cov+X_dis_sq(:,:,vari_i)*theta(vari_i);
            end
            cov=exp(-cov/vari_n^2)+eye(x_n)*((1000+x_n)*eps);
        else
            x_pred_num=size(X_pred,1);
            X_pred_nomlz=(X_pred-aver_X)./stdD_X; % normalize data

            % predict covariance
            cov=zeros(x_n,x_pred_num);
            for vari_i=1:vari_n
                cov=cov+(X_nomlz(:,vari_i)-X_pred_nomlz(:,vari_i)').^2*theta(vari_i);
            end
            cov=exp(-cov/vari_n^2);
        end

        if nargout > 1
            dcov_dhyp=zeros(x_n,x_n,vari_n);
            for vari_i=1:vari_n
                dcov_dhyp(:,:,vari_i)=-(X_dis_sq(:,:,vari_i).*cov)*theta(vari_i)/vari_n^2;
            end
        end
    end
end
