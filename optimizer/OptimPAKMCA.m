classdef OptimPAKMCA < handle
    % PAKM-CA optimization algorithm
    % Parallel Adaptive Kriging Method with Constraint Aggregation
    % only support constraints problem
    %
    % referance: [1] LONG T,WEI Z,SHI R,et al. Parallel Adaptive Kriging Method
    % with Constraint Aggregation for Expensive Black-Box Optimization Problems
    % [J]. AIAA Journal,2021,59(9): 3465-79.
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

        obj_fcn_srgt=[];
        con_fcn_srgt=[];
        ks_fcn_srgt=[];

        Srgt_obj=[];
        Srgt_con=[];
        Srgt_coneq=[];
        Srgt_ks=[];
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
        dataoptim_filestr=''; % optimize save mat name

        add_torl=1000*eps; % surrogate add point protect range
        X_init=[];

        % hyper parameter
        sample_num_init=[];
        sample_num_add=[];

        % hyper parameter
        rou_min=1;
        rou_max=64;
        rou_decrease=0.5;
        rou_increase=2;

        KRG_option=struct('simplify_flag',true);
    end

    % main function
    methods
        function self=OptimPAKMCA(NFE_max,iter_max,obj_torl,con_torl)
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
                    if ~contains(prob_field,'objcon_fcn'), error('OptimRBFCDE.optimize: input problem lack objcon_fcn'); end
                    objcon_fcn=problem.objcon_fcn;
                    if ~contains(prob_field,'vari_num'), error('OptimRBFCDE.optimize: input problem lack vari_num'); end
                    if ~contains(prob_field,'low_bou'), error('OptimRBFCDE.optimize: input problem lack low_bou'); end
                    if ~contains(prob_field,'up_bou'), error('OptimRBFCDE.optimize: input problem lack up_bou'); end
                else
                    prob_method=methods(problem);
                    if ~contains(prob_method,'objcon_fcn'), error('OptimRBFCDE.optimize: input problem lack objcon_fcn'); end
                    objcon_fcn=@(x) problem.objcon_fcn(x);
                    prob_pro=properties(problem);
                    if ~contains(prob_pro,'vari_num'), error('OptimRBFCDE.optimize: input problem lack vari_num'); end
                    if ~contains(prob_pro,'low_bou'), error('OptimRBFCDE.optimize: input problem lack low_bou'); end
                    if ~contains(prob_pro,'up_bou'), error('OptimRBFCDE.optimize: input problem lack up_bou'); end
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
            if isempty(self.sample_num_init)
                if vari_num < 10,self.sample_num_init=min((vari_num+1)*(vari_num+2)/2,5*vari_num);
                else,self.sample_num_init=vari_num+1;end
            end

            if isempty(self.sample_num_add)
                if vari_num <= 5,self.sample_num_add=2;
                else,self.sample_num_add=3;end
            end

            % NFE and iteration setting
            if isempty(self.NFE_max),self.NFE_max=10+10*vari_num;end
            if isempty(self.iter_max),self.iter_max=20+20*vari_num;end

            if isempty(self.dataoptim)
                self.dataoptim.NFE=0;
                self.dataoptim.Add_idx=[];
                self.dataoptim.iter=0;
                self.dataoptim.done=false;
                self.dataoptim.rou=4;
            end

            % step 2-3, generate initial data library
            self.sampleInit(objcon_fcn,vari_num,low_bou,up_bou);

            % step 3-6, adaptive samlping base on optimize strategy
            
            % initialize all data to begin optimize
            [X,Obj,Con,Coneq,Vio]=self.datalibLoad(self.datalib,low_bou,up_bou);
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

            self.Srgt_obj=repmat({self.KRG_option},1,obj_num);
            self.Srgt_con=repmat({self.KRG_option},1,con_num);
            self.Srgt_coneq=repmat({self.KRG_option},1,coneq_num);
            self.Srgt_ks={self.KRG_option};

            self.dataoptim.iter=self.dataoptim.iter+1;
            while ~self.dataoptim.done
                % step 4
                % nomalization all data by max obj and to create surrogate model
                [self.obj_fcn_srgt,self.con_fcn_srgt,...
                    self.Srgt_obj,self.Srgt_con,self.Srgt_coneq]=self.getSrgtFcnKRG...
                    (X,Obj,Con,Coneq,self.Srgt_obj,self.Srgt_con,self.Srgt_coneq);

                % step 5
                % use ga to obtain pseudo-optimum
                ga_option=optimoptions('ga','Display','none','ConstraintTolerance',self.con_torl,'MaxGenerations',10,'HybridFcn','fmincon');
                [x_infill,~,exit_flag,output_ga]=ga...
                    (self.obj_fcn_srgt,vari_num,[],[],[],[],low_bou,up_bou,self.con_fcn_srgt,ga_option);

                % updata infill point
                [self.datalib,x_infill,obj_infill,con_infill,coneq_infill,vio_infill]=self.sample(self.datalib,objcon_fcn,x_infill);
                [X,Obj,Con,Coneq,Vio]=self.datalibLoad(self.datalib,low_bou,up_bou);

                % step 6
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

                % PCFEI Function-Based Infill Sampling Mechanism
                if ~self.dataoptim.done
                    % step 1
                    % construct kriging model of KS function
                    % updata rou
                    if vio_infill > 0
                        self.dataoptim.rou=self.dataoptim.rou*self.rou_increase;
                    else
                        self.dataoptim.rou=self.dataoptim.rou*self.rou_decrease;
                    end
                    self.dataoptim.rou=max(self.dataoptim.rou,self.rou_min);
                    self.dataoptim.rou=min(self.dataoptim.rou,self.rou_max);

                    % Ks_surr=log(sum(exp(Con*self.dataoptim.rou),2))/self.dataoptim.rou; % modify

                    self.Srgt_obj={srgtKRG(X,Obj,self.Srgt_obj{1})};
                    self.obj_fcn_srgt=@(x) self.Srgt_obj{1}.predict(x);
                    self.Srgt_ks={srgtKRG(X,Vio,self.Srgt_ks{1})};
                    self.ks_fcn_srgt=@(x) self.Srgt_ks{1}.predict(x);

                    % step 2
                    % contruct EI,PF function
                    EI_fcn=@(X) EIFcn(self.obj_fcn_srgt,X,min(Obj));
                    PF_fcn=@(X) PFFcn(self.ks_fcn_srgt,X);
                    IF_fcn=@(X) IFFcn(x_infill,X,exp(self.Srgt_obj{1}.hyp),vari_num);

                    % step 3
                    % multi objective optimization to get pareto front
                    PCFEI_fcn=@(x) [-EI_fcn(x),-PF_fcn(x)];

                    gamultiobj_option=optimoptions('gamultiobj','Display','none','MaxGenerations',20);
                    [X_pareto,Obj_pareto,exitflag,output_gamultiobj]=gamultiobj...'
                        (PCFEI_fcn,vari_num,[],[],[],[],low_bou,up_bou,[],gamultiobj_option);

                    % step 4
                    % base on PCFEI value to get first sample_number_iteration point
                    num_add=min(self.sample_num_add,self.NFE_max-self.dataoptim.NFE-1);
                    if size(X_pareto,1) < num_add
                        X_add = X_pareto;
                    else
                        EI_list=-Obj_pareto(:,1);
                        EI_list=EI_list/max(EI_list);
                        PF_list=-Obj_pareto(:,2);
                        PF_list=PF_list/max(PF_list);
                        IF_list=IF_fcn(X_pareto);
                        IF_list=IF_list/max(IF_list);
                        PCFEI_list=EI_list.*PF_list.*IF_list;
                        [~,Idx_sort]=sort(PCFEI_list);

                        X_add=X_pareto(Idx_sort((end+1-num_add):end),:);
                    end

                    % step 5
                    % updata data lib
                    self.datalib=self.sample(self.datalib,objcon_fcn,X_add);
                    [X,Obj,Con,Coneq,Vio]=self.datalibLoad(self.datalib,low_bou,up_bou);
                end

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

            function EI=EIFcn(obj_fcn_srgt,X,obj_min_surr)
                % EI function
                [obj_pred,obj_var]=obj_fcn_srgt(X);
                normal_obj=(obj_min_surr-obj_pred)./sqrt(obj_var);
                EI_l=(obj_min_surr-obj_pred).*normcdf(normal_obj);
                EI_g=sqrt(obj_var).*normpdf(normal_obj);
                EI=EI_l+EI_g;
            end

            function PF=PFFcn(ks_fcn_srgt,X)
                % PF function
                [Ks_pred,Ks_var]=ks_fcn_srgt(X);
                PF=normcdf(-Ks_pred./sqrt(Ks_var));
            end

            function IF=IFFcn(x_best,X,theta,vari_num)
                IF=zeros(size(X,1),1);
                if length(theta) ~= vari_num,theta=[theta,mean(theta)*ones(1,vari_num-length(theta))];end
                for vari_idx=1:vari_num
                    IF=IF+(X(:,vari_idx)-x_best(:,vari_idx)').^2*theta(vari_idx);
                end
                IF=1-exp(-IF);
            end
        end

        function sampleInit(self,objcon_fcn,vari_num,low_bou,up_bou)
            % initial latin hypercube sample
            %

            % obtain datalib
            if isempty(self.datalib)
                self.datalib=self.datalibGet(vari_num,low_bou,up_bou,self.con_torl,self.datalib_filestr);
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

    % common function
    methods(Static)
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
                    Srgt_con{con_idx}=srgtKRG(x_list,con_list(:,con_idx));
                end
            else
                Srgt_con=[];
            end

            % generate coneq surrogate
            if ~isempty(coneq_list)
                if isempty(Srgt_coneq),Srgt_coneq=cell(size(coneq_list,2),1);end
                for coneq_idx=1:size(coneq_list,2)
                    Srgt_coneq{coneq_idx}=srgtKRG(x_list,coneq_list(:,coneq_idx));
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
                if isempty(vio) || vio == 0
                    if obj <= datalib.Obj(Best_idx(end))
                        Best_idx=[Best_idx;size(datalib.X,1)];
                    else
                        Best_idx=[Best_idx;Best_idx(end)];
                    end
                else
                    if vio <= datalib.Vio(Best_idx(end))
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
