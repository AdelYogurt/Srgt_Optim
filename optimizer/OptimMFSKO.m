classdef OptimMFSKO < handle
    % Multiple-Fidelity Sequential Kriging Optimization
    % include MSP, EI, PI, MSE, LCB infill criteria
    %
    % referance:
    % [1] Ruan X, Jiang P, Zhou Q, et al. Variable-fidelity
    % probability of improvement method for efficient global optimization
    % of expensive black-box problems [J]. Structural and multidisciplinary
    % optimization, 2020, 62(6): 3021-52.
    % [2] Huang D, Allen T T, Notz W I,
    % et al. Sequential kriging optimization using multiple-fidelity
    % evaluations [J]. Structural and multidisciplinary optimization, 2006,
    % 32(5): 369-82.
    %
    % Copyright 2023.2 Adel
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

        datalib;
        dataoptim;

        NFE_max;
        iter_max;
        obj_torl;
        con_torl;
    end

    properties
        % surrogate information
        CoKRG_option=struct();
        adapt_hyp=false;

        % strategy parameter option
        sample_num_init=[];

        % optimize process option
        datalib_filestr=''; % datalib save mat name
        dataoptim_filestr=''; % optimize save mat name
        nomlz_value=100; % max obj when normalize obj,con,coneq
        add_torl=1000*eps; % surrogate add point protect range
        X_init=[]; % initial sample point
        criteria='AEI'; % infill criteria
        constraint='auto'; % constraint process method
        fmincon_option=optimoptions('fmincon','Display','none','Algorithm','sqp','ConstraintTolerance',0,'FiniteDifferenceStepSize',1e-5);

        FLAG_CON;
        FLAG_MULTI_OBJ;
        FLAG_MULTI_FIDELITY;
        FLAG_DiISCRETE_VARI;

        FLAG_DRAW_FIGURE=0; % whether draw data
        FLAG_INFORMATION=1; % whether print data
        FLAG_CONV_JUDGE=0; % whether judgment convergence
    end

    % main function
    methods
        function self=OptimMFSKO(NFE_max,iter_max,obj_torl,con_torl)
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


        end

        function [x_best,obj_best,NFE,output,con_best,coneq_best,vio_best]=optimize(self,objcon_fcn_list,vari_num,low_bou,up_bou,cost_list)
            % optimize driver
            %
            fid_num=length(objcon_fcn_list);
            cost_list=cost_list(:);

            % hyper parameter
            if isempty(self.sample_num_init)
                self.sample_num_init=ceil((6*vari_num+3)*(1./cost_list/sum(1./cost_list)));
            end

            % NFE and iteration setting
            if isempty(self.NFE_max),self.NFE_max=10+10*vari_num;end
            if isempty(self.iter_max),self.iter_max=20+20*vari_num;end

            if isempty(self.dataoptim)
                self.dataoptim.NFE=0;
                self.dataoptim.Add_idx=[];
                self.dataoptim.iter=0;
                self.dataoptim.done=false;
            end

            % step 2, latin hypercube sample
            self.sampleInit(objcon_fcn_list,vari_num,low_bou,up_bou,cost_list);

            % step 3, adaptive sample

            % initialize all data to begin optimize
            X=cell(fid_num,1);Obj=cell(fid_num,1);Con=cell(fid_num,1);Coneq=cell(fid_num,1);Vio=cell(fid_num,1);
            for fid_idx=1:fid_num
                [X{fid_idx},Obj{fid_idx},Con{fid_idx},Coneq{fid_idx},Vio{fid_idx}]=self.datalibLoad(self.datalib{fid_idx},low_bou,up_bou);
            end
            obj_num=size(Obj{1},2);con_num=size(Con{1},2);coneq_num=size(Coneq{1},2);vio_num=size(Vio{1},2);
            result_X=zeros(self.iter_max,vari_num);
            result_Obj=zeros(self.iter_max,1);
            if con_num,result_Con=zeros(self.iter_max,con_num);
            else,result_Con=[];end
            if coneq_num,result_Coneq=zeros(self.iter_max,coneq_num);
            else,result_Coneq=[];end
            if vio_num,result_Vio=zeros(self.iter_max,vio_num);
            else,result_Vio=[];end
            x_best=[];obj_best=[];vio_best=[];

            self.Srgt_obj=repmat({self.CoKRG_option},obj_num,1);
            self.Srgt_con=repmat({self.CoKRG_option},con_num,1);
            self.Srgt_coneq=repmat({self.CoKRG_option},coneq_num,1);

            self.dataoptim.iter=self.dataoptim.iter+1;
            while ~self.dataoptim.done
                % construct kriging model
                [self.obj_fcn_srgt,self.con_fcn_srgt,...
                    self.Srgt_obj,self.Srgt_con,self.Srgt_coneq]=self.getSrgtFcnCoKRG...
                    (X,Obj,Con,Coneq,...
                    self.Srgt_obj,self.Srgt_con,self.Srgt_coneq);
                best_idx=self.datalib{end}.Best_idx(end);
                x_min=X{end}(best_idx,:);
                obj_min=Obj{end}(best_idx,:);
                if ~isempty(Vio{end}),vio_min=Vio{end}(best_idx,:);
                else,vio_min=[];end

                % search each fid infill point
                [x_infill,fid_infill,infill_fid_fcn]=self.searchInfill(fid_num,vari_num,low_bou,up_bou,cost_list,x_min,obj_min,vio_min);
                [self.datalib{fid_infill},x_infill,obj_infill]=self.sample...
                    (self.datalib{fid_infill},objcon_fcn_list{fid_infill},x_infill,cost_list(fid_infill),fid_infill);
                [X{fid_infill},Obj{fid_infill},Con{fid_infill},Coneq{fid_infill},Vio{fid_infill}]=self.datalibLoad(self.datalib{fid_infill},low_bou,up_bou);

                best_idx=self.datalib{end}.Best_idx(end);
                x_best=X{end}(best_idx,:);
                result_X(self.dataoptim.iter,:)=x_best;
                obj_best=Obj{end}(best_idx,:);
                result_Obj(self.dataoptim.iter,:)=obj_best;
                if con_num,result_Con(self.dataoptim.iter,:)=Con{end}(best_idx,:);end
                if coneq_num,result_Coneq(self.dataoptim.iter,:)=Coneq{end}(best_idx,:);end
                if vio_num,vio_best=Vio{end}(best_idx,:);result_Vio(self.dataoptim.iter,:)=vio_best;end
                self.dataoptim.iter=self.dataoptim.iter+1;

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
                    fprintf('obj:    %f    vio:    %f    NFE:    %-.4f\n',obj_best,vio_best,self.dataoptim.NFE);
                end

                % forced interrupt
                if self.dataoptim.iter > self.iter_max || self.dataoptim.NFE >= self.NFE_max
                    self.dataoptim.done=1;
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

        function sampleInit(self,objcon_fcn_list,vari_num,low_bou,up_bou,cost_list)
            % initial latin hypercube sample
            %

            fid_num=length(objcon_fcn_list);
            % obtain datalib
            if isempty(self.datalib),self.datalib=cell(fid_num,1);end
            if isempty(self.X_init),self.X_init=cell(fid_num,1);end

            for fid_idx=1:fid_num
                data_lib=self.datalib{fid_idx};
                X_init_fid=self.X_init{fid_idx};
                sample_num_init_fid=self.sample_num_init(fid_idx);
                objcon_fcn=objcon_fcn_list{fid_idx};

                if isempty(data_lib)
                    if isempty(self.datalib_filestr),filestr=[];
                    else,filestr=[self.datalib_filestr,num2str(fid_idx,'_%d')];end
                    data_lib=self.datalibGet(vari_num,low_bou,up_bou,self.con_torl,filestr);
                end

                if size(data_lib.X,1) < sample_num_init_fid
                    if isempty(X_init_fid)
                        % use latin hypercube method to get enough initial sample x_list
                        sample_num=min(sample_num_init_fid-size(data_lib.X,1),self.NFE_max-self.dataoptim.NFE*cost_list(fid_idx));
                        X_init_fid=lhsdesign(sample_num,vari_num,'iterations',50,'criterion','maximin').*(up_bou-low_bou)+low_bou;
                    end

                    % updata data lib by x_list
                    data_lib=self.sample(data_lib,objcon_fcn,X_init_fid,cost_list(fid_idx),fid_idx);
                end

                self.datalib{fid_idx}=data_lib;
                self.X_init{fid_idx}=X_init_fid;
            end

            % detech expensive constraints
            if ~isempty(self.datalib{end}.Vio)
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
        function [x_infill,fid_infill,obj_fid_fcn]=searchInfill...
                (self,fid_num,vari_num,low_bou,up_bou,cost_list,x_init,obj_min,vio_min)
            % MSP, EI, PI, MSE, LCB infill criteria
            %

            var_fid_fcn=self.Srgt_obj{1}.var_fid;
            predict_list=self.Srgt_obj{1}.predict_list;

            alpha_1_fcn=@(x,fid_idx) corrFcn(x,fid_idx,var_fid_fcn,predict_list);
            alpah_3_fcn=@(x,fid_idx) costFcn(x,fid_idx,cost_list);
            switch self.criteria
                case 'AEI'
                    w_l=0.5;
                    infill_fcn=@(x) -EIFcn(x,self.obj_fcn_srgt,obj_min,w_l);

                    % AEI=EI*alpha_1*alpah_2*alpah_3;
                    infill_fcn=@(x,fid_idx) infill_fcn(x).*alpha_1_fcn(x,fid_idx).*alpah_3_fcn(x,fid_idx);
            end

            obj_fid_fcn=infill_fcn;
            con_fcn=[];
            if self.FLAG_CON
                switch self.constraint
                    case 'auto'
                        if (vio_min > 0)
                            obj_fid_fcn=@(x,fid_idx) -POFFcn(x,self.con_fcn_srgt)...
                                .*alpha_1_fcn(x,fid_idx).*alpah_3_fcn(x,fid_idx);
                        else
                            POF_fcn=@(x) POFFcn(x,self.con_fcn_srgt);

                            obj_fid_fcn=@(x,fid_idx) infill_fcn(x,fid_idx)*POF_fcn(x);
                        end
                    case 'POF'
                        POF_fcn=@(x) POFFcn(x,self.con_fcn_srgt);

                        obj_fid_fcn=@(x,fid_idx) infill_fcn(x,fid_idx)*POF_fcn(x);
                    case 'direct'
                        con_fcn=self.con_fcn_srgt;
                end
            end

            problem.solver='fmincon';
            problem.x0=x_init;
            problem.lb=low_bou;
            problem.ub=up_bou;
            problem.nonlcon=con_fcn;
            problem.options=self.fmincon_option;
            ms=MultiStart('Display','off');
            rs=CustomStartPointSet([x_init;lhsdesign(19,vari_num).*(up_bou-low_bou)+low_bou]);

            % search each fidelity min infill value
            x_infill_list=zeros(fid_num,vari_num);
            fval_infill_list=zeros(fid_num,1);
            for fid_idx=1:fid_num
                obj_fcn=@(x) obj_fid_fcn(x,fid_idx);
                problem.objective=obj_fcn;
                [x_infill_list(fid_idx,:),fval_infill_list(fid_idx),exit_flag]=run(ms,problem,rs);

                if (exit_flag == -2 && strcmp(self.constraint,'auto')) ...
                        || norm(x_infill_list(fid_idx,:)-x_init) < self.add_torl
                    obj_fcn=@(x) infill_fcn(x,fid_idx);
                    problem.objective=obj_fcn;
                    problem.nonlcon=self.con_fcn_srgt;
                    rs=CustomStartPointSet([x_init;lhsdesign(19,vari_num).*(up_bou-low_bou)+low_bou]);
                    [x_infill_list(fid_idx,:),fval_infill_list(fid_idx),~]=run(ms,problem,rs);

                    % return to normal
                    problem.nonlcon=con_fcn;
                end
            end

            [~,fid_infill]=min(fval_infill_list);
            x_infill=x_infill_list(fid_infill,:);

            function alpha_1=corrFcn(x,fid_idx,var_fid_fcn,predict_list)
                Cov_pred=var_fid_fcn(x,fid_idx);
                [~,Var_pred_fid]=predict_list{fid_idx}(x);
                [~,Var_pred]=predict_list{end}(x);
                alpha_1=Cov_pred./sqrt(Var_pred_fid.*Var_pred);
                alpha_1(Var_pred_fid < eps | Var_pred < eps)=0;
            end

            function alpha_3=costFcn(~,fid_idx,cost_list)
                alpha_3=cost_list(end)/cost_list(fid_idx);
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

            function POF=POFFcn(x,con_fcn_srgt)
                % calculate feasible probability
                %
                POF=1;
                [con_pred,~,con_var,~]=con_fcn_srgt(x);
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
        function [obj_fcn_srgt,con_fcn_srgt,Srgt_obj,Srgt_con,Srgt_coneq]=getSrgtFcnCoKRG...
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
                Srgt_obj{obj_idx}=srgtExCoKRG(x_list,loadIdx(obj_list,obj_idx),Srgt_obj{obj_idx});
            end

            % generate con surrogate
            if ~isempty(con_list{1})
                if isempty(Srgt_con),Srgt_con=cell(size(con_list,2),1);end
                for con_idx=1:size(con_list,2)
                    Srgt_con{con_idx}=srgtExCoKRG(x_list,loadIdx(con_list,con_idx),Srgt_con{con_idx});
                end
            else
                Srgt_con=[];
            end

            % generate coneq surrogate
            if ~isempty(coneq_list{1})
                if isempty(Srgt_coneq),Srgt_coneq=cell(size(coneq_list,2),1);end
                for coneq_idx=1:size(coneq_list,2)
                    Srgt_coneq{coneq_idx}=srgtExCoKRG(x_list,loadIdx(coneq_list,coneq_idx),Srgt_coneq{coneq_idx});
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

            function fval=loadIdx(fval_list,idx)
                % load specific idx from cell
                %
                fval=cell(length(fval_list),1);
                for fid_idx=1:length(fval_list)
                    fval{fid_idx}=fval_list{fid_idx}(:,idx);
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

