classdef OptimMPSNMF < handle
    % MPS-NMF optimization algorithm
    % Model_function is cell of multi fidelity function handle
    % Cost is array of each fidelity model function
    % from start to end is high to low fidelity
    %
    % Copyright 2023 Adel
    %
    % abbreviation:
    % obj: objective, con: constraint, iter: iteration, tor: torlance
    % fcn: function
    % lib: library, init: initial, rst: restart, potl: potential
    %
    properties % basic parameter
        NFE_max;
        iter_max;
        obj_tol;
        con_tol;

        datalib_HF; % X, Obj, Con, Coneq, Vio
        datalib_LF; % X, Obj, Con, Coneq, Vio
        dataoptim; % NFE, Add_idx, Iter

        obj_fcn_srgt;
        con_fcn_srgt;
        vio_fcn_srgt;

        Srgt_obj;
        Srgt_con;
        Srgt_coneq;
        Srgt_vio;
    end

    properties % problem parameter
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
        X_init_HF=[];
        X_init_LF=[];

        % hyper parameter
        sample_num_init_HF=[];
        sample_num_init_LF=[];
        sample_num_add_HF=[];
        sample_num_add_LF=[];

        trial_num=[];
        sel_prblty_init=[];

        coord_init=[];
        coord_min=[];
        coord_max=[];

        tau_success=[];
        tau_fail=[];
        kappa=[];
        w_list=[];

        % surrogate information
        KRGQdPg_option=struct();
        adapt_hyp=false;
    end

    methods % main
        function self=OptimMPSNMF(NFE_max,iter_max,obj_tol,con_tol)
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
            % multi input
            varargin=[varargin,repmat({[]},1,7-length(varargin))];
            [objcon_fcn_LF,objcon_fcn_HF,vari_num,low_bou,up_bou,cost_LF,cost_HF]=varargin{:};
            if isa(objcon_fcn_LF,'function_handle'),objcon_fcn_LF={objcon_fcn_LF};end
            fid_num=2;LF_num=length(objcon_fcn_LF);

            % NFE and iteration setting
            if isempty(self.NFE_max),self.NFE_max=10+10*vari_num;end
            if isempty(self.iter_max),self.iter_max=20+20*vari_num;end

            if isempty(self.dataoptim)
                self.dataoptim.NFE=0;
                self.dataoptim.Add_idx=[];
                self.dataoptim.iter=0;
                self.dataoptim.done=false;
            end

            % hyper parameter
            if isempty(self.sample_num_init_HF),if vari_num < 7,self.sample_num_init_HF=(vari_num+1)*(vari_num+2)/2+1;
            else,self.sample_num_init_HF=vari_num*6;end;end
            if isempty(self.sample_num_init_LF),self.sample_num_init_LF=ceil(self.sample_num_init_HF*cost_HF./cost_LF);end

            if isempty(self.sample_num_add_HF),self.sample_num_add_HF=max(floor(vari_num/6),1);end
            if isempty(self.sample_num_add_LF),self.sample_num_add_LF=ceil(self.sample_num_add_HF*cost_HF./cost_LF);end

            if isempty(self.trial_num),self.trial_num=min(1000*vari_num,20000);end
            if isempty(self.sel_prblty_init),self.sel_prblty_init=min(20/vari_num,1);end
            if isempty(self.coord_init),self.coord_init=0.2*(up_bou-low_bou);end
            if isempty(self.coord_min),self.coord_min=0.2*1/64*(up_bou-low_bou);end
            if isempty(self.coord_max),self.coord_max=2*(up_bou-low_bou);end
            if isempty(self.tau_success),self.tau_success=3;end
            if isempty(self.tau_fail),self.tau_fail=max(vari_num,5);end
            if isempty(self.kappa),self.kappa=4;end
            if isempty(self.w_list),self.w_list=[0.3,0.5,0.8,0.95];end

            self.add_tol=5e-5*sqrt(vari_num);

            ga_option=optimoptions('ga','Display','none','ConstraintTolerance',self.con_tol,'MaxGenerations',10);
            fmincon_option=optimoptions('fmincon','Display','none','Algorithm','sqp','FiniteDifferenceStepSize',1e-5,'FunctionTolerance',self.obj_tol,'ConstraintTolerance',self.con_tol);

            % step 2, latin hypercube sample
            self.sampleInit(objcon_fcn_LF,objcon_fcn_HF,vari_num,low_bou,up_bou,cost_LF,cost_HF);

            % step 3-4, find feasible sample
            X_LF=cell(1,LF_num);Obj_LF=cell(1,LF_num);Con_LF=cell(1,LF_num);Coneq_LF=cell(1,LF_num);Vio_LF=cell(1,LF_num);
            X_HF=[];Obj_HF=[];Con_HF=[];Coneq_HF=[];Vio_HF=[];
            while all(self.datalib_HF.Vio > 0)
                % forced interrupt
                if self.dataoptim.NFE >= self.NFE_max
                    self.dataoptim.done=true;
                    break;
                end

                for LF_idx=1:LF_num
                    [X_LF{LF_idx},Obj_LF{LF_idx},Con_LF{LF_idx},Coneq_LF{LF_idx},Vio_LF{LF_idx}]=self.datalibLoad(self.datalib_LF{LF_idx},low_bou,up_bou);
                end
                [X_HF,Obj_HF,Con_HF,Coneq_HF,Vio_HF]=self.datalibLoad(self.datalib_HF,low_bou,up_bou);

                % optimize KS
                self.Srgt_vio={srgtdfKRGQdPg(X_LF,Vio_LF,X_HF,Vio_HF)};
                self.vio_fcn_srgt=@(x) self.Srgt_vio{1}.predict(x);
                con_fcn_dist=@(x)self.add_tol^2-min(sum(([cell2mat(X_LF');X_HF]-x).^2,2));
                nonlcon_fcn=@(x)deal(con_fcn_dist(x),[]);
                [x_infill,vio_infill,exit_flag,output_ga]=ga(self.vio_fcn_srgt,vari_num,[],[],[],[],low_bou,up_bou,nonlcon_fcn,ga_option);
                [x_infill,vio_infill,exit_flag,output_fmincon]=fmincon(self.vio_fcn_srgt,x_infill,[],[],[],[],low_bou,up_bou,nonlcon_fcn,fmincon_option);

                for LF_idx=1:LF_num
                    objcon_fcn_LF_item=objcon_fcn_LF{LF_idx};
                    self.datalib_LF{LF_idx}=self.sample(self.datalib_LF{LF_idx},objcon_fcn_LF_item,x_infill,cost_LF(LF_idx),2);
                end
                [self.datalib_HF,X,Obj,Con,Coneq,Vio]=self.sample(self.datalib_HF,objcon_fcn_HF,x_infill,cost_HF,1);
            end

            % step 5-7, adaptive sample

            % initialize all data to begin optimize
            for LF_idx=1:LF_num
                [X_LF{LF_idx},Obj_LF{LF_idx},Con_LF{LF_idx},Coneq_LF{LF_idx},Vio_LF{LF_idx}]=self.datalibLoad(self.datalib_LF{LF_idx},low_bou,up_bou);
            end
            [X_HF,Obj_HF,Con_HF,Coneq_HF,Vio_HF]=self.datalibLoad(self.datalib_HF,low_bou,up_bou);
            obj_num=size(Obj_HF,2);con_num=size(Con_HF,2);coneq_num=size(Coneq_HF,2);vio_num=size(Vio_HF,2);
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

            self.Srgt_obj=repmat({self.KRGQdPg_option},obj_num,1);
            self.Srgt_con=repmat({self.KRGQdPg_option},con_num,1);
            self.Srgt_coneq=repmat({self.KRGQdPg_option},coneq_num,1);

            self.dataoptim.iter=self.dataoptim.iter+1;
            self.dataoptim.coord=self.coord_init;
            self.dataoptim.C_success=0;
            self.dataoptim.C_fail=0;
            while ~self.dataoptim.done
                % step 5, optimize by sqp
                % construct kriging-Quadprogram model
                [self.obj_fcn_srgt,self.con_fcn_srgt,...
                    self.Srgt_obj,self.Srgt_con,self.Srgt_coneq]=getSrgtFcnDF...
                    (X_LF,Obj_LF,Con_LF,Coneq_LF,X_HF,Obj_HF,Con_HF,Coneq_HF,...
                    self.Srgt_obj,self.Srgt_con,self.Srgt_coneq);
                self.Srgt_vio={srgtdfKRGQdPg(X_LF,Vio_LF,X_HF,Vio_HF)};
                self.vio_fcn_srgt=@(x)self.Srgt_vio{1}.predict(x);

                % search potential best
                best_idx=self.datalib_HF.Best_idx(end);x_best=X_HF(best_idx,:);
                [x_infill,~,exit_flag,output_fmincon]=fmincon(self.obj_fcn_srgt,x_best,[],[],[],[],low_bou,up_bou,self.con_fcn_srgt,fmincon_option);
                x_infill=max(x_infill,low_bou);
                x_infill=min(x_infill,up_bou);
                for LF_idx=1:LF_num
                    objcon_fcn_LF_item=objcon_fcn_LF{LF_idx};
                    self.datalib_LF{LF_idx}=self.sample(self.datalib_LF{LF_idx},objcon_fcn_LF_item,x_infill,cost_LF(LF_idx),2);
                end
                [self.datalib_HF,x_infill,obj_infill,con_infill,coneq_infill,vio_infill]=self.sample(self.datalib_HF,objcon_fcn_HF,x_infill,cost_HF,1);
                for LF_idx=1:LF_num
                    [X_LF{LF_idx},Obj_LF{LF_idx},Con_LF{LF_idx},Coneq_LF{LF_idx},Vio_LF{LF_idx}]=self.datalibLoad(self.datalib_LF{LF_idx},low_bou,up_bou);
                end
                [X_HF,Obj_HF,Con_HF,Coneq_HF,Vio_HF]=self.datalibLoad(self.datalib_HF,low_bou,up_bou);

                % step 6
                % find best result to record
                best_idx=self.datalib_HF.Best_idx(end);
                x_best=X_HF(best_idx,:);
                result_X(self.dataoptim.iter,:)=x_best;
                obj_best=Obj_HF(best_idx,:);
                result_Obj(self.dataoptim.iter,:)=obj_best;
                if con_num,result_Con(self.dataoptim.iter,:)=Con_HF(best_idx,:);end
                if coneq_num,result_Coneq(self.dataoptim.iter,:)=Coneq_HF(best_idx,:);end
                if vio_num,vio_best=Vio_HF(best_idx,:);result_Vio(self.dataoptim.iter,:)=vio_best;end
                self.dataoptim.iter=self.dataoptim.iter+1;

                if self.FLAG_DRAW_FIGURE && vari_num < 3
                    cla();
                    displaySrgt([],self.Srgt_obj{1},low_bou,up_bou);
                    % line(x_infill(1),x_infill(2),obj_infill,'Marker','o','color','r');
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

                % step7, DCP
                if ~self.dataoptim.done
                    [self.obj_fcn_srgt,self.con_fcn_srgt,...
                        self.Srgt_obj,self.Srgt_con,self.Srgt_coneq]=getSrgtFcnDF...
                        (X_LF,Obj_LF,Con_LF,Coneq_LF,X_HF,Obj_HF,Con_HF,Coneq_HF,...
                        self.Srgt_obj,self.Srgt_con,self.Srgt_coneq);
                    self.Srgt_vio={srgtdfKRGQdPg(X_LF,Vio_LF,X_HF,Vio_HF)};
                    self.vio_fcn_srgt=@(x)self.Srgt_vio{1}.predict(x);

                    % step 1, calculate gradient by Kriging hyperparameter
                    grad=(self.Srgt_obj{1}.hyp_bias+self.Srgt_vio{1}.hyp_bias)/2;
                    grad=abs(grad);
                    grad_max=max(grad);grad_min=min(grad);
                    grad=(grad-grad_min)./(grad_max-grad_min);

                    sel_prob=self.sel_prblty_init*(1-log(self.dataoptim.NFE-self.sample_num_init_HF+1)/log(self.NFE_max-self.sample_num_init_HF));

                    % step 2, select coords to perturb and generate trial point
                    sel_list=[];
                    for vari_idx=1:vari_num
                        if rand() < sel_prob*grad(vari_idx)
                            sel_list=[sel_list,vari_idx];
                        end
                    end
                    if isempty(sel_list),sel_list=randi(vari_num);end

                    X_trial=repmat(x_best,self.trial_num,1);
                    for LF_sel_idx=1:length(sel_list)
                        sel=sel_list(LF_sel_idx);
                        X_trial(:,sel)=X_trial(:,sel)+...
                            normrnd(0,self.dataoptim.coord(sel),[self.trial_num,1]);
                    end
                    X_trial=max(X_trial,low_bou);
                    X_trial=min(X_trial,up_bou);

                    % step 3, construct filter and filter out trial point
                    pareto_idx_list=getParetoFront(Obj_HF,Vio_HF);
                    X_trial_obj_srgt=self.obj_fcn_srgt(X_trial);
                    X_trial_vio_srgt=self.vio_fcn_srgt(X_trial);
                    in_bool=filterParetoFront(X_trial_obj_srgt,X_trial_vio_srgt,Obj_HF(pareto_idx_list),Vio_HF(pareto_idx_list));
                    X_filter=X_trial(in_bool,:);

                    num_add_LF=self.sample_num_add_LF;
                    num_add_HF=self.sample_num_add_HF;
                    if ~isempty(X_filter)
                        % step 4, select add add
                        add_num=sum(num_add_LF);
                        ft_num=size(X_filter,1);
                        if ft_num <= LF_num
                            X_add_HF=X_filter;
                            X_add_LF=repmat({X_filter},1,LF_num);
                        else
                            if ft_num <= add_num
                                X_sel_best=X_filter;
                                num_add_LF=floor(ft_num*num_add_LF/add_num);
                            else
                                % select point to add
                                w_idx=mod(round(self.dataoptim.NFE)-self.sample_num_init_HF+1,self.kappa);
                                if w_idx == 0
                                    w_R=self.w_list(self.kappa);
                                else
                                    w_R=self.w_list(w_idx);
                                end
                                w_D=1-w_R;

                                % evaluate trial point merit
                                merit_list=meritFcn(self.obj_fcn_srgt,X_filter,[cell2mat(X_LF');X_HF],w_R,w_D);
                                [~,order]=sort(merit_list);
                                X_sel_best=X_filter(order(1:add_num),:);
                            end
                            [X_idx,X_cntr]=kmeans(X_sel_best,num_add_HF);

                            % base on distance to select HF add point
                            dist=zeros(size(X_sel_best,1),size(X_cntr,1));
                            for vari_idx=1:vari_num
                                dist=dist+(X_sel_best(:,vari_num)-X_cntr(:,vari_num)').^2;
                            end
                            [~,HF_sel_idx]=min(dist,[],1);
                            X_add_HF=X_sel_best(HF_sel_idx,:);
                            X_add_LF=cell(1,LF_num);
                            remain_idx=1:size(X_sel_best,1);
                            % remain_idx(HF_sel_idx)=[];
                            for LF_idx=1:LF_num
                                LF_sel_idx=randperm(length(remain_idx),num_add_LF(LF_idx));
                                X_add_LF{LF_idx}=[X_sel_best(remain_idx(LF_sel_idx),:);X_add_HF];
                                remain_idx(LF_sel_idx)=[];
                            end
                        end

                        % add new sample point
                        for LF_idx=1:LF_num
                            objcon_fcn_LF_item=objcon_fcn_LF{LF_idx};
                            self.datalib_LF{LF_idx}=self.sample(self.datalib_LF{LF_idx},objcon_fcn_LF_item,X_add_LF{LF_idx},cost_LF(LF_idx),2);
                        end
                        self.datalib_HF=self.sample(self.datalib_HF,objcon_fcn_HF,X_add_HF,cost_HF,1);
                        for LF_idx=1:LF_num
                            [X_LF{LF_idx},Obj_LF{LF_idx},Con_LF{LF_idx},Coneq_LF{LF_idx},Vio_LF{LF_idx}]=self.datalibLoad(self.datalib_LF{LF_idx},low_bou,up_bou);
                        end
                        [X_HF,Obj_HF,Con_HF,Coneq_HF,Vio_HF]=self.datalibLoad(self.datalib_HF,low_bou,up_bou);

                        % step 5, adjust step size
                        if obj_infill < obj_best
                            self.dataoptim.C_success=self.dataoptim.C_success+1;
                            self.dataoptim.C_fail=0;
                        else
                            self.dataoptim.C_success=0;
                            self.dataoptim.C_fail=self.dataoptim.C_fail+1;
                        end

                        if self.dataoptim.C_success >= self.tau_success
                            self.dataoptim.coord=min(2*self.dataoptim.coord,self.coord_max);
                            self.dataoptim.C_success=0;
                        end

                        if self.dataoptim.C_fail >= self.tau_fail
                            self.dataoptim.coord=max(self.dataoptim.coord/2,self.coord_min);
                            self.dataoptim.C_fail=0;
                        end
                    end

                end

                % save iteration
                if ~isempty(self.dataoptim_filestr)
                    datalib_LF=self.datalib_LF;
                    datalib_HF=self.datalib_HF;
                    dataoptim=self.dataoptim;
                    save(self.dataoptim_filestr,'datalib_LF','datalib_HF','dataoptim');
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
            output.datalib_LF=self.datalib_LF;
            output.datalib_HF=self.datalib_HF;
            output.dataoptim=self.dataoptim;

            function fval=meritFcn(obj_fcn_srgt,X_trial,X,w_R,w_D)
                % function to evaluate sample point
                %

                % value scale
                R=obj_fcn_srgt(X_trial);
                R_min=min(R); R_max=max(R);
                R=(R-R_min)./(R_max-R_min);

                % distance scale
                dis=zeros(size(X_trial,1),size(X,1));
                for vari_i=1:size(X,2)
                    dis=dis+(X_trial(:,vari_i)-X(:,vari_i)').^2;
                end
                D=min(sqrt(dis),[],2);
                D_min=min(D); D_max=max(D);
                D=(D_max-D)./(D_max-D_min);

                fval=w_R*R+w_D*D;
            end
        end

        function sampleInit(self,objcon_fcn_LF,objcon_fcn_HF,vari_num,low_bou,up_bou,cost_LF,cost_HF)
            % initial latin hypercube sample
            %
            fid_num=2;LF_num=length(objcon_fcn_LF);

            % obtain datalib
            if isempty(self.datalib_HF),self.datalib_HF=[];end
            if isempty(self.datalib_LF),self.datalib_LF=cell(1,LF_num);end
            if isempty(self.X_init_HF),self.X_init_HF=[];end
            if isempty(self.X_init_LF),self.X_init_LF=cell(1,LF_num);end

            % low fidelity sample
            X_init_LF_total=[];
            for LF_idx=1:LF_num
                datalib=self.datalib_LF{LF_idx};
                X_init_LF_item=self.X_init_LF{LF_idx};
                sample_num_init_LF_item=self.sample_num_init_LF(LF_idx);
                objcon_fcn_LF_item=objcon_fcn_LF{LF_idx};

                if isempty(datalib)
                    % filestr=self.datalib_filestr_LF{LF_idx};
                    datalib=self.datalibGet(vari_num,low_bou,up_bou,self.con_tol);
                end

                if size(datalib.X,1) < sample_num_init_LF_item
                    if isempty(X_init_LF_item)
                        % use latin hypercube method to get enough initial sample x_list
                        sample_num=min(sample_num_init_LF_item-size(datalib.X,1),floor(self.NFE_max-self.dataoptim.NFE*cost_LF(LF_idx)));
                        X_init_LF_item=lhsdesign(sample_num,vari_num,'iterations',50,'criterion','maximin').*(up_bou-low_bou)+low_bou;
                    end

                    % updata data lib by x_list
                    datalib=self.sample(datalib,objcon_fcn_LF_item,X_init_LF_item,cost_LF(LF_idx),2);
                end

                X_init_LF_total=[X_init_LF_total;datalib.X];
                self.datalib_LF{LF_idx}=datalib;
                self.X_init_LF{LF_idx}=X_init_LF_item;
            end

            % high fidelity sample
            if isempty(self.datalib_HF)
                % filestr=self.datalib_filestr_LF{LF_idx};
                self.datalib_HF=self.datalibGet(vari_num,low_bou,up_bou,self.con_tol);
            end

            if size(self.datalib_HF.X,1) < self.sample_num_init_HF
                if isempty(self.X_init_HF)
                    % use latin hypercube method to get enough initial sample x_list
                    sample_num=min(self.sample_num_init_HF-size(self.datalib_HF.X,1),floor(self.NFE_max-self.dataoptim.NFE*cost_HF));
                    self.X_init_HF=lhdNSLE(X_init_LF_total,sample_num,vari_num,low_bou,up_bou,self.datalib_HF.X);
                end

                % updata data lib by x_list
                self.datalib_HF=self.sample(self.datalib_HF,objcon_fcn_HF,self.X_init_HF,cost_HF,1);
            end

            % detech expensive constraints
            if ~isempty(self.datalib_HF.Vio)
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

        function [datalib,x,obj,con,coneq,vio]=datalibAdd(datalib,objcon_fcn,x)
            % add new x into data lib
            %
            [obj,con,coneq]=objcon_fcn(x);vio=[]; % eval value

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
            if any(Best_idx > size(datalib.X,1))
                error('?');
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

%% common function

function [obj_fcn_srgt,con_fcn_srgt,Srgt_obj,Srgt_con,Srgt_coneq]=getSrgtFcnDF...
    (X_LF,Obj_LF,Con_LF,Coneq_LF,X_HF,Obj_HF,Con_HF,Coneq_HF,Srgt_obj,Srgt_con,Srgt_coneq)
% generate surrogate function of objective and constraints
%
% output:
% obj_fcn_srgt(output is obj_pred, obj_var),...
% con_fcn_srgt(output is con_pred, coneq_pred, con_var, coneq_var)
%

% generate obj surrogate
obj_num=size(Obj_HF,2);Obj_LF=rotateCellToMat(Obj_LF);
if isempty(Srgt_obj),Srgt_obj=cell(obj_num,1);end
for obj_idx=1:obj_num
    Srgt_obj{obj_idx}=srgtdfKRGQdPg(X_LF,Obj_LF{obj_idx},X_HF,Obj_HF(:,obj_idx),Srgt_obj{obj_idx});
end

% generate con surrogate
if ~isempty(Con_HF)
    con_num=size(Con_HF,2);Con_LF=rotateCellToMat(Con_LF);
    if isempty(Srgt_con),Srgt_con=cell(con_num,1);end
    for con_idx=1:con_num
        Srgt_con{con_idx}=srgtdfKRGQdPg(X_LF,Con_LF{con_idx},X_HF,Con_HF(:,con_idx),Srgt_con{con_idx});
    end
else
    Srgt_con=[];
end

% generate coneqeq surrogate
if ~isempty(Coneq_HF)
    coneq_num=size(Coneq_HF,2);Coneq_LF=rotateCellToMat(Coneq_LF);
    if isempty(Srgt_coneq),Srgt_coneq=cell(coneq_num,1);end
    for coneq_idx=1:coneq_num
        Srgt_coneq{coneq_idx}=srgtdfKRGQdPg(X_LF,Coneq_LF{coneq_idx},X_HF,Coneq_HF(:,coneq_idx),Srgt_coneq{coneq_idx});
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
        if nargout < 2
            for con_i=1:length(Srgt_obj)
                [Obj_pred(:,con_i)]=Srgt_obj{con_i}.predict(X_pred);
            end
        else
            Obj_var=zeros(size(X_pred,1),length(Srgt_obj));
            for con_i=1:length(Srgt_obj)
                [Obj_pred(:,con_i),Obj_var(:,con_i)]=Srgt_obj{con_i}.predict(X_pred);
            end
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

            if nargout < 4
                for con_i=1:length(Srgt_con)
                    [Con_pred(:,con_i)]=Srgt_con{con_i}.predict(X_pred);
                end
            else
                Con_var=zeros(size(X_pred,1),length(Srgt_con));
                for con_i=1:length(Srgt_con)
                    [Con_pred(:,con_i),Con_var(:,con_i)]=Srgt_con{con_i}.predict(X_pred);
                end
            end
        end
        if isempty(Srgt_coneq)
            Coneq_pred=[];
            Coneq_var=[];
        else
            Coneq_pred=zeros(size(X_pred,1),length(Srgt_coneq));

            if nargout < 5
                for coneq_i=1:length(Srgt_coneq)
                    [Coneq_pred(:,coneq_i)]=Srgt_coneq{coneq_i}.predict(X_pred);
                end
            else
                Coneq_var=zeros(size(X_pred,1),length(Srgt_coneq));
                for coneq_i=1:length(Srgt_coneq)
                    [Coneq_pred(:,coneq_i),Coneq_var(:,coneq_i)]=Srgt_coneq{coneq_i}.predict(X_pred);
                end
            end
        end
    end

    function mat=rotateCellToMat(orig)
        % orig have each fidelity
        %
        [x_num,f_num]=size(orig{1});fid_num=length(orig);
        mat=cell(1,f_num); % all fval num
        for f_idx=1:f_num
            mat_f=cell(1,fid_num); % each fval
            for d_idx=1:fid_num
                mat_f{d_idx}=orig{d_idx}(:,f_idx);
            end
            mat{f_idx}=mat_f;
        end
    end
end

function PF_idx_list=getParetoFront(obj_list,vio_list)
% distinguish pareto front of data list
% dominate define as followed
% Solution i is feasible and solution j is not.
% Solutions i and j are both infeasible,...
% but solution i has a smaller overall constraint violation.
% Solutions i and j are feasible and solution i dominates solution j
%
x_num=size(obj_list,1);
PF_idx_list=[]; % sort all idx of filter point list

% select no domain filter
for x_idx=1:x_num
    obj=obj_list(x_idx,:);
    vio=vio_list(x_idx,:);

    PF_idx=1;
    add_filter_flag=true(1);
    while PF_idx <= length(PF_idx_list)
        % compare x with exit pareto front point
        x_PF_idx=PF_idx_list(PF_idx,:);

        % contain constraint of x_filter
        obj_PF=obj_list(x_PF_idx,:);
        vio_PF=vio_list(x_PF_idx,:);

        % compare x with x_pareto
        if vio_PF <= 0
            if obj > obj_PF || vio > 0
                add_filter_flag=false(1);
                break;
            end
        else
            if obj > obj_PF && vio > vio_PF
                add_filter_flag=false(1);
                break;
            end
        end

        % if better than exit pareto point,reject pareto point
        delete_filter_flag=false(1);
        if vio <= 0
            if obj_PF > obj || vio_PF > 0
                delete_filter_flag=true(1);
            end
        else
            if obj_PF > obj && vio_PF > vio
                delete_filter_flag=true(1);
            end
        end
        if delete_filter_flag
            PF_idx_list(PF_idx)=[];
            PF_idx=PF_idx-1;
        end

        PF_idx=PF_idx+1;
    end

    % add into pareto list if possible
    if add_filter_flag
        PF_idx_list=[PF_idx_list;x_idx];
    end
end

end

function in_bool=filterParetoFront(obj_list,vio_list,obj_list_PF,vio_list_PF)
% filter point by pareto front
%
x_num=size(obj_list,1);
pareto_num=size(obj_list_PF,1);
in_bool=true(x_num,1);

% select no domain filter
for x_idx=1:x_num
    obj=obj_list(x_idx,:);
    vio=vio_list(x_idx,:);

    PF_idx=1;
    while PF_idx <= pareto_num
        % compare x with exit pareto front point

        % contain constraint of x_filter
        obj_PF=obj_list_PF(PF_idx,:);
        vio_PF=vio_list_PF(PF_idx,:);

        % compare x with x_pareto
        if vio_PF <= 0
            if obj > obj_PF || vio > 0
                in_bool(x_idx)=false(1);
                break;
            end
        else
            if obj > obj_PF && vio > vio_PF
                in_bool(x_idx)=false(1);
                break;
            end
        end

        PF_idx=PF_idx+1;
    end
end
end

function listMatPntAdd = choose_hf_lf(obj, matPntAdd, numPntAddHF)
    % Choose newly added HF/LF data from a dataset.
    
    LF_num = obj.options.numLvl;
    listMatPntAdd = cell(1, LF_num);
    
    [cntr, ~, ~, ~, ~, ~, ~] = cmeans(matPntAdd', numPntAddHF, 2.0, 5e-4, 1000.0);
    neigh = nearestNeighbors(matPntAdd, 1);
    idxHFTmp = unique(neigh.kneighbors(cntr));
    numHFThis = length(idxHFTmp);
    matPntAddHF = matPntAdd(idxHFTmp, :);
    
    idxSup = [];
    if numHFThis < numPntAddHF
        idxTmp = setdiff(1:size(matPntAdd, 1), idxHFTmp);
        idxSup = randperm(length(idxTmp), numPntAddHF - numHFThis);
        matPntAddHF = [matPntAddHF; matPntAdd(idxTmp(idxSup), :)];
    end
    
    matPntAddLF = matPntAdd(setdiff(1:size(matPntAdd, 1), [idxHFTmp; idxSup]), :);
    matIdxLF = reshape(randperm(floor(size(matPntAddLF, 1) / (LF_num - 1)) * (LF_num - 1)), LF_num - 1, []);
    
    for idx = 1:LF_num-1
        listMatPntAdd{idx} = [matPntAddLF(matIdxLF(idx, :), :); matPntAddHF];
    end
    listMatPntAdd{end} = matPntAddHF;
end

function neigh = nearestNeighbors(data, k)
    % Create a nearest neighbors object
    neigh = fitcknn(data, 'NumNeighbors', k);
end


