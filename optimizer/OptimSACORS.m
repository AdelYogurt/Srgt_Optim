classdef OptimSACORS < handle
    % SACO-RS optimization algorithm
    % only support constraints problem
    %
    % Copyright 2023 Adel
    %
    % abbreviation:
    % obj: objective, con: constraint, iter: iteration, torl: torlance
    % fcn: function, srgt: surrogate
    % lib: library, init: initial, rst: restart, pot: potential
    %

    % basic parameter
    properties
        NFE_max;
        iter_max;
        obj_torl;
        con_torl;

        datalib; % X, Obj, Con, Coneq, Vio
        dataoptim; % NFE, Add_idx, Iter, detect_pot, X_pot, Obj_pot, Vio_pot, Idx_best

        obj_fcn_srgt;
        con_fcn_srgt;

        Srgt_obj;
        Srgt_con;
        Srgt_coneq;

        GPC_pareto;
        GPC_conv;
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
        sample_num_rst=[];
        trial_num=[];

        identiy_torl=1e-2; % if inf norm of point less than identiy_torlance, point will be consider as same local best
        min_r_interest=1e-3;
        max_r_interest=1e-1;

        GPC_option=struct('simplify_hyp',true,'hyp',[])
    end

    % main function
    methods
        function self=OptimSACORS(NFE_max,iter_max,obj_torl,con_torl)
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
                    if ~contains(prob_field,'objcon_fcn'), error('OptimSACORS.optimize: input problem lack objcon_fcn'); end
                    objcon_fcn=problem.objcon_fcn;
                    if ~contains(prob_field,'vari_num'), error('OptimSACORS.optimize: input problem lack vari_num'); end
                    if ~contains(prob_field,'low_bou'), error('OptimSACORS.optimize: input problem lack low_bou'); end
                    if ~contains(prob_field,'up_bou'), error('OptimSACORS.optimize: input problem lack up_bou'); end
                else
                    prob_method=methods(problem);
                    if ~contains(prob_method,'objcon_fcn'), error('OptimSACORS.optimize: input problem lack objcon_fcn'); end
                    objcon_fcn=@(x) problem.objcon_fcn(x);
                    prob_pro=properties(problem);
                    if ~contains(prob_pro,'vari_num'), error('OptimSACORS.optimize: input problem lack vari_num'); end
                    if ~contains(prob_pro,'low_bou'), error('OptimSACORS.optimize: input problem lack low_bou'); end
                    if ~contains(prob_pro,'up_bou'), error('OptimSACORS.optimize: input problem lack up_bou'); end
                end
                vari_num=problem.vari_num;
                low_bou=problem.low_bou;
                up_bou=problem.up_bou;
            else
                % multi input
                varargin=[varargin,repmat({[]},1,4-length(varargin))];
                [objcon_fcn,vari_num,low_bou,up_bou]=varargin{:};
            end

            % NFE and iteration setting
            if isempty(self.NFE_max),self.NFE_max=10+10*vari_num;end
            if isempty(self.iter_max),self.iter_max=20+20*vari_num;end

            % hyper parameter
            if isempty(self.sample_num_init),self.sample_num_init=6+3*vari_num;end
            if isempty(self.sample_num_add),self.sample_num_add=ceil(log(self.sample_num_init)/2);end
            if isempty(self.sample_num_rst),self.sample_num_rst=self.sample_num_init;end
            if isempty(self.trial_num),self.trial_num=min(100*vari_num,100);end

            if isempty(self.dataoptim)
                self.dataoptim.NFE=0;
                self.dataoptim.Add_idx=[];
                self.dataoptim.iter=0;
                self.dataoptim.done=false;
                self.dataoptim.detect_pot=true;
                self.dataoptim.X_pot=[];
                self.dataoptim.Obj_pot=[];
                self.dataoptim.Vio_pot=[];
                self.dataoptim.Idx_best=[];
            end

            % step 2, generate initial data library
            self.sampleInit(objcon_fcn,vari_num,low_bou,up_bou);

            % step 3-7, adaptive samlping base on optimize strategy
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

            self.GPC_pareto=self.GPC_option;
            self.GPC_conv=self.GPC_option;

            self.dataoptim.iter=self.dataoptim.iter+1;
            self.datalib.Bool_conv=false(size(X,1),1);
            while ~self.dataoptim.done
                % step 3
                % construct surrogate objcon_fcn
                [self.obj_fcn_srgt,self.con_fcn_srgt,...
                    self.Srgt_obj,self.Srgt_con,self.Srgt_coneq]=self.getSrgtFcnRBF...
                    (X,Obj,Con,Coneq);

                % step 4
                % updata or identify potential local best
                self.updataLocalPotential(X,vari_num,low_bou,up_bou);

                % step 5
                % select best potential point as x_infill and updata data_lib
                x_infill=self.dataoptim.X_pot(1,:);
                [self.datalib,x_infill,obj_infill,~,~,vio_infill,repeat_idx]=self.sample(self.datalib,objcon_fcn,x_infill);
                self.datalib.Bool_conv=[self.datalib.Bool_conv;false(sum(repeat_idx == 0),1)];
                [X,Obj,Con,Coneq,Vio]=self.datalibLoad(self.datalib,low_bou,up_bou);
                if repeat_idx == 0,idx_infill=size(self.datalib.X,1);
                else,idx_infill=repeat_idx;end

                % find best result to record
%                 Best_idx_unconv=self.datalib.Best_idx(~Bool_conv);
%                 best_idx=Best_idx_unconv(end);
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

                % step 6 and 7
                if ~self.dataoptim.done
                    X_add=[];

                    % check if converage
                    if (self.dataoptim.iter > 2) &&...
                            ( abs((obj_infill-obj_infill_old)/obj_infill_old) < self.obj_torl && ...
                            ((~isempty(vio_infill) && vio_infill == 0) || isempty(vio_infill)) )
                        % resample LHD
                        % step 7.1
                        self.dataoptim.Idx_best=[self.dataoptim.Idx_best;idx_infill];
                        self.dataoptim.X_pot=self.dataoptim.X_pot(2:end,:);
                        self.dataoptim.Obj_pot=self.dataoptim.Obj_pot(2:end,:);
                        self.dataoptim.Vio_pot=self.dataoptim.Vio_pot(2:end,:);

                        % if potential point list is empty, restart detect
                        if isempty(self.dataoptim.X_pot),self.dataoptim.detect_pot=true;end

                        % step 7.2
                        % judge converage X
                        fmincon_option=optimoptions('fmincon','Display','none','Algorithm','interior-point','ConstraintTolerance',self.con_torl);
                        for x_idx=1:size(X,1)
                            if ~self.datalib.Bool_conv(x_idx)
                                x_conv=fmincon(self.obj_fcn_srgt,X(x_idx,:),[],[],[],[],...
                                    low_bou,up_bou,self.con_fcn_srgt,fmincon_option);

                                dist=vecnorm(self.datalib.X(self.dataoptim.Idx_best,:)-x_conv,2,2);
                                % if converage to local minimum, set to infeasible
                                if any(dist/vari_num < self.identiy_torl),self.datalib.Bool_conv(x_idx)=true(1);end
                            end
                        end

                        % step 7.3
                        % use GPC to limit do not converage to exist local best
                        if ~all(self.datalib.Bool_conv)
                            Class=-1*ones(size(X,1),1);
                            Class(self.datalib.Bool_conv)=1; % cannot go into converage area

                            self.GPC_conv=classifyGPC(X,Class,self.GPC_conv);
                            con_GPC_fcn=@(x) self.conFcnGPC(x,@(x) self.GPC_conv.predict(x));
                        else
                            con_GPC_fcn=[];
                        end

                        % step 7.4
                        % resample latin hypercubic and updata into data lib
                        try
                            X_add=self.lhdESLHS(min(floor(self.sample_num_rst),self.NFE_max-self.dataoptim.NFE-1),vari_num,...
                                low_bou,up_bou,X,con_GPC_fcn);
                        catch
                            X_add=lhsdesign(min(floor(self.sample_num_rst),self.NFE_max-self.dataoptim.NFE-1),vari_num).*(up_bou-low_bou)+low_bou;
                        end

                        if self.FLAG_DRAW_FIGURE && vari_num < 3
                            classifyVisualize(self.GPC_conv,low_bou,up_bou);
                            line(X(:,1),X(:,2),'Marker','o','color','k','LineStyle','none');
                        end
                    else
                        % step 6.1
                        % check if improve
                        improve_flag=false;

                        if self.FLAG_CON
                            Bool_feas=Vio == 0;
                            Bool_comp=(~self.datalib.Bool_conv)&Bool_feas;
                            Bool_comp(end)=false(1);

                            min_vio=min(Vio(~self.datalib.Bool_conv(1:end-1)));
                            min_obj=min(Obj(Bool_comp));

                            % if all point is infeasible,violation of point infilled is
                            % less than min violation of all point means improve.if
                            % feasible point exist,obj of point infilled is less than min
                            % obj means improve
                            if vio_infill == 0 || vio_infill < min_vio
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
                            Bool_comp=~self.datalib.Bool_conv;
                            Bool_comp(end)=false(1);

                            min_obj=min(Obj(Bool_comp));

                            % obj of point infilled is less than min obj means improve
                            if obj_infill < min_obj
                                % imporve, continue local search
                                improve_flag=true;
                            end
                        end

                        % step 6.2
                        % if obj no improve, use GPC to identify interest area
                        % than, imporve interest area surrogate quality
                        if ~improve_flag
                            fmincon_option=optimoptions('fmincon','Display','none','Algorithm','sqp','ConstraintTolerance',self.con_torl);
                            % construct GPC
                            train_num=min(size(self.datalib.X,1),11*vari_num-1+25);
                            [self.GPC_pareto,x_pareto_center,pareto_idx]=self.trainFilter(self.datalib,x_infill,train_num,self.GPC_pareto);
                            con_GPC_fcn=@(x) self.conFcnGPC(x,@(x) self.GPC_pareto.predict(x));

                            % step 6.3
                            % identify interest area
                            center_point=fmincon(con_GPC_fcn,x_pareto_center,[],[],[],[],low_bou,up_bou,[],fmincon_option);

                            r_interest=norm((center_point-x_infill)./(up_bou-low_bou),1)/vari_num;
                            r_interest=max(self.min_r_interest,r_interest);
                            r_interest=min(self.max_r_interest,r_interest);

                            % generate trial point
                            trial_point=repmat(x_infill,self.trial_num,1);
                            trial_point=trial_point+...
                                normrnd(0,r_interest,[self.trial_num,vari_num]).*(up_bou-low_bou);
                            trial_point=max(trial_point,low_bou);
                            trial_point=min(trial_point,up_bou);

                            Bool_negetive=con_GPC_fcn(trial_point) < 0.5;
                            if sum(Bool_negetive) < self.sample_num_add
                                value=con_GPC_fcn(trial_point);
                                thres=quantile(value,0.25);
                                Bool_negetive=value<thres;
                            end
                            trial_point_filter=trial_point(Bool_negetive,:);

                            % step 6.4
                            % select point
                            if size(trial_point_filter,1) <= min(self.sample_num_add,self.NFE_max-self.dataoptim.NFE-1)
                                X_add=trial_point_filter;
                            else
                                max_dist=0;
                                iter_select=1;
                                while iter_select < 100
                                    select_idx=randperm(size(trial_point_filter,1),min(self.sample_num_add,self.NFE_max-self.dataoptim.NFE-1));
                                    dist=self.calMinDistanceIter(trial_point_filter(select_idx,:),X);
                                    if max_dist < dist
                                        X_add=trial_point_filter(select_idx,:);
                                        max_dist=dist;
                                    end

                                    iter_select=iter_select+1;
                                end
                            end

                            if self.FLAG_DRAW_FIGURE && vari_num < 3
                                classifyVisualize(self.GPC_pareto,low_bou,up_bou);
                                line(trial_point(:,1),trial_point(:,2),'Marker','o','color','k','LineStyle','none');
                                line(X_add(:,1),X_add(:,2),'Marker','o','color','g','LineStyle','none');
                            end
                        end
                    end

                    % step 6, 7
                    % updata data lib
                    [self.datalib,~,~,~,~,~,repeat_idx]=self.sample(self.datalib,objcon_fcn,X_add);
                    [X,Obj,Con,Coneq,Vio]=self.datalibLoad(self.datalib,low_bou,up_bou);
                    self.datalib.Bool_conv=[self.datalib.Bool_conv;false(sum(repeat_idx == 0),1)];
                end

                obj_infill_old=obj_infill;

                % save iteration
                if ~isempty(self.dataoptim_filestr)
                    datalib=self.datalib;
                    dataoptim=self.dataoptim;
                    save(self.dataoptim_filestr,'datalib','dataoptim');
                end
            end

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
        function updataLocalPotential(self,X,vari_num,low_bou,up_bou)
            % updata potential point of optimizer
            %

            fmincon_option=optimoptions('fmincon','Display','none','Algorithm','sqp','ConstraintTolerance',self.con_torl);
            if self.dataoptim.detect_pot
                % detech potential local best point
                for x_idx=1:size(X,1)

                    x_init=X(x_idx,:);
                    [x_pot,obj_pot_pred,exit_flag,output_fmincon]=fmincon(self.obj_fcn_srgt,x_init,[],[],[],[],...
                        low_bou,up_bou,self.con_fcn_srgt,fmincon_option);

                    % check if x_pot have existed
                    add_flag=true;
                    dist=vecnorm([self.dataoptim.X_pot;self.datalib.X(self.dataoptim.Idx_best,:)]-x_pot,2,2);
                    if any(dist/vari_num < self.identiy_torl)
                        add_flag=false;
                    end

                    % updata into self.dataoptim.X_pot
                    if add_flag
                        self.dataoptim.X_pot=[self.dataoptim.X_pot;x_pot];
                        self.dataoptim.Obj_pot=[self.dataoptim.Obj_pot;obj_pot_pred];
                        if self.FLAG_CON
                            [con_potl,coneq_potl]=self.con_fcn_srgt(x_pot);
                            vio=[];
                            if ~isempty(con_potl),vio=[vio,max(max(con_potl-self.con_torl,0),[],2)];end
                            if ~isempty(coneq_potl),vio=[vio,max(abs(coneq_potl-self.con_torl),[],2)];end
                            vio=max(vio,[],2);
                            self.dataoptim.Vio_pot=[self.dataoptim.Vio_pot;vio];
                        end
                    end
                end

                self.dataoptim.detect_pot=false;
            else
                % updata X potential
                for x_idx=1:size(self.dataoptim.X_pot,1)
                    x_pot=self.dataoptim.X_pot(x_idx,:);

                    [x_pot,obj_pot_pred,exit_flag,output_fmincon]=fmincon(self.obj_fcn_srgt,x_pot,[],[],[],[],...
                        low_bou,up_bou,self.con_fcn_srgt,fmincon_option);

                    self.dataoptim.X_pot(x_idx,:)=x_pot;
                    self.dataoptim.Obj_pot(x_idx,:)=obj_pot_pred;
                    if self.FLAG_CON
                        [con_potl,coneq_potl]=self.con_fcn_srgt(x_pot);
                        vio=[];
                        if ~isempty(con_potl),vio=[vio,max(max(con_potl-self.con_torl,0),[],2)];end
                        if ~isempty(coneq_potl),vio=[vio,max(abs(coneq_potl-self.con_torl),[],2)];end
                        vio=max(vio,[],2);
                        self.dataoptim.Vio_pot(x_idx,:)=vio;
                    end
                end

                % merge X potential
                % Upward merge
                for x_idx=size(self.dataoptim.X_pot,1):-1:1
                    x_pot=self.dataoptim.X_pot(x_idx,:);

                    % check if x_pot have existed
                    merge_flag=false(1);
                    dist=vecnorm(self.dataoptim.X_pot(1:x_idx-1,:)-x_pot,2,2);
                    if any(dist/vari_num < self.identiy_torl)
                        merge_flag=true(1);
                    end

                    % updata into self.dataoptim.X_pot
                    if merge_flag
                        self.dataoptim.X_pot(x_idx,:)=[];
                        self.dataoptim.Obj_pot(x_idx,:)=[];
                        if ~isempty(self.dataoptim.Vio_pot),self.dataoptim.Vio_pot(x_idx,:)=[];end
                    end
                end
            end

            % sort X potential by Obj and Vio
            feas_idx=find(self.dataoptim.Vio_pot == 0,1,'last');
            if isempty(feas_idx) % mean do not have fesaible point
                [self.dataoptim.Obj_pot,idx]=sort(self.dataoptim.Obj_pot);
                if ~isempty(self.dataoptim.Vio_pot),self.dataoptim.Vio_pot=self.dataoptim.Vio_pot(idx,:);end
                self.dataoptim.X_pot=self.dataoptim.X_pot(idx,:);
            else
                [self.dataoptim.Obj_pot(1:feas_idx,:),idx_feas]=sort(self.dataoptim.Obj_pot(1:feas_idx,:));
                [self.dataoptim.Obj_pot(feas_idx+1:end,:),idx_infeas]=sort(self.dataoptim.Obj_pot(feas_idx+1:end,:));
                idx=[idx_feas;idx_infeas+feas_idx];

                if ~isempty(self.dataoptim.Vio_pot),self.dataoptim.Vio_pot=self.dataoptim.Vio_pot(idx,:);end
                self.dataoptim.X_pot=self.dataoptim.X_pot(idx,:);
            end

        end

        function [model_GPC,x_pareto_center,pareto_idx]=trainFilter(self,datalib,x_infill,train_num,model_GPC)
            % train filter of gaussian process classifer
            %
            if nargin < 1
                model_GPC=[];
            end

            [X,Obj,~,~,Vio]=self.datalibLoad(datalib);
            x_dist=vecnorm(X-x_infill,2,2);
            [~,idx]=sort(x_dist);
            X=X(idx(1:train_num),:);
            Obj=Obj(idx(1:train_num),:);

            if self.FLAG_CON
                % base on filter to decide which x should be choose
                Vio=Vio(idx(1:train_num),:);
                pareto_idx=self.getParetoFront(Obj,Vio);

                Class=ones(size(X,1),1);
                Class(pareto_idx)=0;

                x_pareto_center=sum(X(pareto_idx,:),1)/length(pareto_idx);

                model_GPC=classifyGPC(X,Class,model_GPC);
            else
                obj_threshold=prctile(Obj,50-40*self.dataoptim.NFE/self.NFE_max);

                Class=ones(size(X,1),1);
                Class(Obj < obj_threshold)=0;

                pareto_idx=find(X(Obj < obj_threshold,:));
                x_pareto_center=sum(X(pareto_idx,:),1)/sum(Obj < obj_threshold);

                model_GPC=classifyGPC(X,Class,model_GPC);
            end
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

        function [con,coneq]=conFcnGPC(x,pred_fcn_GPC)
            % function to obtain probability predict function
            %
            [~,~,con]=pred_fcn_GPC(x);
            coneq=[];
        end

        function pareto_idx_list=getParetoFront(obj_list,vio_list)
            % distinguish pareto front of data list
            % dominate define as followed
            % Solution i is feasible and solution j is not.
            % Solutions i and j are both infeasible,...
            % but solution i has a smaller overall constraint violation.
            % Solutions i and j are feasible and solution i dominates solution j
            %
            x_number=size(obj_list,1);
            pareto_idx_list=[]; % sort all idx of filter point list

            % select no domain filter
            for x_idx=1:x_number
                obj=obj_list(x_idx,:);
                ks=vio_list(x_idx,:);

                pareto_idx=1;
                add_filter_flag=true(1);
                while pareto_idx <= length(pareto_idx_list)
                    % compare x with exit pareto front point
                    x_pareto_idx=pareto_idx_list(pareto_idx,:);

                    % contain constraint of x_filter
                    obj_pareto=obj_list(x_pareto_idx,:);
                    ks_pareto=vio_list(x_pareto_idx,:);

                    % compare x with x_pareto
                    if ks_pareto <= 0
                        if obj > obj_pareto || ks > 0
                            add_filter_flag=false(1);
                            break;
                        end
                    else
                        if obj > obj_pareto && ks > ks_pareto
                            add_filter_flag=false(1);
                            break;
                        end
                    end

                    % if better than exit pareto point,reject pareto point
                    delete_filter_flag=false(1);
                    if ks <= 0
                        if obj_pareto > obj || ks_pareto > 0
                            delete_filter_flag=true(1);
                        end
                    else
                        if obj_pareto > obj && ks_pareto > ks
                            delete_filter_flag=true(1);
                        end
                    end
                    if delete_filter_flag
                        pareto_idx_list(pareto_idx)=[];
                        pareto_idx=pareto_idx-1;
                    end

                    pareto_idx=pareto_idx+1;
                end

                % add into pareto list if possible
                if add_filter_flag
                    pareto_idx_list=[pareto_idx_list;x_idx];
                end
            end

        end

        function dist_min=calMinDistanceIter(X,X_exist)
            % get distance min from x_list
            % this version do not consider distance between x exist
            %

            % sort x_supply_list_initial to decrese distance calculate times
            X=sortrows(X,1);
            [sample_num,vari_num]=size(X);
            dist_min=vari_num;
            for x_idx=1:sample_num
                x_curr=X(x_idx,:);
                x_next_idx=x_idx + 1;
                % only search in min_distance(x_list had been sort)
                search_range=vari_num;
                while x_next_idx <= sample_num &&...
                        (X(x_next_idx,1)-X(x_idx,1))^2 ...
                        < search_range
                    x_next=X(x_next_idx,:);
                    distance_temp=sum((x_next-x_curr).^2);
                    if distance_temp < dist_min
                        dist_min=distance_temp;
                    end
                    if distance_temp < search_range
                        search_range=distance_temp;
                    end
                    x_next_idx=x_next_idx+1;
                end
                for x_exist_idx=1:size(X_exist,1)
                    x_next=X_exist(x_exist_idx,:);
                    distance_temp=sum((x_next-x_curr).^2);
                    if distance_temp < dist_min
                        dist_min=distance_temp;
                    end
                end
            end
            dist_min=sqrt(dist_min);
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

%% surrogate function


%% machine learning


%% latin hypercube design
