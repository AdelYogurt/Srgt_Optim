classdef OptimTRARSM < handle
    % TR-ARSM optimization algorithm
    % Trust-Region-Based Adaptive Response Surface Method optimization algorithm
    %
    % reference: [1] LONG T, LI X, SHI R, et al., Gradient-Free
    % Trust-Region-Based Adaptive Response Surface Method for Expensive
    % Aircraft Optimization[J]. AIAA Journal, 2018, 56(2): 862-73.
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

        SVM_pareto;
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
        enlarge_range=2; % adapt region enlarge parameter
        range_max=0.5;
        range_min=0.01;
        scale_min=0.5;
        r1=0.1;
        r2=0.75;
        c1=0.75;
        c2=1.25;

        % augmented lagrange parameter
        lambda_initial=10;
        miu_max=1000;
        gama=2;
    end

    % main function
    methods
        function self=OptimTRARSM(NFE_max,iter_max,obj_torl,con_torl)
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
            if length(varargin) == 1
                % input is struct or object
                problem=varargin{1};
                if isstruct(problem)
                    prob_field=fieldnames(problem);
                    if ~contains(prob_field,'objcon_fcn'), error('OptimTRARSM.optimize: input problem lack objcon_fcn'); end
                    objcon_fcn=problem.objcon_fcn;
                    if ~contains(prob_field,'vari_num'), error('OptimTRARSM.optimize: input problem lack vari_num'); end
                    if ~contains(prob_field,'low_bou'), error('OptimTRARSM.optimize: input problem lack low_bou'); end
                    if ~contains(prob_field,'up_bou'), error('OptimTRARSM.optimize: input problem lack up_bou'); end
                    if ~contains(prob_field,'con_fcn_cheap'), problem.con_fcn_cheap=[]; end
                    con_fcn_cheap=problem.con_fcn_cheap;
                    clear('prob_field');
                else
                    prob_method=methods(problem);
                    if ~contains(prob_method,'objcon_fcn'), error('OptimTRARSM.optimize: input problem lack objcon_fcn'); end
                    objcon_fcn=@(x) problem.objcon_fcn(x);
                    prob_pro=properties(problem);
                    if ~contains(prob_pro,'vari_num'), error('OptimTRARSM.optimize: input problem lack vari_num'); end
                    if ~contains(prob_pro,'low_bou'), error('OptimTRARSM.optimize: input problem lack low_bou'); end
                    if ~contains(prob_pro,'up_bou'), error('OptimTRARSM.optimize: input problem lack up_bou'); end
                    if ~contains(prob_method,'con_fcn_cheap')
                        con_fcn_cheap=[];
                    else
                        con_fcn_cheap=@(x) problem.con_fcn_cheap(x);
                    end
                    clear('prob_method','prob_pro');
                end
                vari_num=problem.vari_num;
                low_bou=problem.low_bou;
                up_bou=problem.up_bou;
            else
                % multi input
                varargin=[varargin,repmat({[]},1,5-length(varargin))];
                [objcon_fcn,vari_num,low_bou,up_bou,con_fcn_cheap]=varargin{:};
            end

            % Latin hypercube sample count
            sample_num=(vari_num+1)*(vari_num+2)/2;

            % NFE and iteration setting
            if isempty(self.NFE_max),self.NFE_max=10+10*vari_num;end
            if isempty(self.iter_max),self.iter_max=20+20*vari_num;end

            if isempty(self.dataoptim)
                self.dataoptim.NFE=0;
                self.dataoptim.Add_idx=[];
                self.dataoptim.iter=0;
                self.dataoptim.done=false;
            end

            % step 1-2, generate initial data library
            self.sampleInit(objcon_fcn,vari_num,low_bou,up_bou,con_fcn_cheap);

            % trust region updata
            low_bou_TR=low_bou;
            up_bou_TR=up_bou;

            % initialize augmented Lagrange method parameter
            lambda_con=self.lambda_initial*ones(1,size(self.datalib.Con,2));
            lambda_coneq=self.lambda_initial*ones(1,size(self.datalib.Coneq,2));
            miu=1;

            % initialize all data to begin optimize
            [X,Obj,Con,Coneq,Vio]=self.datalibLoad(self.datalib,low_bou,up_bou);
            obj_num=size(Obj,2);con_num=size(Con,2);coneq_num=size(Coneq,2);vio_num=size(Vio,2);
            result_X=zeros(self.iter_max,vari_num);
            result_Obj=zeros(self.iter_max,obj_num);
            if con_num,result_Con=zeros(self.iter_max,con_num);
            else,result_Con=[];end
            if coneq_num,result_Coneq=zeros(self.iter_max,coneq_num);
            else,result_Coneq=[];end
            if vio_num,result_Vio=zeros(self.iter_max,vio_num);
            else,result_Vio=[];end
            con_best=[];coneq_best=[];vio_best=[];
            
            self.dataoptim.iter=self.dataoptim.iter+1;
            while ~self.dataoptim.done
                % Step 3
                % load data of trust region
                [X,Obj,Con,Coneq,~]=self.datalibLoad(self.datalib,low_bou_TR,up_bou_TR);

                % if point too less,add more point
                if size(X,1) < (vari_num+1)*(vari_num+2)/2
                    % generate latin hypercube sequence
                    X_add=lhdESLHS(min(self.NFE_max-self.dataoptim.NFE-1,sample_num-size(X,1)),vari_num,...
                        low_bou_TR,up_bou_TR,X,con_fcn_cheap);

                    % update x_updata_list into data library
                    [self.datalib,X_add,Obj_add,Con_add,Coneq_add,~]=self.sample...
                        (self.datalib,objcon_fcn,X_add);

                    % normalization data and updata into list
                    X=[X;X_add];
                    Obj=[Obj;Obj_add];
                    if ~isempty(Con),Con=[Con;Con_add];end
                    if ~isempty(Coneq),Coneq=[Coneq;Coneq_add];end
                end

                % generate surrogate model
                [self.obj_fcn_srgt,self.con_fcn_srgt,...
                    self.Srgt_obj,self.Srgt_con,self.Srgt_coneq]=self.getSrgtFcnRSM...
                    (X,Obj,Con,Coneq);

                % generate merit function
                if self.FLAG_CON
                    penalty_fcn=@(x) self.penaltyFcn...
                        (x,self.obj_fcn_srgt,self.con_fcn_srgt,...
                        lambda_con,lambda_coneq,miu);
                end

                % Step 4
                % get x_infill
%                 B=self.Srgt_obj{1}.beta(2:1+vari_num);
%                 temp=self.Srgt_obj{1}.beta(2+vari_num:end);
%                 C=diag(temp(1:vari_num)); temp=temp(vari_num+1:end);
%                 for vari_idx=1:vari_num-1
%                     C(vari_idx,1+vari_idx:end)=temp(1:vari_idx)'/2;
%                     C(1+vari_idx:end,vari_idx)=temp(1:vari_idx)/2;
%                     temp=temp(vari_idx:end);
%                 end
%                 x_init=(-C\B)';

                x_init=rand(1,vari_num).*(up_bou_TR-low_bou_TR)+low_bou_TR;
                problem.solver='fmincon';
                if self.FLAG_CON
                    problem.objective=penalty_fcn;
                else
                    problem.objective=self.obj_fcn_srgt;
                end
                problem.x0=x_init;
                problem.lb=low_bou_TR;
                problem.ub=up_bou_TR;
                problem.nonlcon=con_fcn_cheap;
                problem.options=optimoptions('fmincon','Display','none','Algorithm','sqp','ConstraintTolerance',self.con_torl);

                gs=GlobalSearch('Display','off','NumTrialPoints',10*vari_num+200);
                [x_infill,obj_infill_pred]=run(gs,problem);

%                 fmincon_option=optimoptions('fmincon','Display','none','Algorithm','sqp','ConstraintTolerance',self.con_torl);
                if self.FLAG_CON
%                     [x_infill,~,~,~]=fmincon...
%                         (penalty_fcn,x_init,[],[],[],[],low_bou_TR,up_bou_TR,con_fcn_cheap,fmincon_option);
                    obj_infill_pred=self.obj_fcn_srgt(x_infill);
                else
%                     [x_infill,obj_infill_pred,~,~]=fmincon...
%                         (self.obj_fcn_srgt,x_init,[],[],[],[],low_bou_TR,up_bou_TR,con_fcn_cheap,fmincon_option);
                end

                % updata infill point
                [self.datalib,x_infill,obj_infill,con_infill,coneq_infill,vio_infill]=self.sample(self.datalib,objcon_fcn,x_infill);

                % step 6
                % find best result to record
                best_idx=self.datalib.Best_idx(end);
                x_best=self.datalib.X(best_idx,:);
                result_X(self.dataoptim.iter,:)=x_best;
                obj_best=self.datalib.Obj(best_idx,:);
                result_Obj(self.dataoptim.iter,:)=obj_best;
                if con_num,result_Con(self.dataoptim.iter,:)=self.datalib.Con(best_idx,:);end
                if coneq_num,result_Coneq(self.dataoptim.iter,:)=self.datalib.Coneq(best_idx,:);end
                if vio_num,vio_best=self.datalib.Vio(best_idx,:);result_Vio(self.dataoptim.iter,:)=vio_best;end
                self.dataoptim.iter=self.dataoptim.iter+1;

                % information
                if self.FLAG_DRAW_FIGURE && vari_num < 3
                    surrogateVisualize(self.Srgt_obj{1},low_bou_TR,up_bou_TR);
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

                % updata penalty factor and lagrangian
                if self.FLAG_CON
                    lambda_con=lambda_con+2*miu*max(con_infill,-lambda_con/2/miu);
                    lambda_coneq=lambda_coneq+2*miu*coneq_infill;
                    if miu < self.miu_max
                        miu=self.gama*miu;
                    else
                        miu=self.miu_max;
                    end
                end

                % trust region sampling
                if ~self.dataoptim.done
                    % Step 6
                    % TR guideline first iteration
                    if self.dataoptim.iter == 2
                        TR_range_nomlz=0.5;
                    else
                        best_idx=self.datalib.Best_idx(end-1); % last time best
                        x_center=self.datalib.X(best_idx,:);
                        r=(obj_infill_old-obj_infill)/(obj_infill_old-obj_infill_pred);
                        x_dis=norm((x_center-x_infill)./(up_bou-low_bou),1)/vari_num;

                        % % origin TRSS strategy
                        % if r < 0
                        %     TR_range_nomlz=self.range_min;
                        % elseif r < self.r1
                        %     TR_range_nomlz=self.c1*x_dis;
                        % elseif r > self.r2
                        %     TR_range_nomlz=min(self.c2*x_dis,self.range_max);
                        % else
                        %     TR_range_nomlz=x_dis;
                        % end
                        % TR_range_nomlz=max(TR_range_nomlz,self.range_min);

                        % function fit TRSS strategy
                        scale=(1.25*x_dis/(1+exp(-5*r-0.5))+self.range_min)/TR_range_nomlz;
                        scale=max(scale,self.scale_min);
                        TR_range_nomlz=scale*TR_range_nomlz;
                    end
                    TR_range=TR_range_nomlz.*(up_bou-low_bou);

                    % updata trust range
                    low_bou_TR=x_best-TR_range;
                    low_bou_TR=max(low_bou_TR,low_bou);
                    up_bou_TR=x_best+TR_range;
                    up_bou_TR=min(up_bou_TR,up_bou);

                    % Step 7
                    % check whether exist data
                    X_exit=self.datalibLoad(self.datalib,low_bou_TR,up_bou_TR);

                    % generate latin hypercube sequence
                    X_add=lhdESLHS(min(self.NFE_max-self.dataoptim.NFE-1,sample_num-size(X_exit,1)),vari_num,...
                        low_bou_TR,up_bou_TR,X_exit,con_fcn_cheap);

                    % updata data lib
                    [self.datalib]=self.sample(self.datalib,objcon_fcn,X_add);
                end

                % save iteration
                if ~isempty(self.dataoptim_filestr)
                    datalib=self.datalib;
                    dataoptim=self.dataoptim;
                    save(self.dataoptim_filestr,'datalib','dataoptim');
                end

                x_infill_old=x_infill;
                obj_infill_old=obj_infill;
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

        function sampleInit(self,objcon_fcn,vari_num,low_bou,up_bou,con_fcn_cheap)
            % initial latin hypercube sample
            %
            sample_num=(vari_num+1)*(vari_num+2)/2;

            % obtain datalib
            if isempty(self.datalib)
                self.datalib=self.datalibGet(vari_num,low_bou,up_bou,self.con_torl,self.datalib_filestr);
            end

            if size(self.datalib.X,1) < sample_num
                if isempty(self.X_init)
                    % use latin hypercube method to get enough initial sample x_list
                    sample_num=min(sample_num-size(self.datalib.X,1),self.NFE_max-self.dataoptim.NFE);
                    self.X_init=lhdESLHS(sample_num,vari_num,low_bou,up_bou,self.datalib.X,con_fcn_cheap);
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
        function [obj_fcn_srgt,con_fcn_srgt,Srgt_obj,Srgt_con,Srgt_coneq]=getSrgtFcnRSM...
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
                Srgt_obj{obj_idx}=srgtRSM(x_list,obj_list(:,obj_idx));
            end

            % generate con surrogate
            if ~isempty(con_list)
                Srgt_con=cell(size(con_list,2),1);
                for con_idx=1:size(con_list,2)
                    Srgt_con{con_idx}=srgtRSM(x_list,con_list(:,con_idx));
                end
            else
                Srgt_con=[];
            end

            % generate coneq surrogate
            if ~isempty(coneq_list)
                Srgt_coneq=cell(size(coneq_list,2),1);
                for coneq_idx=1:size(coneq_list,2)
                    Srgt_coneq{coneq_idx}=srgtRSM(x_list,coneq_list(:,coneq_idx));
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

        function fval=penaltyFcn...
                (x,obj_fcn_srgt,con_fcn_srgt,lambda_con,lambda_coneq,miu)
            % penalty function
            % augmented lagrange multiplier method was used
            %
            fval=obj_fcn_srgt(x);
            if ~isempty(con_fcn_srgt)
                [con,coneq]=con_fcn_srgt(x);
                if ~isempty(con)
                    psi=max(con,-lambda_con/2/miu);
                    fval=fval+sum(lambda_con.*psi+miu*psi.*psi);
                end
                if ~isempty(coneq)
                    fval=fval+sum(lambda_coneq.*coneq+miu*coneq.*coneq);
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

%% surrogate function


%% latin hypercube design

