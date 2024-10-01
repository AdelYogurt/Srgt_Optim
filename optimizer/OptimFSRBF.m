classdef OptimFSRBF < handle
    % FSRBF and SRBF-SVM optimization algorithm
    %
    % referance:
    % [1] SHI R,LIU L,LONG T,et al. Sequential Radial Basis Function Using
    % Support Vector Machine for Expensive Design Optimization [J]. AIAA
    % Journal,2017,55(1): 214-27. [2] Shi R, Liu L, Long T, et al.
    % Filter-Based Sequential Radial Basis Function Method for Spacecraft
    % Multidisciplinary Design Optimization[J]. AIAA Journal, 2018, 57(3):
    % 1019-31.
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

        add_tol=1000*eps; % surrogate add point protect range
        X_init=[];

        % hyper parameter
        sample_num_init=[];
        sample_num_add=[];
        sample_num_data=[];
        eta=[];

        penalty_SVM=100; % SVM parameter
        CF_m=2; % clustering parameter
        FROI_min=1e-3; % min boundary of interest sample area
    end

    % main function
    methods
        function self=OptimFSRBF(NFE_max,iter_max,obj_tol,con_tol)
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
                    if ~contains(prob_field,'objcon_fcn'), error('OptimFSRBF.optimize: input problem lack objcon_fcn'); end
                    objcon_fcn=problem.objcon_fcn;
                    if ~contains(prob_field,'vari_num'), error('OptimFSRBF.optimize: input problem lack vari_num'); end
                    if ~contains(prob_field,'low_bou'), error('OptimFSRBF.optimize: input problem lack low_bou'); end
                    if ~contains(prob_field,'up_bou'), error('OptimFSRBF.optimize: input problem lack up_bou'); end
                    clear('prob_field');
                else
                    prob_method=methods(problem);
                    if ~contains(prob_method,'objcon_fcn'), error('OptimFSRBF.optimize: input problem lack objcon_fcn'); end
                    objcon_fcn=@(x) problem.objcon_fcn(x);
                    prob_prop=properties(problem);
                    if ~contains(prob_prop,'vari_num'), error('OptimFSRBF.optimize: input problem lack vari_num'); end
                    if ~contains(prob_prop,'low_bou'), error('OptimFSRBF.optimize: input problem lack low_bou'); end
                    if ~contains(prob_prop,'up_bou'), error('OptimFSRBF.optimize: input problem lack up_bou'); end
                    clear('prob_method','prob_prop');
                end
                vari_num=problem.vari_num;
                low_bou=problem.low_bou;
                up_bou=problem.up_bou;
            else
                % multi input
                varargin=[varargin,repmat({[]},1,4-length(varargin))];
                [objcon_fcn,vari_num,low_bou,up_bou]=varargin{:};
            end

            if isempty(self.sample_num_init),self.sample_num_init=min((vari_num+1)*(vari_num+2)/2,5*vari_num);end
            if isempty(self.sample_num_add),self.sample_num_add=vari_num;end
            if isempty(self.sample_num_data),self.sample_num_data=100*self.sample_num_init;end
            if isempty(self.eta),self.eta=1/vari_num;end % space decrease coefficient

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
            X_data=lhsdesign(self.sample_num_data,vari_num).*(up_bou-low_bou)+low_bou;

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
                % step 3, construct surrogate
                [self.obj_fcn_srgt,self.con_fcn_srgt,...
                    self.Srgt_obj,self.Srgt_con,self.Srgt_coneq]=getSrgtFcnVar...
                    (X,Obj,Con,Coneq);

                ga_option=optimoptions('ga','Display','none','ConstraintTolerance',0,'MaxGenerations',10,'HybridFcn','fmincon');

                % MSP guideline to obtain x_adapt
                [x_infill,~,exit_flag,output_ga]=ga...
                    (self.obj_fcn_srgt,vari_num,[],[],[],[],low_bou,up_bou,self.con_fcn_srgt,ga_option);

                if exit_flag == -2
                    % step 4
                    % optimal feasiblilty if do not exist feasible point
                    vio_fcn_surr=@(x) vioFcnSurr(x,self.con_fcn_srgt,self.con_tol);
                    [x_infill,~,exit_flag,output_ga]=ga...
                        (vio_fcn_surr,vari_num,[],[],[],[],low_bou,up_bou,[],ga_option);
                end

                x_infill=max(x_infill,low_bou);
                x_infill=min(x_infill,up_bou);

                % add infill point
                [self.datalib,x_infill,obj_infill,~,~,vio_infill]=self.sample(self.datalib,objcon_fcn,x_infill);
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

                % Interest space sampling
                if ~self.dataoptim.done
                    % step 7-1.2
                    % using SVM to identify area which is interesting
                    [self.SVM_pareto,x_pareto_center,pareto_idx]=self.trainFilter();

                    % get data to obtain clustering center
                    x_sup_list=X_data(self.SVM_pareto.predict(X_data) == 1,:);

                    % step 7-3
                    if isempty(x_sup_list)
                        % no sup point found use x_pareto_center
                        x_center=x_pareto_center;
                    else
                        model_FCM=clusterFCM(x_sup_list,1,self.CF_m);
                        x_center=model_FCM.center_list;
                    end

                    % updata FROI
                    x_infill_nomlz=(x_infill-low_bou)./(up_bou-low_bou);
                    x_center_nomlz=(x_center-low_bou)./(up_bou-low_bou);
                    FROI_range_nomlz=self.eta*norm(x_infill_nomlz-x_center_nomlz,2);
                    if FROI_range_nomlz < self.FROI_min
                        FROI_range_nomlz=self.FROI_min;
                    end
                    FROI_range=FROI_range_nomlz.*(up_bou-low_bou);
                    low_bou_FROI=x_infill-FROI_range;
                    low_bou_FROI=max(low_bou_FROI,low_bou);
                    up_bou_FROI=x_infill+FROI_range;
                    up_bou_FROI=min(up_bou_FROI,up_bou);

                    if self.FLAG_DRAW_FIGURE && vari_num < 3
                        displayClassify(self.SVM_pareto,low_bou,up_bou);
                        bou_line=[low_bou_FROI;[low_bou_FROI(1),up_bou_FROI(2)];up_bou_FROI;[up_bou_FROI(1),low_bou_FROI(2)];low_bou_FROI];
                        line(bou_line(:,1),bou_line(:,2));
                        line(x_infill(1),x_infill(2),'Marker','x')
                    end

                    % sampling in ISR
                    X_add=lhsdesign(min(self.sample_num_add,self.NFE_max-self.dataoptim.NFE-1),vari_num).*(up_bou_FROI-low_bou_FROI)+low_bou_FROI;

                    % add point
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
        function [model_SVM,x_pareto_center,pareto_idx_list]=trainFilter(self)
            % train filter of gaussian process classifer
            %
            [X,Obj,~,~,Vio]=self.datalibLoad(self.datalib);

            if self.FLAG_CON
                % base on filter to decide which x should be choose
                pareto_idx_list=getParetoFront(Obj,Vio);

                Class=ones(size(X,1),1);
                Class(pareto_idx_list)=0;

                x_pareto_center=sum(X(pareto_idx_list,:),1)/length(pareto_idx_list);

                model_SVM=classifySVM(X,Class,struct('box_con',self.penalty_SVM));
            else
                obj_threshold=prctile(Obj,50);

                Class=ones(size(X,1),1);
                Class(Obj < obj_threshold)=0;

                pareto_idx_list=find(X(Obj < obj_threshold,:));
                x_pareto_center=sum(pareto_idx_list,1)/sum(Obj < obj_threshold);

                model_SVM=classifySVM(X,Class,struct('box_con',self.penalty_SVM));
            end
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

%% common function

function [obj_fcn_srgt,con_fcn_srgt,Srgt_obj,Srgt_con,Srgt_coneq]=getSrgtFcnVar...
    (x_list,obj_list,con_list,coneq_list)
% generate surrogate function of objective and constraints
%
% output:
% obj_fcn_srgt(output is obj_pred, obj_var),...
% con_fcn_srgt(output is con_pred, coneq_pred, con_var, coneq_var)
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
end

function vio=vioFcnSurr(x,con_fcn_srgt,con_tol)
% calculate violation by con_fcn_srgt(x)
%
[con,coneq]=con_fcn_srgt(x);vio=[];
% calculate vio
if ~isempty(con),vio=[vio,max(max(con-con_tol,0),[],2)];end
if ~isempty(coneq),vio=[vio,max(max(abs(coneq)-con_tol,0),[],2)];end
vio=max(vio,[],2);
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
    vio=vio_list(x_idx,:);

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
            if obj > obj_pareto || vio > 0
                add_filter_flag=false(1);
                break;
            end
        else
            if obj > obj_pareto && vio > ks_pareto
                add_filter_flag=false(1);
                break;
            end
        end

        % if better than exit pareto point,reject pareto point
        delete_filter_flag=false(1);
        if vio <= 0
            if obj_pareto > obj || ks_pareto > 0
                delete_filter_flag=true(1);
            end
        else
            if obj_pareto > obj && ks_pareto > vio
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
