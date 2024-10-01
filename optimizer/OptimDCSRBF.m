classdef OptimDCSRBF < handle
    % DYCORS-LMSRBF optimization algorithm
    % Combining dynamic coordinate search with local metric stochastic RBF optimization algorithm
    %
    % reference:
    % [1] Regis R G, Shoemaker C A. Combining radial basis function
    % surrogates and dynamic coordinate search in high-dimensional
    % expensive black-box optimization[J]. Engineering Optimization, 2013,
    % 45: 529-55.
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
        trial_num=[];
        sel_prblty_init=[];

        coord_init=[];
        coord_min=[];
        coord_max=[];

        tau_success=[];
        tau_fail=[];
        kappa=[];
        w_list=[];
    end

    % main function
    methods
        function self=OptimDCSRBF(NFE_max,iter_max,obj_tol,con_tol)
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
                    if ~contains(prob_field,'objcon_fcn'), error('OptimDCSRBF.optimize: input problem lack objcon_fcn'); end
                    objcon_fcn=problem.objcon_fcn;
                    if ~contains(prob_field,'vari_num'), error('OptimDCSRBF.optimize: input problem lack vari_num'); end
                    if ~contains(prob_field,'low_bou'), error('OptimDCSRBF.optimize: input problem lack low_bou'); end
                    if ~contains(prob_field,'up_bou'), error('OptimDCSRBF.optimize: input problem lack up_bou'); end
                    clear('prob_field');
                else
                    prob_method=methods(problem);
                    if ~contains(prob_method,'objcon_fcn'), error('OptimDCSRBF.optimize: input problem lack objcon_fcn'); end
                    objcon_fcn=@(x) problem.objcon_fcn(x);
                    prob_prop=properties(problem);
                    if ~contains(prob_prop,'vari_num'), error('OptimDCSRBF.optimize: input problem lack vari_num'); end
                    if ~contains(prob_prop,'low_bou'), error('OptimDCSRBF.optimize: input problem lack low_bou'); end
                    if ~contains(prob_prop,'up_bou'), error('OptimDCSRBF.optimize: input problem lack up_bou'); end
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

            if isempty(self.sample_num_init),self.sample_num_init=ceil(vari_num/10)*20;end
            if isempty(self.trial_num),self.trial_num=min(100*vari_num,100);end
            if isempty(self.sel_prblty_init),self.sel_prblty_init=min(20/vari_num,1);end
            if isempty(self.coord_init),self.coord_init=0.2*(up_bou-low_bou);end
            if isempty(self.coord_min),self.coord_min=0.2*1/64*(up_bou-low_bou);end
            if isempty(self.coord_max),self.coord_max=2*(up_bou-low_bou);end
            if isempty(self.tau_success),self.tau_success=3;end
            if isempty(self.tau_fail),self.tau_fail=max(vari_num,5);end
            if isempty(self.kappa),self.kappa=4;end
            if isempty(self.w_list),self.w_list=[0.3,0.5,0.8,0.95];end

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
            self.dataoptim.coord=self.coord_init;
            self.dataoptim.C_success=0;
            self.dataoptim.C_fail=0;
            while ~self.dataoptim.done
                % step 4, construct surrogate
                [self.obj_fcn_srgt,self.con_fcn_srgt,self.Srgt_obj,self.Srgt_con,self.Srgt_coneq]=getSrgtFcnVar...
                    (X,Obj,Con,Coneq);

                % step 5, perturbed to get trial point

                % calculate gradient
                best_idx=self.datalib.Best_idx(end);
                x_best=X(best_idx,:);
                [~,grad]=differ(self.obj_fcn_srgt,x_best);
                grad=abs(grad);
                grad_max=max(grad);grad_min=min(grad);
                grad=(grad-grad_min)./(grad_max-grad_min);

                sel_prob=self.sel_prblty_init*(1-log(self.dataoptim.NFE-self.sample_num_init+1)/log(self.NFE_max-self.sample_num_init));

                % select coords to perturb
                sel_list=[];
                for vari_idx=1:vari_num
                    if rand() < sel_prob*grad(vari_idx)
                        sel_list=[sel_list,vari_idx];
                    end
                end
                if isempty(sel_list),sel_list=randi(vari_num);end

                % generate trial point
                X_trial=repmat(x_best,self.trial_num,1);
                for sel_idx=1:length(sel_list)
                    sel=sel_list(sel_idx);
                    X_trial(:,sel)=X_trial(:,sel)+...
                        normrnd(0,self.dataoptim.coord(sel),[self.trial_num,1]);
                end
                X_trial=max(X_trial,low_bou);
                X_trial=min(X_trial,up_bou);

                % step 6
                % select point to add
                w_idx=mod(self.dataoptim.NFE-self.sample_num_init+1,self.kappa);
                if w_idx == 0
                    w_R=self.w_list(self.kappa);
                else
                    w_R=self.w_list(w_idx);
                end
                w_D=1-w_R;

                % evaluate trial point merit
                merit_list=meritFcn(self.obj_fcn_srgt,X_trial,X,w_R,w_D);
                [~,idx]=min(merit_list);
                x_infill=X_trial(idx,:);

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

                % step 7
                % adjust step size
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

                % information
                if self.FLAG_DRAW_FIGURE && vari_num < 3
                    cla;
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

            function [fval,grad]=differ(differ_fcn,x)
                % differ function to get gradient
                % support matrix output
                %
                num=length(x);
                ds=1e-5;
                fval=differ_fcn(x);

                % gradient
                grad=zeros(num,1);

                for vi=1:num
                    x_fwd=x;
                    x_fwd(vi)=x_fwd(vi)+ds;
                    grad(vi)=(differ_fcn(x_fwd)-fval)/ds;
                end
            end

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

%% common function

function [obj_fcn_srgt,con_fcn_srgt,Srgt_obj,Srgt_con,Srgt_coneq]=getSrgtFcnVar...
    (x_list,obj_list,con_list,coneq_list,Srgt_obj,Srgt_con,Srgt_coneq)
% generate surrogate function of objective and constraints
%
% output:
% obj_fcn_srgt(output is obj_pred, obj_var),...
% con_fcn_srgt(output is con_pred, coneq_pred, con_var, coneq_var)
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
Srgt_obj=cell(size(obj_list,2),1);
for obj_idx=1:size(obj_list,2)
    Srgt_obj{obj_idx}=srgtsfRBF(x_list,obj_list(:,obj_idx),Srgt_obj{obj_idx});
end

% generate con surrogate
if ~isempty(con_list)
    Srgt_con=cell(size(con_list,2),1);
    for con_idx=1:size(con_list,2)
        Srgt_con{con_idx}=srgtsfRBF(x_list,con_list(:,con_idx),Srgt_con{con_idx});
    end
else
    Srgt_con=[];
end

% generate coneq surrogate
if ~isempty(coneq_list)
    Srgt_coneq=cell(size(coneq_list,2),1);
    for coneq_idx=1:size(coneq_list,2)
        Srgt_coneq{coneq_idx}=srgtsfRBF(x_list,coneq_list(:,coneq_idx),Srgt_coneq{coneq_idx});
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

