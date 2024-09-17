classdef OptimSACOAMS < handle
    % SCAO-AMS optimization algorithm
    % Model_function is cell of multi fidelity function handle
    % Cost is array of each fidelity model function
    % from start to end is high to low fidelity
    %
    % Copyright 2023 4 Adel
    %
    % abbreviation:
    % obj: objective, con: constraint, iter: iteration, tor: torlance
    % fcn: function
    % lib: library, init: initial, rst: restart, potl: potential
    %
    properties
        % basic parameter
        NFE_max;
        iter_max;
        obj_torl;
        con_torl;
        datalib_hf;
        datalib_lf;

        FLAG_DRAW_FIGURE=0; % whether draw data
        FLAG_INFORMATION=1; % whether print data
        FLAG_CONV_JUDGE=0; % whether judgment convergence
        WRIRE_FILE_FLAG=0; % whether write to file
        dataoptim_filestr=''; % optimize save mat name

        nomlz_value=100; % max obj when normalize obj,con,coneq
        protect_range=1e-16; % surrogate add point protect range
        identiy_torl=1e-2; % if inf norm of point less than self.identiy_torl, point will be consider as same local best

        str_data_file_hf='result_total_hf.txt'
        str_data_file_lf='result_total_lf.txt'
    end

    properties
        % problem parameter
        FLAG_CON;

        X_local_best=[];
        Obj_local_best=[];

        X_potl=[];
        Obj_potl=[];
        Vio_potl=[];

        detect_local_flag=true(1);
        add_lf_flag=true(1);

        model_GPC=struct('hyp',struct('mean',0,'cov',[0,0]));
        model_MFGPC=struct('hyp',struct('mean',0,'cov',[0,0,0,0,0]));

        obj_fcn_srgt;
        con_fcn_srgt;
        ks_fcn_surr;

        obj_surr;
        con_surr_list;
        coneq_surr_list;
        ks_surr;
    end

    methods % main
        function self=OptimSACOAMS(NFE_max,iter_max,obj_torl,con_torl,datalib_hf,datalib_lf,dataoptim_filestr)
            % initialize optimization
            %
            if nargin < 7
                dataoptim_filestr='';
                if nargin < 6
                    datalib_lf=[];
                    if nargin < 5
                        datalib_hf=[];
                        if nargin < 4 || isempty(con_torl)
                            con_torl=1e-3;
                            if nargin < 3 || isempty(obj_torl)
                                obj_torl=1e-6;
                                if nargin < 2
                                    iter_max=[];
                                    if nargin < 1
                                        NFE_max=[];
                                    end
                                end
                            end
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
            self.datalib_hf=datalib_hf;
            self.datalib_lf=datalib_lf;
            self.dataoptim_filestr=dataoptim_filestr;
        end

        function [x_best,obj_best,NFE,output]=optimize(self,varargin)
            % main optimize function
            %
            if length(varargin) == 1
                problem=varargin{1};
                objcon_fcn_list=problem.objcon_fcn_list;
                Cost=problem.Cost;
                Ratio=problem.Ratio;
                vari_num=problem.vari_num;
                low_bou=problem.low_bou;
                up_bou=problem.up_bou;
                con_fcn_cheap=problem.con_fcn_cheap;
            else
                varargin=[varargin,repmat({[]},1,7-length(varargin))];
                [objcon_fcn_list,Cost,Ratio,vari_num,low_bou,up_bou,con_fcn_cheap]=varargin{:};
            end

            % NFE and iteration setting
            if isempty(self.NFE_max)
                self.NFE_max=10+10*vari_num;
            end

            if isempty(self.iter_max)
                self.iter_max=20+20*vari_num;
            end

            fidelity_num=length(objcon_fcn_list);
            objcon_fcn_hf=objcon_fcn_list{1};
            objcon_fcn_lf=objcon_fcn_list{2};
            cost_hf=Cost(1);
            cost_lf=Cost(2);
            ratio_hf=Ratio(1);
            ratio_lf=Ratio(2);

            % hyper parameter
            sample_num_init=4+2*vari_num;
            sample_num_rst=sample_num_init;
            sample_num_add=ceil(log(sample_num_init)/2);

            % GPC sample parameter
            min_bou_interest=1e-3;
            max_bou_interest=1e-1;
            trial_num=min(100*vari_num,100);

            done=0;NFE=0;iter=0;NFE_list=zeros(fidelity_num,1);

            % step 1
            % generate initial data lib
            sample_num_init_hf=ceil(sample_num_init*ratio_hf);
            sample_num_init_lf=ceil(sample_num_init*ratio_lf);

            if isempty(self.datalib_lf)
                self.datalib_lf=struct('objcon_fcn',objcon_fcn_lf,'vari_num',vari_num,'low_bou',low_bou,'up_bou',up_bou,'con_torl',self.con_torl,...
                    'result_best_idx',[],'X',[],'Obj',[],'Con',[],'Coneq',[],'Vio',[],'Ks',[]);
            else
                % load exist data
                self.datalib_lf=struct('objcon_fcn',objcon_fcn_lf,'vari_num',vari_num,'low_bou',low_bou,'up_bou',up_bou,'con_torl',self.con_torl,...
                    'result_best_idx',[],'X',self.datalib_lf.X,'Obj',self.datalib_lf.Obj,'Con',self.datalib_lf.Con,'Coneq',self.datalib_lf.Coneq,'Vio',self.datalib_lf.Vio,'Ks',self.datalib_lf.Ks);
            end
            if size(self.datalib_lf.X,1) < sample_num_init_lf
                X_updata_lf=lhsdesign(sample_num_init_lf-size(self.datalib_lf.X,1),vari_num,'iterations',50,'criterion','maximin').*(up_bou-low_bou)+low_bou;

                % updata data lib by x_list
                [self.datalib_lf,~,~,~,~,~,~,~,NFE_updata]=datalibAdd(self.datalib_lf,X_updata_lf,0,self.WRIRE_FILE_FLAG,self.str_data_file_lf);
                NFE=NFE+NFE_updata*cost_lf;
                NFE_list(2)=NFE_list(2)+NFE_updata;
            end

            if isempty(self.datalib_hf)
                self.datalib_hf=struct('objcon_fcn',objcon_fcn_hf,'vari_num',vari_num,'low_bou',low_bou,'up_bou',up_bou,'con_torl',self.con_torl,...
                    'result_best_idx',[],'X',[],'Obj',[],'Con',[],'Coneq',[],'Vio',[],'Ks',[]);
            else
                % load exist data
                self.datalib_hf=struct('objcon_fcn',objcon_fcn_hf,'vari_num',vari_num,'low_bou',low_bou,'up_bou',up_bou,'con_torl',self.con_torl,...
                    'result_best_idx',[],'X',self.datalib_hf.X,'Obj',self.datalib_hf.Obj,'Con',self.datalib_hf.Con,'Coneq',self.datalib_hf.Coneq,'Vio',self.datalib_hf.Vio,'Ks',self.datalib_hf.Ks);
            end

            if size(self.datalib_hf.X,1) < sample_num_init_hf
                X_updata_hf=self.lhdNSLE(self.datalib_lf.X,sample_num_init_hf-size(self.datalib_hf.X,1),vari_num,low_bou,up_bou);

                % detech expensive constraints and initialize data lib
                [self.datalib_hf,~,~,~,~,~,~,~,NFE_updata]=datalibAdd(self.datalib_hf,X_updata_hf,0,self.WRIRE_FILE_FLAG,self.str_data_file_hf);
                NFE=NFE+NFE_updata*cost_hf;
                NFE_list(1)=NFE_list(1)+NFE_updata;
            end

            % detech expensive constraints
            if ~isempty(self.datalib_hf.Vio)
                self.FLAG_CON=true(1);
            else
                self.FLAG_CON=false(1);
            end

            % find fesiable data in current data lib
            if self.FLAG_CON
                Bool_feas_hf=self.datalib_hf.Vio == 0;
                Bool_feas_lf=self.datalib_lf.Vio == 0;
            end
            Bool_conv_hf=false(size(self.datalib_hf.X,1),1);
            Bool_conv_lf=false(size(self.datalib_lf.X,1),1);

            fmincon_options=optimoptions('fmincon','Display','none','Algorithm','sqp','ConstraintTolerance',0,'FiniteDifferenceStepSize',1e-5);

            result_x_best=zeros(self.iter_max,vari_num);
            result_obj_best=zeros(self.iter_max,1);

            iter=iter+1;

            while ~done
                % step 2
                % nomalization all data by max obj and to create surrogate model
                [XHF,Obj_hf,Con_hf,Coneq_hf,Vio_hf,Ks_hf]=datalibLoad(self.datalib_hf);
                [XLF,Obj_lf,Con_lf,Coneq_lf,Vio_lf,Ks_lf]=datalibLoad(self.datalib_lf);

                XMF={XHF,XLF};Obj_MF={Obj_hf,Obj_lf};
                if ~isempty(Con_hf)
                    Con_MF={Con_hf;Con_lf};
                else
                    Con_MF=[];
                end
                if ~isempty(Coneq_hf)
                    Coneq_MF={Coneq_hf;Coneq_lf};
                else
                    Coneq_MF=[];
                end
                if ~isempty(Vio_hf)
                    Vio_MF={Vio_hf;Vio_lf};
                else
                    Vio_MF=[];
                end
                if ~isempty(Ks_hf)
                    Ks_MF={Ks_hf;Ks_lf};
                else
                    Ks_MF=[];
                end

                [X_surr_MF,Obj_surr_MF,Con_surr_MF,Coneq_surr_MF,Vio_surr_MF,Ks_surr_MF,...
                    obj_max,con_max_list,coneq_max_list,~,~]=self.preSurrData...
                    (fidelity_num,XMF,Obj_MF,Con_MF,Coneq_MF,Vio_MF,Ks_MF,self.nomlz_value);

                % get local infill point, construct surrogate model
                [self.obj_fcn_srgt,self.con_fcn_srgt,output_model]=self.getSrgtFcnMF...
                    (X_surr_MF,Obj_surr_MF,Con_surr_MF,Coneq_surr_MF);
                self.obj_surr=output_model.model_obj;
                self.con_surr_list=output_model.model_con_list;
                self.coneq_surr_list=output_model.model_coneq_list;
                obj_surr_type=output_model.model_obj_type;
                con_surr_type_list=output_model.model_con_type_list;
                coneq_surr_type_list=output_model.model_coneq_type_list;
                [self.ks_fcn_surr,self.ks_surr,~]=self.getBestModel(XMF,Ks_surr_MF);

                % check if all model is SF
                all_SF_flag=true(1);
                if strcmp(obj_surr_type,'MF')
                    all_SF_flag=false(1);
                else
                    if ~isempty(con_surr_type_list)
                        for con_idx =1:length(con_surr_type_list)
                            if strcmp(con_surr_type_list{con_idx},'MF')
                                all_SF_flag=false(1);
                                break;
                            end
                        end
                    end

                    if ~isempty(coneq_surr_type_list)
                        for coneq_idx =1:length(coneq_surr_type_list)
                            if strcmp(coneq_surr_type_list{coneq_idx},'MF')
                                all_SF_flag=false(1);
                                break;
                            end
                        end
                    end
                end

                if all_SF_flag
                    self.add_lf_flag=false(1);
                end

                % step 3
                % updata or identify potential local best
                self.updataLocalPotential...
                    (X_surr_MF{1},Vio_surr_MF{1},vari_num,low_bou,up_bou,con_fcn_cheap,fmincon_options,...
                    obj_max,con_max_list,coneq_max_list);

                % step 4
                % select best potential point as x_infill
                x_infill=self.X_potl(1,:);
                obj_infill_pred=self.Obj_potl(1,:);

                % updata infill point
                [self.datalib_hf,x_infill,obj_infill,con_infill,coneq_infill,vio_infill,ks_infill,repeat_idx,NFE_updata]=...
                    datalibAdd(self.datalib_hf,x_infill,self.protect_range,self.WRIRE_FILE_FLAG,self.str_data_file_hf);
                NFE=NFE+NFE_updata*cost_hf;
                NFE_list(1)=NFE_list(1)+NFE_updata;

                if isempty(x_infill)
                    % process error
                    x_infill=self.datalib_hf.X(repeat_idx,:);
                    obj_infill=self.datalib_hf.Obj(repeat_idx,:);
                    if ~isempty(Con_hf)
                        con_infill=self.datalib_hf.Con(repeat_idx,:);
                    end
                    if ~isempty(Coneq_hf)
                        coneq_infill=self.datalib_hf.Coneq(repeat_idx,:);
                    end
                    if ~isempty(Vio_hf)
                        vio_infill=self.datalib_hf.Vio(repeat_idx,:);
                    end
                else
                    if ~isempty(vio_infill) && vio_infill > 0
                        Bool_feas_hf=[Bool_feas_hf;false(1)];
                    else
                        Bool_feas_hf=[Bool_feas_hf;true(1)];
                    end
                    Bool_conv_hf=[Bool_conv_hf;false(1)];
                end
                self.Obj_potl(1,:)=obj_infill;
                self.Vio_potl(1,:)=vio_infill;

                if self.FLAG_DRAW_FIGURE && vari_num < 3
                    surrogateVisualize(obj_surr,low_bou,up_bou);
                    line(x_infill(1),x_infill(2),obj_infill./obj_max*nomlz_value,'Marker','o','color','r','LineStyle','none');
                end

                % find best result to record
                [XHF,Obj_hf,~,~,Vio_hf,~]=datalibLoad(self.datalib_hf);
                X_unconv_hf=XHF(~Bool_conv_hf,:);
                Obj_unconv_hf=Obj_hf(~Bool_conv_hf,:);
                if ~isempty(Vio_hf)
                    Vio_unconv_hf=Vio_hf(~Bool_conv_hf,:);
                else
                    Vio_unconv_hf=[];
                end
                idx=find(Vio_unconv_hf == 0);
                if isempty(idx)
                    [vio_best,min_idx]=min(Vio_unconv_hf);
                    obj_best=Obj_unconv_hf(min_idx);
                    x_best=X_unconv_hf(min_idx,:);
                else
                    [obj_best,min_idx]=min(Obj_unconv_hf(idx));
                    vio_best=0;
                    x_best=X_unconv_hf(idx(min_idx),:);
                end

                if self.FLAG_INFORMATION
                    fprintf('obj:    %f    violation:    %f    NFE:    %-3d\n',obj_best,vio_best,NFE);
                    %         fprintf('iteration:          %-3d    NFE:    %-3d\n',iteration,NFE);
                    %         fprintf('x:          %s\n',num2str(x_infill));
                    %         fprintf('value:      %f\n',obj_infill);
                    %         fprintf('violation:  %s  %s\n',num2str(con_infill),num2str(coneq_infill));
                    %         fprintf('\n');
                end

                result_x_best(iter,:)=x_best;
                result_obj_best(iter,:)=obj_best;
                iter=iter+1;

                % forced interrupt
                if iter > self.iter_max || NFE >= self.NFE_max
                    done=1;
                end

                % convergence judgment
                if self.FLAG_CONV_JUDGE
                    if ( ((iter > 2) && (abs((obj_infill-obj_infill_old)/obj_infill_old) < self.obj_torl)) && ...
                            ((~isempty(vio_infill) && vio_infill == 0) || isempty(vio_infill)) )
                        done=1;
                    end
                end

                if ~done
                    [XHF,Obj_hf,~,~,Vio_hf,~]=datalibLoad(self.datalib_hf);
                    [XLF,Obj_lf,~,~,Vio_lf,~]=datalibLoad(self.datalib_lf);
                    X_add_hf=[];
                    X_add_lf=[];

                    % check if converage
                    if ( ((iter > 2) && (abs((obj_infill-obj_infill_old)/obj_infill_old) < self.obj_torl)) && ...
                            ((~isempty(vio_infill) && vio_infill == 0) || isempty(vio_infill)) )
                        % resample LHD
                        % step 6.1
                        self.X_local_best=[self.X_local_best;x_infill];
                        self.Obj_local_best=[self.Obj_local_best;obj_infill];
                        self.X_potl=self.X_potl(2:end,:);
                        self.Obj_potl=self.Obj_potl(2:end,:);
                        self.Vio_potl=self.Vio_potl(2:end,:);

                        if isempty(self.X_potl)
                            self.detect_local_flag=true(1);
                            self.add_lf_flag=true(1);
                        end

                        % step 6.2
                        % detech converage
                        for x_idx=1:size(XHF,1)
                            if ~Bool_conv_hf(x_idx)
                                x_single_pred=fmincon(self.obj_fcn_srgt,XHF(x_idx,:),[],[],[],[],low_bou,up_bou,self.con_fcn_srgt,fmincon_options);

                                converage_flag=false(1);

                                for x_check_idx=1:size(self.X_local_best,1)
                                    if sum(abs(self.X_local_best(x_check_idx,:)-x_single_pred),2)/vari_num < self.identiy_torl
                                        converage_flag=true(1);
                                        break;
                                    end
                                end

                                if converage_flag
                                    % if converage to local minimum, set to infeasible
                                    Bool_conv_hf(x_idx)=true(1);
                                end
                            end
                        end

                        for x_idx=1:size(XLF,1)
                            if ~Bool_conv_lf(x_idx)
                                x_single_pred=fmincon(self.obj_fcn_srgt,XLF(x_idx,:),[],[],[],[],low_bou,up_bou,self.con_fcn_srgt,fmincon_options);

                                converage_flag=false(1);

                                for x_check_idx=1:size(self.X_local_best,1)
                                    if sum(abs(self.X_local_best(x_check_idx,:)-x_single_pred),2)/vari_num < self.identiy_torl
                                        converage_flag=true(1);
                                        break;
                                    end
                                end

                                if converage_flag
                                    % if converage to local minimum, set to infeasible
                                    Bool_conv_lf(x_idx)=true(1);
                                end
                            end
                        end

                        % step 6.3
                        % use GPC to limit do not converage to exist local best
                        if ~all(Bool_conv_hf)
                            Class_hf=-1*ones(size(XHF,1),1);
                            Class_hf(Bool_conv_hf)=1; % cannot go into converage area

                            Class_lf=-1*ones(size(XLF,1),1);
                            Class_lf(Bool_conv_lf)=1; % cannot go into converage area

                            if self.add_lf_flag
                                [pred_func_MFGPC,self.model_MFGPC]=self.classifyGaussProcessMultiFidelity(XHF,Class_hf,XLF,Class_lf,self.model_MFGPC.hyp,true(1));
                                con_GPC_fcn=@(x) conFcnGPC(x,pred_func_MFGPC);
                            else
                                [pred_func_GPC,self.model_GPC]=self.classifyGaussProcess(XHF,Class_hf,self.model_GPC.hyp,true(1));
                                con_GPC_fcn=@(x) conFcnGPC(x,pred_func_GPC);
                            end
                        else
                            con_GPC_fcn=[];
                        end

                        % step 6.4
                        % resample latin hypercubic and updata into data lib
                        usable_NFE=self.NFE_max-NFE;
                        sample_num_rst=min(sample_num_rst,ceil(usable_NFE/sum(Ratio)));
                        sample_num_rst_hf=ceil(sample_num_rst*ratio_hf);
                        sample_num_rst_lf=ceil(sample_num_rst*ratio_lf);

                        % resample LF origin
                        try
                            X_add_lf=self.lhdESLHS(sample_num_rst_lf,vari_num,...
                                low_bou,up_bou,XLF,con_GPC_fcn);
                        catch
                            X_add_lf=lhsdesign(sample_num_rst_lf,vari_num).*(up_bou-low_bou)+low_bou;
                        end
                        % resample HF from x_updata_lf
                        X_add_hf=self.lhdNSLE(X_add_lf,sample_num_rst_hf,vari_num,...
                            low_bou,up_bou,XHF);

                        if self.FLAG_DRAW_FIGURE && vari_num < 3
                            classifyVisualization(self.model_MFGPC,low_bou,up_bou);
                            line(XHF(:,1),XHF(:,2),'Marker','o','color','k','LineStyle','none');
                        end
                    else
                        % step 5.1
                        % check if improve
                        improve=0;
                        if isempty(repeat_idx)
                            Bool_comp=(~Bool_conv_hf)&Bool_feas_hf;
                            Bool_comp(end)=false(1);
                            if self.FLAG_CON
                                min_vio=min(Vio_hf(~Bool_conv_hf(1:end-1)));
                                min_obj=min(Obj_hf(Bool_comp));

                                % if all point is infeasible,violation of point infilled is
                                % less than min violation of all point means improve.if
                                % feasible point exist,obj of point infilled is less than min
                                % obj means improve
                                if vio_infill == 0 || vio_infill < min_vio
                                    if ~isempty(min_obj)
                                        if obj_infill < min_obj
                                            % improve, continue local search
                                            improve=1;
                                        end
                                    else
                                        % improve, continue local search
                                        improve=1;
                                    end
                                end
                            else
                                min_obj=min(Obj_hf(Bool_comp));

                                % obj of point infilled is less than min obj means improve
                                if obj_infill < min_obj
                                    % imporve, continue local search
                                    improve=1;
                                end
                            end
                        end

                        % step 5.2
                        % if obj no improve, use GPC to identify interest area
                        % than, imporve interest area surrogate quality
                        if ~improve
                            if self.add_lf_flag
                                % construct GPCMF
                                train_num=min(size(self.datalib_hf.X,1),ceil((11*vari_num-1+25)/(ratio_hf+ratio_lf)));
                                [pred_func_MFGPC,x_pareto_center]=self.trainFilterMF(Ratio,x_infill,train_num,Bool_conv_hf,Bool_conv_lf);
                                con_GPC_fcn=@(x) conFcnGPC(x,pred_func_MFGPC);
                            else
                                % construct GPC
                                train_num=min(size(self.datalib_hf.X,1),11*vari_num-1+25);
                                [pred_fcn_GPC,x_pareto_center]=self.trainFilter(x_infill,train_num,Bool_conv_hf);
                                con_GPC_fcn=@(x) self.conFcnGPC(x,pred_fcn_GPC);
                            end

                            % step 5.3
                            % identify interest area
                            center_point=fmincon(con_GPC_fcn,x_pareto_center,[],[],[],[],low_bou,up_bou,con_fcn_cheap,fmincon_options);

                            bou_interest=abs(center_point-x_infill);
                            bou_interest=max(min_bou_interest.*(up_bou-low_bou),bou_interest);
                            bou_interest=min(max_bou_interest.*(up_bou-low_bou),bou_interest);

                            % generate trial point
                            usable_NFE=self.NFE_max-NFE;
                            sample_num_rst=min(sample_num_add,ceil(usable_NFE/sum(Ratio)));
                            sample_num_rst_hf=ceil(sample_num_rst*ratio_hf);
                            sample_num_rst_lf=ceil(sample_num_rst*ratio_lf);

                            trial_point=repmat(x_infill,trial_num,1);
                            for variable_idx=1:vari_num
                                trial_point(:,variable_idx)=trial_point(:,variable_idx)+...
                                    normrnd(0,bou_interest(variable_idx),[trial_num,1]);
                            end
                            trial_point=max(trial_point,low_bou);
                            trial_point=min(trial_point,up_bou);

                            Bool_negetive=pred_func_MFGPC(trial_point) == -1;
                            if sum(Bool_negetive) < sample_num_rst_lf
                                value=con_GPC_fcn(trial_point);
                                thres=quantile(value,0.25);
                                Bool_negetive=value<thres;
                            end
                            trial_point=trial_point(Bool_negetive,:);

                            % step 5.4
                            if self.add_lf_flag
                                % select LF point from trial_point

                                max_dist=0;
                                iter_select=1;
                                while iter_select < 100
                                    select_idx=randperm(size(trial_point,1),sample_num_rst_lf);
                                    dist=self.calMinDistanceIter(trial_point(select_idx,:),XLF);
                                    if max_dist < dist
                                        X_add_lf=trial_point(select_idx,:);
                                        max_dist=dist;
                                    end
                                    iter_select=iter_select+1;
                                end

                                % select HF point from LF
                                max_dist=0;
                                iter_select=1;
                                while iter_select < 100
                                    select_idx=randperm(size(X_add_lf,1),sample_num_rst_hf);
                                    dist=self.calMinDistanceIter(X_add_lf(select_idx,:),XHF);
                                    if max_dist < dist
                                        X_add_hf=X_add_lf(select_idx,:);
                                        max_dist=dist;
                                    end
                                    iter_select=iter_select+1;
                                end
                            else
                                X_add_lf=[];

                                % select HF point from trial_point
                                max_dist=0;
                                iter_select=1;
                                while iter_select < 100
                                    select_idx=randperm(size(trial_point,1),sample_num_rst_hf);
                                    dist=self.calMinDistanceIter(trial_point(select_idx,:),XHF);
                                    if max_dist < dist
                                        X_add_hf=trial_point(select_idx,:);
                                        max_dist=dist;
                                    end
                                    iter_select=iter_select+1;
                                end
                            end

                            if self.FLAG_DRAW_FIGURE && vari_num < 3
                                classifyVisualization(self.model_MFGPC,low_bou,up_bou);
                                line(trial_point(:,1),trial_point(:,2),'Marker','o','color','k','LineStyle','none');
                                line(X_add_hf(:,1),X_add_hf(:,2),'Marker','o','color','g','LineStyle','none');
                            end
                        end
                    end

                    % step 7
                    % updata data lib
                    [self.datalib_hf,X_add_hf,~,~,~,Vio_updata_hf,~,~,NFE_updata]=...
                        datalibAdd(self.datalib_hf,X_add_hf,self.protect_range,self.WRIRE_FILE_FLAG,self.str_data_file_hf);
                    NFE=NFE+NFE_updata*cost_hf;
                    NFE_list(1)=NFE_list(1)+NFE_updata;
                    Bool_feas_hf=[Bool_feas_hf;Vio_updata_hf==0];
                    Bool_conv_hf=[Bool_conv_hf;false(size(X_add_hf,1),1)];

                    [self.datalib_lf,X_add_lf,~,~,~,Vio_updata_lf,~,~,NFE_updata]=...
                        datalibAdd(self.datalib_lf,X_add_lf,self.protect_range,self.WRIRE_FILE_FLAG,self.str_data_file_lf);
                    NFE=NFE+NFE_updata*cost_lf;
                    NFE_list(2)=NFE_list(2)+NFE_updata;
                    Bool_feas_lf=[Bool_feas_lf;Vio_updata_lf==0];
                    Bool_conv_lf=[Bool_conv_lf;false(size(X_add_lf,1),1)];

                    % forced interrupt
                    if iter > self.iter_max || NFE >= self.NFE_max
                        done=1;
                    end
                end

                obj_infill_old=obj_infill;

                % save iteration
                if ~isempty(self.dataoptim_filestr)
                    datalib=self.datalib;
                    save(self.dataoptim_filestr,'datalib');
                end
            end

            % find best result to record
            x_best=self.datalib_hf.X(self.datalib_hf.result_best_idx(end),:);
            obj_best=self.datalib_hf.Obj(self.datalib_hf.result_best_idx(end),:);

            result_x_best=result_x_best(1:iter-1,:);
            result_obj_best=result_obj_best(1:iter-1);

            output.result_x_best=result_x_best;
            output.result_obj_best=result_obj_best;

            output.x_local_best=self.X_local_best;
            output.obj_local_best=self.Obj_local_best;
            output.NFE_list=NFE_list;

            output.datalib_hf=self.datalib_hf;
            output.datalib_lf=self.datalib_lf;

            function [con,coneq]=conFcnGPC(x,pred_fcn_GPC)
                % function to obtain probability predict function
                %
                [~,~,con]=pred_fcn_GPC(x);
                coneq=[];
            end

            function distance_min__=calMinDistanceIter...
                    (x_list__,x_exist_list__)
                % get distance min from x_list
                % this version do not consider distance between x exist
                %

                % sort x_supply_list_initial to decrese distance calculate times
                x_list__=sortrows(x_list__,1);
                [sample_number__,variable_number__]=size(x_list__);
                distance_min__=variable_number__;
                for x_idx__=1:sample_number__
                    x_curr__=x_list__(x_idx__,:);
                    x_next_idx__=x_idx__ + 1;
                    % only search in min_distance(x_list had been sort)
                    search_range__=variable_number__;
                    while x_next_idx__ <= sample_number__ &&...
                            (x_list__(x_next_idx__,1)-x_list__(x_idx__,1))^2 ...
                            < search_range__
                        x_next__=x_list__(x_next_idx__,:);
                        distance_temp__=sum((x_next__-x_curr__).^2);
                        if distance_temp__ < distance_min__
                            distance_min__=distance_temp__;
                        end
                        if distance_temp__ < search_range__
                            search_range__=distance_temp__;
                        end
                        x_next_idx__=x_next_idx__+1;
                    end
                    for x_exist_idx=1:size(x_exist_list__,1)
                        x_next__=x_exist_list__(x_exist_idx,:);
                        distance_temp__=sum((x_next__-x_curr__).^2);
                        if distance_temp__ < distance_min__
                            distance_min__=distance_temp__;
                        end
                    end
                end
                distance_min__=sqrt(distance_min__);
            end

        end

        function [obj_fcn_srgt,con_fcn_srgt,output]=getSrgtFcnMF...
                (self,XMF,Obj_MF,Con_MF,Coneq_MF)
            % base on input data to generate surrogate predict function
            % con_fcn_srgt if format of nonlcon function in fmincon
            % judge MF and SF quality and select best one
            %

            [pred_func_obj,model_obj,model_obj_type]=self.getBestModel(XMF,Obj_MF);

            if ~isempty(Con_MF)
                con_number=size(Con_MF{1},2);
                pred_funct_con_list=cell(con_num,1ber);
                model_con_list=cell(con_num,1ber);
                model_con_type_list=cell(con_num,1ber);
                for con_idx=1:con_number
                    [pred_funct_con_list{con_idx},model_con_list{con_idx},model_con_type_list{con_idx}]=self.getBestModel...
                        (XMF,{Con_MF{1}(:,con_idx),Con_MF{2}(:,con_idx)});
                end
            else
                pred_funct_con_list=[];
                model_con_list=[];
                model_con_type_list=[];
            end

            if ~isempty(Coneq_MF)
                coneq_number=size(Coneq_MF{1},2);
                pred_funct_coneq=cell(coneq_num,1ber);
                model_coneq_list=cell(coneq_num,1ber);
                model_coneq_type_list=cell(coneq_num,1ber);
                for coneq_idx=1:size(Coneq_MF,2)
                    [pred_funct_coneq{coneq_idx},model_coneq_list{coneq_idx},model_coneq_type_list{coneq_idx}]=self.getBestModel...
                        (XMF,{Coneq_MF{1}(:,coneq_idx),Coneq_MF{2}(:,coneq_idx)});
                end
            else
                pred_funct_coneq=[];
                model_coneq_list=[];
                model_coneq_type_list=[];
            end

            obj_fcn_srgt=@(X_pred) objFcnSurr(X_pred,pred_func_obj);
            if isempty(pred_funct_con_list) && isempty(pred_funct_coneq)
                con_fcn_srgt=[];
            else
                con_fcn_srgt=@(X_pred) conFcnSurr(X_pred,pred_funct_con_list,pred_funct_coneq);
            end

            output.model_obj=model_obj;
            output.model_con_list=model_con_list;
            output.model_coneq_list=model_coneq_list;
            output.model_obj_type=model_obj_type;
            output.model_con_type_list=model_con_type_list;
            output.model_coneq_type_list=model_coneq_type_list;

            function obj=objFcnSurr...
                    (X_pred,pred_fcn_obj)
                % connect all predict favl
                %
                obj=pred_fcn_obj(X_pred);
            end

            function [con,coneq]=conFcnSurr...
                    (X_pred,pred_fcn_con,pred_fcn_coneq)
                % connect all predict con and coneq
                %
                if isempty(pred_fcn_con)
                    con=[];
                else
                    con=zeros(size(X_pred,1),length(pred_fcn_con));
                    for con_idx__=1:length(pred_fcn_con)
                        con(:,con_idx__)=....
                            pred_fcn_con{con_idx__}(X_pred);
                    end
                end
                if isempty(pred_fcn_coneq)
                    coneq=[];
                else
                    coneq=zeros(size(X_pred,1),length(pred_fcn_coneq));
                    for coneq_idx__=1:length(pred_fcn_coneq)
                        coneq(:,coneq_idx__)=...
                            pred_fcn_coneq{coneq_idx__}(X_pred);
                    end
                end
            end

        end

        function updataLocalPotential...
                (self,X_surr,Vio_model,vari_num,low_bou,up_bou,con_fcn_cheap,fmincon_options,...
                obj_max,con_max_list,coneq_max_list)
            if ~isempty(self.con_fcn_srgt) || ~isempty(con_fcn_cheap)
                con_fcn=@(x) self.conFcnTotal...
                    (x,self.con_fcn_srgt,con_fcn_cheap);
            else
                con_fcn=[];
            end

            if self.detect_local_flag
                % detech potential local best point
                for x_idx=1:size(X_surr,1)
                    x_initial=X_surr(x_idx,:);
                    [x_potl,obj_potl_pred,exit_flag,output_fmincon]=fmincon(self.obj_fcn_srgt,x_initial,[],[],[],[],...
                        low_bou,up_bou,con_fcn,fmincon_options);

                    if exit_flag == 1 || exit_flag == 2
                        % check if x_potl have existed
                        add_flag=true(1);
                        for x_check_idx=1:size(self.X_potl,1)
                            if sum(abs(self.X_potl(x_check_idx,:)-x_potl),2)/vari_num < self.identiy_torl
                                add_flag=false(1);
                                break;
                            end
                        end
                        for x_check_idx=1:size(self.X_local_best,1)
                            if sum(abs(self.X_local_best(x_check_idx,:)-x_potl),2)/vari_num < self.identiy_torl
                                add_flag=false(1);
                                break;
                            end
                        end

                        % updata into self.X_potl
                        if add_flag
                            self.X_potl=[self.X_potl;x_potl];
                            self.Obj_potl=[self.Obj_potl;obj_potl_pred/self.nomlz_value.*obj_max];
                            [con_potential,coneq_potential]=self.con_fcn_srgt(x_potl);
                            if ~isempty(con_potential)
                                con_potential=con_potential/self.nomlz_value.*con_max_list;
                            end
                            if ~isempty(coneq_potential)
                                coneq_potential=coneq_potential/self.nomlz_value.*coneq_max_list;
                            end
                            self.Vio_potl=[self.Vio_potl;calViolation(con_potential,coneq_potential,self.con_torl)];
                        end
                    end
                end

                % if self.X_potl is empty, try to use KS surrogate as x potential
                if isempty(self.X_potl)
                    [~,x_idx]=min(Vio_model);
                    x_initial=X_surr(x_idx,:);
                    [x_potl,~,exit_flag,output_fmincon]=fmincon(self.ks_fcn_surr,x_initial,[],[],[],[],...
                        low_bou,up_bou,con_fcn_cheap,fmincon_options);
                    obj_potl_pred=self.obj_fcn_srgt(x_potl);

                    self.X_potl=[self.X_potl;x_potl];
                    self.Obj_potl=[self.Obj_potl;obj_potl_pred/self.nomlz_value*obj_max];
                    [con_potential,coneq_potential]=self.con_fcn_srgt(x_potl);
                    if ~isempty(con_potential)
                        con_potential=con_potential/self.nomlz_value.*con_max_list;
                    end
                    if ~isempty(coneq_potential)
                        coneq_potential=coneq_potential/self.nomlz_value.*coneq_max_list;
                    end
                    self.Vio_potl=[self.Vio_potl;calViolation(con_potential,coneq_potential,self.con_torl)];
                end

                % sort X potential by Vio
                [self.Vio_potl,idx]=sort(self.Vio_potl);
                self.Obj_potl=self.Obj_potl(idx,:);
                self.X_potl=self.X_potl(idx,:);

                self.detect_local_flag=false(1);
            else
                % updata X potential
                for x_idx=1:size(self.X_potl,1)
                    x_potl=self.X_potl(x_idx,:);

                    [x_potl,obj_potl_pred,exit_flag,output_fmincon]=fmincon(self.obj_fcn_srgt,x_potl,[],[],[],[],...
                        low_bou,up_bou,con_fcn,fmincon_options);

                    self.X_potl(x_idx,:)=x_potl;
                    self.Obj_potl(x_idx,:)=obj_potl_pred/self.nomlz_value*obj_max;
                    [con_potential,coneq_potential]=self.con_fcn_srgt(x_potl);
                    if ~isempty(con_potential)
                        con_potential=con_potential/self.nomlz_value.*con_max_list;
                    end
                    if ~isempty(coneq_potential)
                        coneq_potential=coneq_potential/self.nomlz_value.*coneq_max_list;
                    end
                    self.Vio_potl(x_idx,:)=calViolation(con_potential,coneq_potential,self.con_torl);
                end

                % merge X potential
                % Upward merge
                for x_idx=size(self.X_potl,1):-1:1
                    x_potl=self.X_potl(x_idx,:);

                    % check if x_potl have existed
                    merge_flag=false(1);
                    for x_check_idx=1:x_idx-1
                        if sum(abs(self.X_potl(x_check_idx,:)-x_potl),2)/vari_num < self.identiy_torl
                            merge_flag=true(1);
                            break;
                        end
                    end

                    % updata into self.X_potl
                    if merge_flag
                        self.X_potl(x_idx,:)=[];
                        self.Obj_potl(x_idx,:)=[];
                        self.Vio_potl(x_idx,:)=[];
                    end
                end
            end

            % sort X potential by Obj
            flag=find(self.Vio_potl == 0, 1, 'last' );
            if isempty(flag) % mean do not have fesaible point
                [self.Obj_potl,idx]=sort(self.Obj_potl);
                self.Vio_potl=self.Vio_potl(idx,:);
                self.X_potl=self.X_potl(idx,:);
            else
                [self.Obj_potl(1:flag,:),idx_feas]=sort(self.Obj_potl(1:flag,:));
                [self.Obj_potl(flag+1:end,:),idx_infeas]=sort(self.Obj_potl(flag+1:end,:));
                idx=[idx_feas;idx_infeas+flag];

                self.Vio_potl=self.Vio_potl(idx,:);
                self.X_potl=self.X_potl(idx,:);
            end

        end

        function [pred_fcn_MFGPC,x_pareto_center]=trainFilterMF(self,Ratio,x_infill,train_num,Bool_conv_hf,Bool_conv_lf)
            % train filter of gaussian process classifer
            %
            ratio_hf=Ratio(1);
            ratio_lf=Ratio(2);

            [XHF,Obj_hf,~,~,~,Ks_hf]=datalibLoad(self.datalib_hf);
            train_num_hf=min(train_num*ratio_hf,size(XHF,1));
            % base on distance select usable point
            x_distance=sum(abs(XHF-x_infill),2);
            [~,idx]=sort(x_distance);
            Obj_hf=Obj_hf(idx(1:train_num_hf),:);
            Ks_hf=Ks_hf(idx(1:train_num_hf),:);
            XHF=XHF(idx(1:train_num_hf),:);
            Bool_conv_hf=Bool_conv_hf(idx(1:train_num_hf),:);

            [XLF,Obj_lf,~,~,~,Ks_lf]=datalibLoad(self.datalib_lf);
            train_num_lf=min(train_num*ratio_lf,size(XLF,1));
            % base on distance select usable point
            x_distance=sum(abs(XLF-x_infill),2);
            [~,idx]=sort(x_distance);
            Obj_lf=Obj_lf(idx(1:train_num_lf),:);
            Ks_lf=Ks_lf(idx(1:train_num_lf),:);
            XLF=XLF(idx(1:train_num_lf),:);
            Bool_conv_lf=Bool_conv_lf(idx(1:train_num_lf),:);

            if self.FLAG_CON
                % base on filter to decide which x should be choose
                pareto_idx_list=self.getParetoFront([Obj_hf,Ks_hf]);

                Class_hf=ones(size(XHF,1),1);
                Class_hf(pareto_idx_list)=-1;
                Class_hf(Bool_conv_hf)=1; % cannot go into convarage area

                x_pareto_center=sum(XHF(pareto_idx_list,:),1)/length(pareto_idx_list);

                pareto_idx_list=self.getParetoFront([Obj_lf,Ks_lf]);

                Class_lf=ones(size(XLF,1),1);
                Class_lf(pareto_idx_list)=-1;
                Class_lf(Bool_conv_lf)=1; % cannot go into convarage area

                [pred_fcn_MFGPC,self.model_MFGPC]=self.classifyGaussProcessMultiFidelity...
                    (XHF,Class_hf,XLF,Class_lf,self.model_MFGPC.hyp,true(1));

            else
                obj_threshold=prctile(Obj_hf,50-40*sqrt(NFE/self.NFE_max));
                Class_hf=ones(size(XHF,1),1);
                Class_hf(Obj_hf < obj_threshold)=-1;
                Class_hf(Bool_conv_hf)=1; % cannot go into convarage area

                Class_lf=ones(size(XLF,1),1);
                Class_lf(Obj_lf < obj_threshold)=-1;
                Class_lf(Bool_conv_lf)=1; % cannot go into convarage area

                x_pareto_center=sum(XHF(Obj < obj_threshold,:),1)/sum(Obj_hf < obj_threshold);

                [pred_fcn_MFGPC,self.model_MFGPC]=self.classifyGaussProcessMultiFidelity...
                    (XHF,Class_hf,XLF,Class_lf,self.model_MFGPC.hyp,true(1));

            end

        end

        function [pred_func_GPC,x_pareto_center]=trainFilter(self,x_infill,train_num,Bool_conv)
            % train filter of gaussian process classifer
            %
            [X,Obj,~,~,~,Ks]=datalibLoad(self.datalib_hf);
            % base on distance select usable point
            x_dist=sum(abs(X-x_infill),2);
            [~,idx]=sort(x_dist);
            Obj=Obj(idx(1:train_num),:);
            Ks=Ks(idx(1:train_num),:);
            X=X(idx(1:train_num),:);
            Bool_conv=Bool_conv(idx(1:train_num),:);

            if self.FLAG_CON
                % base on filter to decide which x should be choose
                %     pareto_idx_list=self.getParetoFront([Obj(~Bool_feas),Ks(~Bool_feas)]);
                pareto_idx_list=self.getParetoFront([Obj,Ks]);

                Class=ones(size(X,1),1);
                Class(pareto_idx_list)=-1;
                %     Class(Bool_feas)=-1; % can go into feasiable area
                Class(Bool_conv)=1; % cannot go into convarage area

                x_pareto_center=sum(X(pareto_idx_list,:),1)/length(pareto_idx_list);

                [pred_func_GPC,self.model_GPC]=self.classifyGaussProcess(X,Class,self.model_GPC.hyp,true(1));
            else
                obj_threshold=prctile(Obj,50-40*sqrt(NFE/self.NFE_max));

                Class=ones(size(X,1),1);
                Class(Obj < obj_threshold)=-1;
                Class(Bool_conv)=1; % cannot go into convarage area

                x_pareto_center=sum(X(Obj < obj_threshold,:),1)/sum(Obj < obj_threshold);

                [pred_func_GPC,self.model_GPC]=self.classifyGaussProcess(X,Class,self.model_GPC.hyp,true(1));
            end
        end

        function [pred_fcn,model,type]=getBestModel(self,XMF,Obj_MF)
            % judge use single fidelity of mulit fidelity by R^2
            %

            xhf_num=size(XMF{1},1);
            [pred_func_MF,model_obj_MF]=self.srgtsfRBFMultiFidelityPreModel...
                (XMF{1},Obj_MF{1},[],XMF{2},Obj_MF{2},[]);
            error_MF=( model_obj_MF.beta./diag(model_obj_MF.H(:,1:xhf_num)\eye(xhf_num))+...
                model_obj_MF.alpha./diag(model_obj_MF.H(:,xhf_num+1:end)\eye(xhf_num)) )*model_obj_MF.stdD_Y;
            Rsq_MF=1-sum(error_MF.^2)/sum((mean(Obj_MF{1})-Obj_MF{1}).^2);

            [pred_func_SF,model_obj_SF]=self.srgtsfRBF...
                (XMF{1},Obj_MF{1});
            error_SF=(model_obj_SF.beta./diag(model_obj_SF.inv_RBF_matrix))*model_obj_SF.stdD_Y;
            Rsq_SF=1-sum(error_SF.^2)/sum((mean(Obj_MF{1})-Obj_MF{1}).^2);

            if Rsq_MF > Rsq_SF
                pred_fcn=pred_func_MF;
                model=model_obj_MF;
                type='MF';
            else
                pred_fcn=pred_func_SF;
                model=model_obj_SF;
                type='SF';
            end

        end
    end

    methods(Static) % auxiliary function
        function [X_model_MF,Obj_model_MF,Con_model_MF,Coneq_model_MF,Vio_model_MF,Ks_model_MF,...
                obj_max,con_max_list,coneq_max_list,vio_max,ks_max]=preSurrData...
                (fidelity_number,XMF,Obj_MF,Con_MF,Coneq_MF,Vio_MF,Ks_MF,nomlz_value)
            % normalize data to construct surrogate model
            %
            X_model_MF=XMF;

            [Obj_model_MF,obj_max]=calNormalizeData...
                (fidelity_number,Obj_MF,nomlz_value);

            if ~isempty(Con_MF)
                [Con_model_MF,con_max_list]=calNormalizeData...
                    (fidelity_number,Con_MF,nomlz_value);
            else
                Con_model_MF=[];
                con_max_list=[];
            end

            if ~isempty(Coneq_MF)
                [Coneq_model_MF,coneq_max_list]=calNormalizeData...
                    (fidelity_number,Coneq_MF,nomlz_value);
            else
                Coneq_model_MF=[];
                coneq_max_list=[];
            end

            if ~isempty(Vio_MF)
                [Vio_model_MF,vio_max]=calNormalizeData...
                    (fidelity_number,Vio_MF,nomlz_value);
            else
                Vio_model_MF=[];
                vio_max=[];
            end

            if ~isempty(Ks_MF)
                [Ks_model_MF,ks_max]=calNormalizeData...
                    (fidelity_number,Ks_MF,nomlz_value);
            else
                Ks_model_MF=[];
                ks_max=[];
            end

            function [Data_model_MF,data_max]=calNormalizeData...
                    (fidelity_number,Data_MF,nomlz_value)
                Data_model_MF=cell(1,fidelity_number);
                Data_total=[];
                for fidelity_idx=1:fidelity_number
                    Data_total=[Data_total;Data_MF{fidelity_idx}];
                end
                data_max=max(abs(Data_total),[],1);
                for fidelity_idx=1:fidelity_number
                    Data_model_MF{fidelity_idx}=Data_MF{fidelity_idx}./data_max*nomlz_value;
                end
            end
        end

        function pareto_idx_list=getParetoFront(data_list)
            % distinguish pareto front of data list
            % data_list is x_number x data_number matrix
            % notice if all data of x1 is less than x2,x1 domain x2
            %
            x_number=size(data_list,1);
            pareto_idx_list=[]; % sort all idx of filter point list

            % select no domain filter
            for x_idx=1:x_number
                data=data_list(x_idx,:);
                pareto_idx=1;
                add_filter_flag=1;
                while pareto_idx <= length(pareto_idx_list)
                    % compare x with exit pareto front point
                    x_pareto_idx=pareto_idx_list(pareto_idx,:);

                    % contain constraint of x_filter
                    data_pareto=data_list(x_pareto_idx,:);

                    % compare x with x_pareto
                    judge=data >= data_pareto;
                    if ~sum(~judge)
                        add_filter_flag=0;
                        break;
                    end

                    % if better or equal than exit pareto point,reject pareto point
                    judge=data <= data_pareto;
                    if ~sum(~judge)
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

        function [con,coneq]=conFcnTotal...
                (x,con_fcn,con_fcn_cheap)
            con=[];
            coneq=[];

            if ~isempty(con_fcn)
                [expencon,expenconeq]=con_fcn(x);
                con=[con,expencon];
                coneq=[coneq,expenconeq];
            end

            if ~isempty(con_fcn_cheap)
                [expencon,expenconeq]=con_fcn_cheap(x);
                con=[con,expencon];
                coneq=[coneq,expenconeq];
            end

        end

        function [con,coneq]=conFcnGPC(x,pred_fcn_GPC)
            % function to obtain probability predict function
            %
            [~,~,con]=pred_fcn_GPC(x);
            coneq=[];
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

    methods(Static) % machine learning

    end

    methods(Static) % LHD
        function [X_sample,dist_min_nomlz,X_total]=lhdESLHS...
                (sample_number,variable_number,...
                low_bou,up_bou,X_exist,cheapcon_function)
            % generate latin hypercube design
            % ESLHS method is used(sample and iteration)
            % election combination mode of point and find best combination
            %
            % input:
            % sample_number(new point to sample), variable_number, ...
            % low_bou, up_bou, X_exist(exist point), cheapcon_function
            %
            % output:
            % X_sample, dist_min_nomlz(min distance of normalize data), ...
            % X_total(include all data in area)
            %
            % reference: [1] LONG T, LI X, SHI R, et al., Gradient-Free
            % Trust-Region-Based Adaptive Response Surface Method for Expensive
            % Aircraft Optimization[J]. AIAA Journal, 2018, 56(2): 862-73.
            %
            % Copyright 2023 03 Adel
            %
            if nargin < 6
                cheapcon_function=[];
                if nargin < 5
                    X_exist=[];
                    if nargin < 4
                        up_bou=ones(1,variable_number);
                        if nargin < 3
                            low_bou=zeros(1,variable_number);
                            if nargin < 2
                                error('getLatinHypercube: lack variable_number');
                            end
                        end
                    end
                end
            end

            iteration_max=100*sample_number;

            % check x_exist_list if meet boundary
            if ~isempty(X_exist)
                if size(X_exist,2) ~= variable_number
                    error('getLatinHypercube: x_exist_list variable_number error');
                end
                X_exist_nomlz=(X_exist-low_bou)./(up_bou-low_bou);
            else
                X_exist_nomlz=[];
            end

            exist_number=size(X_exist,1);
            total_number=sample_number+exist_number;
            if sample_number <= 0
                X_total=X_exist;
                X_sample=[];
                dist_min_nomlz=getMinDistance(X_exist_nomlz);
                return;
            end

            % get quasi-feasible point
            x_initial_number=100*sample_number;
            x_quasi_number=10*sample_number;
            if ~isempty(cheapcon_function)
                X_supply_quasi_nomlz=[];

                % check if have enough X_supply_nomlz
                iteration=0;
                while size(X_supply_quasi_nomlz,1) < x_quasi_number && iteration < 100
                    X_supply_initial_nomlz=lhsdesign(x_initial_number,variable_number);

                    qusai_index=[];
                    for x_index=1:size(X_supply_initial_nomlz,1)
                        if cheapcon_function(X_supply_initial_nomlz(x_index,:).*(up_bou-low_bou)+low_bou) <= 0
                            qusai_index=[qusai_index,x_index];
                        end
                    end
                    X_supply_quasi_nomlz=[X_supply_quasi_nomlz;X_supply_initial_nomlz(qusai_index,:)];

                    iteration=iteration+1;
                end

                if iteration == 100 && isempty(X_supply_quasi_nomlz)
                    error('getLatinHypercube: feasible quasi point cannot be found');
                end
            else
                X_supply_quasi_nomlz=lhsdesign(x_quasi_number,variable_number);
            end

            % iterate and get final x_supply_list
            iteration=0;
            x_supply_quasi_number=size(X_supply_quasi_nomlz,1);
            dist_min_nomlz=0;
            X_sample_nomlz=[];

            % dist_min_nomlz_result=zeros(1,iteration);
            while iteration <= iteration_max
                % random select x_new_number X to X_trial_nomlz
                x_select_index=randperm(x_supply_quasi_number,sample_number);

                % get distance min itertion X_
                distance_min_iteration=getMinDistanceIter...
                    (X_supply_quasi_nomlz(x_select_index,:),X_exist_nomlz);

                % if distance_min_iteration is large than last time
                if distance_min_iteration > dist_min_nomlz
                    dist_min_nomlz=distance_min_iteration;
                    X_sample_nomlz=X_supply_quasi_nomlz(x_select_index,:);
                end

                iteration=iteration+1;
                %     dist_min_nomlz_result(iteration)=dist_min_nomlz;
            end
            dist_min_nomlz=getMinDistance([X_sample_nomlz;X_exist_nomlz]);
            X_sample=X_sample_nomlz.*(up_bou-low_bou)+low_bou;
            X_total=[X_exist;X_sample];

            function distance_min__=getMinDistance(x_list__)
                % get distance min from x_list
                %
                if isempty(x_list__)
                    distance_min__=[];
                    return;
                end

                % sort x_supply_list_initial to decrese distance calculate times
                x_list__=sortrows(x_list__,1);
                [sample_number__,variable_number__]=size(x_list__);
                distance_min__=variable_number__;
                for x_index__=1:sample_number__
                    x_curr__=x_list__(x_index__,:);
                    x_next_index__=x_index__ + 1;
                    % only search in min_distance(x_list had been sort)
                    search_range__=variable_number__;
                    while x_next_index__ <= sample_number__ &&...
                            (x_list__(x_next_index__,1)-x_list__(x_index__,1))^2 ...
                            < search_range__
                        x_next__=x_list__(x_next_index__,:);
                        distance_temp__=sum((x_next__-x_curr__).^2);
                        if distance_temp__ < distance_min__
                            distance_min__=distance_temp__;
                        end
                        if distance_temp__ < search_range__
                            search_range__=distance_temp__;
                        end
                        x_next_index__=x_next_index__+1;
                    end
                end
                distance_min__=sqrt(distance_min__);
            end
            function distance_min__=getMinDistanceIter...
                    (x_list__,x_exist_list__)
                % get distance min from x_list
                % this version do not consider distance between x exist
                %

                % sort x_supply_list_initial to decrese distance calculate times
                x_list__=sortrows(x_list__,1);
                [sample_number__,variable_number__]=size(x_list__);
                distance_min__=variable_number__;
                for x_index__=1:sample_number__
                    x_curr__=x_list__(x_index__,:);
                    x_next_index__=x_index__ + 1;
                    % only search in min_distance(x_list had been sort)
                    search_range__=variable_number__;
                    while x_next_index__ <= sample_number__ &&...
                            (x_list__(x_next_index__,1)-x_list__(x_index__,1))^2 ...
                            < search_range__
                        x_next__=x_list__(x_next_index__,:);
                        distance_temp__=sum((x_next__-x_curr__).^2);
                        if distance_temp__ < distance_min__
                            distance_min__=distance_temp__;
                        end
                        if distance_temp__ < search_range__
                            search_range__=distance_temp__;
                        end
                        x_next_index__=x_next_index__+1;
                    end
                    for x_exist_index=1:size(x_exist_list__,1)
                        x_next__=x_exist_list__(x_exist_index,:);
                        distance_temp__=sum((x_next__-x_curr__).^2);
                        if distance_temp__ < distance_min__
                            distance_min__=distance_temp__;
                        end
                    end
                end
                distance_min__=sqrt(distance_min__);
            end
        end

        function [X_sample,dist_min_nomlz,X_total]=lhdNSLE...
                (X_base,sample_number,variable_number,...
                low_bou,up_bou,X_exist)
            % generate nested latin hypercube design
            % SLE method is used(sample and iteration, select max min distance group)
            % election combination mode of point and find best combination
            %
            % input:
            % X_base(which will be sample), sample number(new point to sample), ...
            % variable_number, low_bou, up_bou, X_exist(exist point)
            %
            % output:
            % X_sample, dist_min_nomlz(min distance of normalize data)
            % X_total(include all data in area)
            %
            if nargin < 6
                X_exist=[];
                if nargin < 5
                    up_bou=ones(1,variable_number);
                    if nargin < 4
                        low_bou=zeros(1,variable_number);
                        if nargin < 3
                            error('getLatinHypercube: lack input');
                        end
                    end
                end
            end

            iteration_max=100*sample_number;

            % check x_exist_list if meet boundary
            if ~isempty(X_exist)
                if size(X_exist,2) ~= variable_number
                    error('getLatinHypercube: x_exist_list variable_number error');
                end
                index=find(X_exist < low_bou);
                index=[index,find(X_exist > up_bou)];
                if ~isempty(index)
                    error('getLatinHypercube: x_exist_list range error');
                end
                X_exist_nomlz=(X_exist-low_bou)./(up_bou-low_bou);
            else
                X_exist_nomlz=[];
            end

            exist_number=size(X_exist,1);
            total_number=sample_number+exist_number;
            if sample_number <= 0
                X_total=X_exist;
                X_sample=[];
                dist_min_nomlz=getMinDistance(X_exist_nomlz);
                return;
            end

            % get quasi-feasible point
            X_base_nomlz=(X_base-low_bou)./(up_bou-low_bou);

            % iterate and get final x_supply_list
            iteration=0;
            x_supply_quasi_number=size(X_base_nomlz,1);
            dist_min_nomlz=0;
            X_sample_nomlz=[];

            % dist_min_nomlz_result=zeros(1,iteration);
            while iteration <= iteration_max
                % random select x_new_number X to X_trial_nomlz
                x_select_index=randperm(x_supply_quasi_number,sample_number);

                % get distance min itertion X_
                distance_min_iteration=getMinDistanceIter...
                    (X_base_nomlz(x_select_index,:),X_exist_nomlz);

                % if distance_min_iteration is large than last time
                if distance_min_iteration > dist_min_nomlz
                    dist_min_nomlz=distance_min_iteration;
                    X_sample_nomlz=X_base_nomlz(x_select_index,:);
                end

                iteration=iteration+1;
                %     dist_min_nomlz_result(iteration)=dist_min_nomlz;
            end
            dist_min_nomlz=getMinDistance([X_sample_nomlz;X_exist_nomlz]);
            X_sample=X_sample_nomlz.*(up_bou-low_bou)+low_bou;
            X_total=[X_exist;X_sample];

            function distance_min__=getMinDistance(x_list__)
                % get distance min from x_list
                %
                if isempty(x_list__)
                    distance_min__=[];
                    return;
                end

                % sort x_supply_list_initial to decrese distance calculate times
                x_list__=sortrows(x_list__,1);
                [sample_number__,variable_number__]=size(x_list__);
                distance_min__=variable_number__;
                for x_index__=1:sample_number__
                    x_curr__=x_list__(x_index__,:);
                    x_next_index__=x_index__ + 1;
                    % only search in min_distance(x_list had been sort)
                    search_range__=variable_number__;
                    while x_next_index__ <= sample_number__ &&...
                            (x_list__(x_next_index__,1)-x_list__(x_index__,1))^2 ...
                            < search_range__
                        x_next__=x_list__(x_next_index__,:);
                        distance_temp__=sum((x_next__-x_curr__).^2);
                        if distance_temp__ < distance_min__
                            distance_min__=distance_temp__;
                        end
                        if distance_temp__ < search_range__
                            search_range__=distance_temp__;
                        end
                        x_next_index__=x_next_index__+1;
                    end
                end
                distance_min__=sqrt(distance_min__);
            end
            function distance_min__=getMinDistanceIter...
                    (x_list__,x_exist_list__)
                % get distance min from x_list
                % this version do not consider distance between x exist
                %

                % sort x_supply_list_initial to decrese distance calculate times
                x_list__=sortrows(x_list__,1);
                [sample_number__,variable_number__]=size(x_list__);
                distance_min__=variable_number__;
                for x_index__=1:sample_number__
                    x_curr__=x_list__(x_index__,:);
                    x_next_index__=x_index__ + 1;
                    % only search in min_distance(x_list had been sort)
                    search_range__=variable_number__;
                    while x_next_index__ <= sample_number__ &&...
                            (x_list__(x_next_index__,1)-x_list__(x_index__,1))^2 ...
                            < search_range__
                        x_next__=x_list__(x_next_index__,:);
                        distance_temp__=sum((x_next__-x_curr__).^2);
                        if distance_temp__ < distance_min__
                            distance_min__=distance_temp__;
                        end
                        if distance_temp__ < search_range__
                            search_range__=distance_temp__;
                        end
                        x_next_index__=x_next_index__+1;
                    end
                    for x_exist_index=1:size(x_exist_list__,1)
                        x_next__=x_exist_list__(x_exist_index,:);
                        distance_temp__=sum((x_next__-x_curr__).^2);
                        if distance_temp__ < distance_min__
                            distance_min__=distance_temp__;
                        end
                    end
                end
                distance_min__=sqrt(distance_min__);
            end
        end

    end

end

%% data library function
function [datalib,x_new,obj_new,con_new,coneq_new,vio_new,ks_new,repeat_idx,NFE]=datalibAdd...
    (datalib,x_origin_new,protect_range,write_file_flag,str_data_file)
% updata data lib
% updata format:
% variable_number,obj_number,con_number,coneq_number
% x,obj,con,coneq
%
if nargin < 3
    protect_range=0;
end

[x_new_num,~]=size(x_origin_new);
x_new=[];
obj_new=[];
con_new=[];
coneq_new=[];
vio_new=[];
ks_new=[];
repeat_idx=[];
NFE=0;

if write_file_flag
    file_data=fopen(str_data_file,'a');
end

% updata format:
% variable_number,obj_number,con_number,coneq_number
% x,obj,con,coneq
for x_idx=1:x_new_num
    x=x_origin_new(x_idx,:);

    if protect_range ~= 0
        % updata data with same_point_avoid protect
        % check x_potl if exist in data lib
        % if exist, jump updata
        distance=sum((abs(x-datalib.X)./(datalib.up_bou-datalib.low_bou)),2);
        [distance_min,min_idx]=min(distance);
        if distance_min < datalib.vari_num*protect_range
            % distance to exist point of point to add is small than protect_range
            repeat_idx=[repeat_idx;min_idx];
            continue;
        end
    end

    [obj,con,coneq]=datalib.objcon_fcn(x); % eval value
    NFE=NFE+1;

    con=con(:)';
    coneq=coneq(:)';
    % calculate vio
    if isempty(con) && isempty(coneq)
        vio=[];
        ks=[];
    else
        vio=calViolation(con,coneq,datalib.con_torl);
        ks=max([con,coneq]);
    end


    x_new=[x_new;x];
    obj_new=[obj_new;obj];
    if ~isempty(con)
        con_new=[con_new;con];
    end
    if ~isempty(coneq)
        coneq_new=[coneq_new;coneq];
    end
    if ~isempty(vio)
        vio_new=[vio_new;vio];
    end
    if ~isempty(ks)
        ks_new=[ks_new;ks];
    end

    if write_file_flag
        % write data to txt_result
        fprintf(file_data,'%d ',repmat('%.8e ',1,datalib.vari_num));
        fprintf(file_data,'%d ',length(obj));
        fprintf(file_data,'%d ',length(con));
        fprintf(file_data,'%d ',length(coneq));

        fprintf(file_data,datalib.x_format,x);
        fprintf(file_data,repmat('%.8e ',1,length(obj)),obj);
        fprintf(file_data,repmat('%.8e ',1,length(con)),con);
        fprintf(file_data,repmat('%.8e ',1,length(coneq)),coneq);
        fprintf(file_data,'\n');
    end

    datalib=dataJoin(datalib,x,obj,con,coneq,vio,ks);

    % record best
    if isempty(datalib.result_best_idx)
        datalib.result_best_idx=1;
    else
        if isempty(vio) || vio == 0
            if obj <= datalib.Obj(datalib.result_best_idx(end))
                datalib.result_best_idx=[datalib.result_best_idx;size(datalib.X,1)];
            else
                datalib.result_best_idx=[datalib.result_best_idx;datalib.result_best_idx(end)];
            end
        else
            if vio <= datalib.Vio(datalib.result_best_idx(end))
                datalib.result_best_idx=[datalib.result_best_idx;size(datalib.X,1)];
            else
                datalib.result_best_idx=[datalib.result_best_idx;datalib.result_best_idx(end)];
            end
        end
    end
end

if write_file_flag
    fclose(file_data);
    clear('file_data');
end
end

function [x_list,obj_list,con_list,coneq_list,vio_list,ks_list]=datalibLoad...
    (datalib,low_bou,up_bou)
% updata data to exist data lib
%
if nargin < 3
    up_bou=realmax;
    if nargin < 2
        low_bou=-realmax;
    end
end

idx=[];
for x_idx=1:size(datalib.X,1)
    x=datalib.X(x_idx,:);
    if all(x > low_bou) && all(x < up_bou)
        idx=[idx;x_idx];
    end
end

x_list=datalib.X(idx,:);
obj_list=datalib.Obj(idx,:);
if ~isempty(datalib.Con)
    con_list=datalib.Con(idx,:);
else
    con_list=[];
end
if ~isempty(datalib.Coneq)
    coneq_list=datalib.Coneq(idx,:);
else
    coneq_list=[];
end
if ~isempty(datalib.Vio)
    vio_list=datalib.Vio(idx,:);
else
    vio_list=[];
end
if ~isempty(datalib.Ks)
    ks_list=datalib.Ks(idx);
else
    ks_list=[];
end
end

function datalib=dataJoin(datalib,x,obj,con,coneq,vio,ks)
% updata data to exist data lib
%
datalib.X=[datalib.X;x];
datalib.Obj=[datalib.Obj;obj];
if ~isempty(datalib.Con) || ~isempty(con)
    datalib.Con=[datalib.Con;con];
end
if ~isempty(datalib.Coneq) || ~isempty(coneq)
    datalib.Coneq=[datalib.Coneq;coneq];
end
if ~isempty(datalib.Vio) || ~isempty(vio)
    datalib.Vio=[datalib.Vio;vio];
end
if ~isempty(datalib.Ks) || ~isempty(ks)
    datalib.Ks=[datalib.Ks;ks];
end
end

function vio_list=calViolation(con_list,coneq_list,con_torl)
% calculate violation of data
%
if isempty(con_list) && isempty(coneq_list)
    vio_list=[];
else
    vio_list=zeros(max(size(con_list,1),size(coneq_list,1)),1);
    if ~isempty(con_list)
        vio_list=vio_list+sum(max(con_list-con_torl,0),2);
    end
    if ~isempty(coneq_list)
        vio_list=vio_list+sum(max(abs(coneq_list)-con_torl,0),2);
    end
end
end
