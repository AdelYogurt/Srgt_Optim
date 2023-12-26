classdef DataLibrary < matlab.mixin.Copyable
    properties
        model_function=[];
        variable_number=[];
        up_bou=[];
        low_bou=[];
        nonlcon_torlance=[];

        x_list=[];
        fval_list=[];
        con_list=[];
        coneq_list=[];
        vio_list=[];
        ks_list=[];

        x_best=[];

        filename_data;
        write_file_flag;

        value_format;
    end
    methods
        function self = DataLibrary(model_function,variable_number,low_bou,up_bou,...
                nonlcon_torlance,filename_data,write_file_flag)
            if nargin < 7 || isempty(write_file_flag)
                self.write_file_flag = true(1);
                if nargin < 6
                    filename_data = [];
                    if nargin < 5
                        nonlcon_torlance = [];
                    end
                end
            else
                self.write_file_flag = write_file_flag;
            end

            if isempty(filename_data)
                self.filename_data = 'result_total.txt';
            else
                if length(filename_data) < 4 || ~strcmp(filename_data(end-3:end),'.txt')
                    filename_data = [filename_data,'.txt'];
                end
                self.filename_data = filename_data;
            end

            if isempty(nonlcon_torlance)
                self.nonlcon_torlance = 0;
            else
                self.nonlcon_torlance = nonlcon_torlance;
            end

            if self.write_file_flag
                file_result = fopen(self.filename_data,'a');
                fprintf(file_result,'%s\n',datetime);
                fclose(file_result);
                clear('file_result');
            end

            % store format
            self.value_format = '%.8e ';

            self.model_function=model_function;
            self.variable_number=variable_number;
            self.low_bou=low_bou;
            self.up_bou=up_bou;
        end

        function [x_new,fval_new,con_new,coneq_new,vio_new,ks_new,repeat_index,NFE] = dataUpdata...
                (self,x_origin_new,protect_range)
            % updata data library
            % updata format:
            % variable_number,fval_number,con_number,coneq_number
            % x,fval,con,coneq
            %
            if nargin < 3
                protect_range = 0;
            end

            [x_new_num,~] = size(x_origin_new);
            x_new = [];
            fval_new = [];
            con_new = [];
            coneq_new = [];
            vio_new = [];
            ks_new = [];
            repeat_index = [];
            NFE = 0;

            if self.write_file_flag
                file_data = fopen('result_total.txt','a');
            end

            % updata format:
            % variable_number,fval_number,con_number,coneq_number
            % x,fval,con,coneq
            for x_index = 1:x_new_num
                x = x_origin_new(x_index,:);

                if protect_range ~= 0
                    % updata data with same_point_avoid protect
                    % check x_potential if exist in data library
                    % if exist, jump updata
                    distance = sum((abs(x-self.x_list)./(self.up_bou-self.low_bou)),2);
                    [distance_min,min_index] = min(distance);
                    if distance_min < self.variable_number*protect_range
                        % distance to exist point of point to add is small than protect_range
                        repeat_index = [repeat_index;min_index];
                        continue;
                    end
                end

                [fval,con,coneq] = self.model_function(x); % eval value
                NFE = NFE+1;

                fval = fval(:)';
                con = con(:)';
                coneq = coneq(:)';
                % calculate vio
                vio = self.calViolation(con,coneq,self.nonlcon_torlance);
                ks = max([con,coneq]);

                x_new = [x_new;x];
                fval_new = [fval_new;fval];
                if ~isempty(con)
                    con_new = [con_new;con];
                end
                if ~isempty(coneq)
                    coneq_new = [coneq_new;coneq];
                end
                if ~isempty(vio)
                    vio_new = [vio_new;vio];
                end
                if ~isempty(ks)
                    ks_new = [ks_new;ks];
                end

                if self.write_file_flag
                    % write data to txt_result
                    fprintf(file_data,'%d ',self.variable_number);
                    fprintf(file_data,'%d ',length(fval));
                    fprintf(file_data,'%d ',length(con));
                    fprintf(file_data,'%d ',length(coneq));

                    fprintf(file_data,self.x_format,x);
                    fprintf(file_data,repmat(self.value_format,1,length(fval)),fval);
                    fprintf(file_data,repmat(self.value_format,1,length(con)),con);
                    fprintf(file_data,repmat(self.value_format,1,length(coneq)),coneq);
                    fprintf(file_data,'\n');
                end

                self.dataJoin(x,fval,con,coneq,vio,ks);
            end

            if self.write_file_flag
                fclose(file_data);
                clear('file_data');
            end
        end

        %         function [x_list,fval_list,con_list,coneq_list] = dataFileRead...
        %                 (data_library_name)
        %             % load data from data library
        %             % low_bou,up_bou is range of data
        %             % updata format:
        %             % variable_number,fval_number,con_number,coneq_number
        %             % x,fval,con,coneq
        %             %
        %             if nargin < 3
        %                 up_bou = inf;
        %                 if nargin < 2
        %                     low_bou = -inf;
        %                     if nargin < 1
        %                         error('dataLibraryLoad: lack data_library_name');
        %                     end
        %                 end
        %             end
        %
        %             if ~strcmp(data_library_name(end-3:end),'.txt')
        %                 data_library_name = [data_library_name,'.txt'];
        %             end
        %
        %             % updata format:
        %             % variable_number,fval_number,con_number,coneq_number
        %             % x,fval,con,coneq
        %             if exist(data_library_name,'file') == 2
        %                 data_list = importdata(data_library_name);
        %                 if ~isempty(data_list)
        %                     % search whether exist point
        %                     x_list = [];
        %                     fval_list = [];
        %                     con_list = [];
        %                     coneq_list = [];
        %
        %                     for data_index = 1:size(data_list,1)
        %                         data = data_list(data_index,:);
        %
        %                         variable_number = data(1);
        %                         fval_number = data(2);
        %                         con_number = data(3);
        %                         coneq_number = data(4);
        %
        %                         base = 5;
        %                         x = data(base:base+variable_number-1);
        %                         judge = sum(x < low_bou)+sum(x > up_bou);
        %                         if ~judge
        %                             x_list = [x_list;x];
        %                             base = base+variable_number;
        %                             fval_list = [fval_list;data(base:base+fval_number-1)];
        %                             base = base+fval_number;
        %                             con = data(base:base+con_number-1);
        %                             if ~isempty(con)
        %                                 con_list = [con_list;con];
        %                             end
        %                             base = base+con_number;
        %                             coneq = data(base:base+coneq_number-1);
        %                             if ~isempty(coneq)
        %                                 coneq_list = [coneq_list;coneq];
        %                             end
        %                         end
        %                     end
        %                 else
        %                     x_list = [];
        %                     fval_list = [];
        %                     con_list = [];
        %                     coneq_list = [];
        %                 end
        %             else
        %                 x_list = [];
        %                 fval_list = [];
        %                 con_list = [];
        %                 coneq_list = [];
        %             end
        %         end

        function dataJoin(self,x,fval,con,coneq,vio,ks)
            % updata data to exist data library
            %
            self.x_list = [self.x_list;x];
            self.fval_list = [self.fval_list;fval];
            if ~isempty(self.con_list) || ~isempty(con)
                self.con_list = [self.con_list;con];
            end
            if ~isempty(self.coneq_list) || ~isempty(coneq)
                self.coneq_list = [self.coneq_list;coneq];
            end
            if ~isempty(self.vio_list) || ~isempty(vio)
                self.vio_list = [self.vio_list;vio];
            end
            if ~isempty(self.ks_list) || ~isempty(ks)
                self.ks_list = [self.ks_list;ks];
            end
        end

        function [x_list,fval_list,con_list,coneq_list,vio_list,ks_list] = dataLoad...
                (self,low_bou,up_bou)
            % updata data to exist data library
            %
            if nargin < 3
                up_bou = realmax;
                if nargin < 2
                    low_bou = -realmax;
                end
            end
            
            index=[];
            for x_index=1:size(self.x_list,1)
                x=self.x_list(x_index,:);
                if all(x > low_bou) && all(x < up_bou)
                    index=[index;x_index];
                end
            end

            x_list = self.x_list(index,:);
            fval_list = self.fval_list(index,:);
            if ~isempty(self.con_list)
                con_list = self.con_list(index,:);
            else
                con_list = [];
            end
            if ~isempty(self.coneq_list)
                coneq_list = self.coneq_list(index,:);
            else
                coneq_list = [];
            end
            if ~isempty(self.vio_list)
                vio_list = self.vio_list(index,:);
            else
                vio_list = [];
            end
            if ~isempty(self.ks_list)
                ks_list = self.ks_list(index);
            else
                ks_list = [];
            end
        end

        function vio_list = calViolation(self,con_list,coneq_list,nonlcon_torlance)
            % calculate violation of data
            %
            if isempty(con_list) && isempty(coneq_list)
                vio_list = [];
            else
                vio_list = zeros(max(size(con_list,1),size(coneq_list,1)),1);
                if ~isempty(con_list)
                    vio_list = vio_list+sum(max(con_list-nonlcon_torlance,0),2);
                end
                if ~isempty(coneq_list)
                    vio_list = vio_list+sum((abs(coneq_list)-nonlcon_torlance),2);
                end
            end
        end
    end
end