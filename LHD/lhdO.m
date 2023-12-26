function [X_sample,dist_min_nomlz,X_total]=lhdO(sample_num,vari_num,...
    low_bou,up_bou,X_exist,con_fcn_cheap)
% generate sample sequence latin hypercube
% iteration optimal method is used
% sample number is total point in area
% default low_bou is 0, up_bou is 1, con_fcn_cheap is []
% low_bou and up_bou is colume vector
% x in x_exist_list, x_list, supply_x_list is row vector
% x_exist_list should meet bou
%
% Copyright 2022 Adel
%
if nargin < 6
    con_fcn_cheap=[];
    if nargin < 5
        X_exist=[];
        if nargin < 4
            up_bou=[];
            if nargin < 3
                low_bou=[];
                if nargin < 2
                    error('lhdO: lack vari_num');
                end
            end
        end
    end
end

if isempty(low_bou),low_bou=zeros(1,vari_num);end
if isempty(up_bou),up_bou=ones(1,vari_num);end

iter_max=min(10*sample_num,100);

% check x_exist_list if meet boundary
if ~isempty(X_exist)
    if size(X_exist,2) ~= vari_num
        error('lhdO: vari_num of X_exist inequal to input vari_num');
    end
    X_exist_nomlz=(X_exist-low_bou)./(up_bou-low_bou);
else
    X_exist_nomlz=[];
end

% check x_new_number
x_new_number=sample_num-size(X_exist,1);
if x_new_number < 0
    X_total=X_exist;
    X_sample=[];
    dist_min_nomlz=getMinDistance(X_exist_nomlz);
    return;
end

X_new_nomlz=lhsdesign(x_new_number,vari_num);

% x is nomalize, so constraint function should change
if ~isempty(con_fcn_cheap)
    con_fcn_cheap=@(x) max(sample_num*con_fcn_cheap(x.*(up_bou-low_bou)+low_bou)+1,0);
end
low_bou_nomlz=zeros(1,vari_num);
up_bou_nomlz=ones(1,vari_num);

iter=1;
while iter <= iter_max
    for X_new_index=1:x_new_number
        % iteration
        object_function=@(x) objectFunctionXPlace...
            (x,[X_new_nomlz(1:X_new_index-1,:);X_new_nomlz(X_new_index+1:end,:);X_exist_nomlz],...
            vari_num,low_bou_nomlz,up_bou_nomlz,con_fcn_cheap);
        %         drawFcn(object_function,low_bou_nomlz,up_bou_nomlz,100,-inf,1)
        %         fminunc_options=optimoptions('fminunc','display','none','StepTolerance',1e-2);
        %         [x,~,~,output]=fminunc(object_function,X_new(X_new_index,:)',fminunc_options);
        x=optimalNewton(object_function,X_new_nomlz(X_new_index,:));
        x(find(x > 1))=1;
        x(find(x < 0))=0;
        X_new_nomlz(X_new_index,:)=x;
    end
    iter=iter+1;
end
dist_min_nomlz=getMinDistance([X_new_nomlz;X_exist_nomlz]);
X_sample=X_new_nomlz.*(up_bou-low_bou)+low_bou;
X_total=[X_sample;X_exist];

    function x=optimalNewton(object_function,x,torlance,max_iteration__)
        % simple newton method optimal
        %
        if nargin < 4
            max_iteration__=50;
            if nargin < 3
                torlance=1e-2;
                if nargin < 2
                    error('lhdO:optimalNewton: lack x');
                end
            end
        end
        done__=0;
        iteration__=1;
        
        while ~done__
            [~,gradient,hessian]=object_function(x);
            %             if rcond(hessian) <1e-6
            %                 disp('error');
            %             end
            x=x-hessian\gradient;
            if norm(gradient,2) < torlance || iteration__ >= max_iteration__
                done__=1;
            end
            iteration__=iteration__+1;
        end
    end

    function [fval,gradient,hessian]=objectFunctionXPlace...
            (x,X_surplus,variable_number,low_bou,up_bou,con_fcn_cheap)
        % function describe distance between X and X_supply
        % X__ is colume vector and X_supply__ is matrix which is num-1 x var
        % low_bou_limit__ and up_bou_limit__ is colume vector
        % variable in colume
        %
        [~,variable_number__]=size(X_surplus);
        
        sigma__=10;
        boundary__=0.1^variable_number__;
        
        sign__=((x > X_surplus)-0.5)*2;
        
        xi__=-sigma__*(x-X_surplus).*sign__;
        sum_xi__=sum(xi__,1);
        psi__=sigma__*(low_bou-x);
        zeta__=sigma__*(x-up_bou);
        
        exp_sum_xi__=exp(sum_xi__);
        exp_psi__=exp(psi__);
        exp_zeta__=exp(zeta__);
        
        xi_DF=-sigma__*sign__;
        % sum_xi_DF=sum(xi_DF,2);
        psi_DF=-sigma__*ones(variable_number__,1);
        zeta_DF=sigma__*ones(variable_number__,1);
        
        % get fval
        fval=sum(exp_sum_xi__,2)+...
            sum(boundary__*exp_psi__+...
            boundary__*exp_zeta__,1);
        
        % get gradient
        gradient=sum(exp_sum_xi__.*xi_DF,2)+...
            boundary__*exp_psi__.*psi_DF+...
            boundary__*exp_zeta__.*zeta_DF;
        
        % get hessian
        hessian=exp_sum_xi__.*xi_DF*xi_DF'+...
            diag(boundary__*exp_psi__.*psi_DF.*psi_DF+...
            boundary__*exp_zeta__.*zeta_DF.*zeta_DF);
        
        if ~isempty(con_fcn_cheap)
            fval_con=con_fcn_cheap(x);
            fval=fval+fval_con;
            [gradient_con,hessian_con]=differ...
                (con_fcn_cheap,x,fval_con,variable_number);
            gradient=gradient+gradient_con;
            hessian=hessian+hessian_con;
        end
        
        function [gradient,hessian]=differ(differ_function,x,fval,variable_number,step)
            % differ function to get gradient and hessian
            %
            if nargin < 5
                step=1e-6;
            end
            fval__=zeros(variable_number,2); % backward is 1, forward is 2
            gradient=zeros(1,variable_number);
            hessian=zeros(variable_number);
            
            fval__(:,2)=differ_function(x);
            % fval and gradient
            for variable_index__=1:variable_number
                x_forward__=x;
                x_backward__=x;
                x_backward__(variable_index__)=x_backward__(variable_index__)-step;
                fval__(variable_index__,1)=differ_function(x_backward__);
                
                x_forward__(variable_index__)=x_forward__(variable_index__)+step;
                fval__(variable_index__,2)=differ_function(x_forward__);
                
                gradient(variable_index__)=...
                    (fval__(variable_index__,2)-fval__(variable_index__,1))/2/step;
            end
            
            % hessian
            for variable_index__=1:variable_number
                hessian(variable_index__,variable_index__)=...
                    (fval__(variable_index__,2)-2*fval+fval__(variable_index__,1))/step/step;
                for variable_index_next__=variable_index__+1:variable_number
                    x_for_for=x;
                    x_for_for(variable_index__)=x_for_for(variable_index__)+step;
                    x_for_for(variable_index_next__)=x_for_for(variable_index_next__)+step;
                    
                    hessian(variable_index__,variable_index_next__)=(...
                        differ_function(x_for_for)-...
                        fval__(variable_index__,2)-fval__(variable_index_next__,2)+...
                        fval...
                        )/step/step;
                    hessian(variable_index_next__,variable_index__)=...
                        hessian(variable_index__,variable_index_next__);
                end
            end
        end
    end

    function distance_min__=getMinDistance(x_list__)
        % get distance min from x_list
        %
        
        % sort x_supply_list_initial to decrese distance calculate times
        x_list__=sortrows(x_list__,1);
        sample_number__=size(x_list__,1);
        variable_number__=size(x_list__,2);
        distance_min__=variable_number__;
        for x_index__=1:sample_number__
            x_curr__=x_list__(x_index__,:);
            x_next_index__=x_index__ + 1;
            % first dimension only search in min_distance
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
end
