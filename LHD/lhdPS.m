function [X_sample,dist_min_nomlz,X_total]=lhdPS(sample_num,vari_num,...
    low_bou,up_bou,X_exist,con_fcn_cheap)
% generate latin hypercube desgin
%
% more uniform point distribution by simulating particle motion
%
% input:
% sample number(new point to sample),variable_number
% low_bou,up_bou,x_exist_list,con_fcn_cheap
%
% output:
% X_sample,dist_min_nomlz(min distance of normalize data)
% X_total,include all data in area
%
% Copyright 2023 3 Adel
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
                    error('getLatinHypercube: lack variable_number');
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

exist_num=size(X_exist,1);
total_num=sample_num+exist_num;
if sample_num < 0
    X_total=X_exist;
    X_sample=[];
    dist_min_nomlz=getMinDistance(X_exist_nomlz);
    return;
end

low_bou_nomlz=zeros(1,vari_num);
up_bou_nomlz=ones(1,vari_num);

% obtain initial point
if ~isempty(con_fcn_cheap)
    % obtian feasible point
    X_quasi_nomlz=[];

    % check if have enough X_supply_nomlz
    iter=0;
    while size(X_quasi_nomlz,1) < 10*sample_num && iter < 500
        X_quasi_nomlz_initial=lhsdesign(10*sample_num,vari_num);

        qusai_index=[];
        for x_index=1:size(X_quasi_nomlz_initial,1)
            if con_fcn_cheap(X_quasi_nomlz_initial(x_index,:).*(up_bou-low_bou)+low_bou) <= 0
                qusai_index=[qusai_index,x_index];
            end
        end
        X_quasi_nomlz=[X_quasi_nomlz;X_quasi_nomlz_initial(qusai_index,:)];

        iter=iter+1;
    end

    if iter == 500 && size(X_quasi_nomlz,1) < sample_num
        error('getLatinHypercube: feasible quasi point cannot be found');
    end

    % use fuzzy clustering to get feasible point center
    model_FCM=clusterFCM(X_quasi_nomlz,sample_num,2);
    X_sample_nomlz=model_FCM.center_list;
    X_feas_center_nomlz=X_sample_nomlz;

    scatter(X_quasi_nomlz(:,1),X_quasi_nomlz(:,2));
    hold on;
    scatter(X_sample_nomlz(:,1),X_sample_nomlz(:,2),'red');
    hold off;
else
    X_sample_nomlz=rand(sample_num,vari_num);
end

% pic_num=1;

iter=0;
fval_list=zeros(sample_num,1);
gradient_list=zeros(sample_num,vari_num);
while iter < iter_max    
    % change each x place by newton methods
    for x_index=1:sample_num
        
        % get gradient
        [fval_list(x_index,1),gradient_list(x_index,:)]=calParticleEnergy...
            (X_sample_nomlz(x_index,:),[X_sample_nomlz(1:x_index-1,:);X_sample_nomlz(x_index+1:end,:);X_exist_nomlz],...
            sample_num,vari_num,low_bou_nomlz-1/2/sample_num,up_bou_nomlz+1/2/sample_num);
        
%         energy_function=@(x) calParticleEnergy...
%             ([x],[X_sample_nomlz(1:x_index-1,:);X_sample_nomlz(x_index+1:end,:);X_exist_nomlz],...
%             sample_number,variable_number,low_bou_nomlz-1/2/sample_number,up_bou_nomlz+1/2/sample_number);
%         drawFcn(energy_function,low_bou_nomlz(1:2),up_bou_nomlz(1:2))
        
%         [fval,gradient]=differ(energy_function,X_sample_nomlz(x_index,:));
%         gradient'-gradient_list(x_index,:)
    end

    C=(1-iter/iter_max/2)*0.5;
    
    % updata partical location
    for x_index=1:sample_num
        x=X_sample_nomlz(x_index,:);
        gradient=gradient_list(x_index,:);

        % check if feasible
        if ~isempty(con_fcn_cheap)
            con=con_fcn_cheap(x.*(up_bou-low_bou)+low_bou);
            % if no feasible,move point to close point
            if con > 0
                %                 % search closest point
                %                 dx_center=x-X_feas_center_nomlz;
                %                 [~,index]=min(norm(dx_center,"inf"));
                %                 gradient=dx_center(index(1),:);

                gradient=x-X_feas_center_nomlz(x_index,:);
            end
        end

        gradient=min(gradient,0.5);
        gradient=max(gradient,-0.5);
        x=x-gradient*C;

        boolean=x < low_bou_nomlz;
        x(boolean)=-x(boolean);
        boolean=x > up_bou_nomlz;
        x(boolean)=2-x(boolean);
        X_sample_nomlz(x_index,:)=x;
    end
    
    iter=iter+1;

    scatter(X_sample_nomlz(:,1),X_sample_nomlz(:,2));
    bou=[low_bou_nomlz(1:2);up_bou_nomlz(1:2)];
    axis(bou(:));
    grid on;
%     
%     radius=1;
%     hold on;
%     rectangle('Position',[-radius,-radius,2*radius,2*radius],'Curvature',[1 1])
%     hold off;
%     
    drawnow;
%     F=getframe(gcf);
%     I=frame2im(F);
%     [I,map]=rgb2ind(I,256);
%     if pic_num == 1
%         imwrite(I,map,'show_trajectory_constrain.gif','gif','Loopcount',inf,'DelayTime',0.1);
%     else
%         imwrite(I,map,'show_trajectory_constrain.gif','gif','WriteMode','append','DelayTime',0.1);
%     end
%     pic_num=pic_num + 1;
end

% process out of boundary point
for x_index=1:sample_num
    x=X_sample_nomlz(x_index,:);
    % check if feasible
    if ~isempty(con_fcn_cheap)
        con=con_fcn_cheap(x);
        % if no feasible,move point to close point
        if con > 0
            % search closest point
            dx_center=x-X_feas_center_nomlz;
            [~,index]=min(norm(dx_center,"inf"));

            gradient=dx_center(index(1),:);
        end
    end
    x=x-gradient*C;

    boolean=x < low_bou_nomlz;
    x(boolean)=-x(boolean);
    boolean=x > up_bou_nomlz;
    x(boolean)=2-x(boolean);
    X_sample_nomlz(x_index,:)=x;
end

dist_min_nomlz=getMinDistance([X_sample_nomlz;X_exist_nomlz]);
X_sample=X_sample_nomlz.*(up_bou-low_bou)+low_bou;
X_total=[X_sample;X_exist];

    function [fval,gradient]=calParticleEnergy...
            (x,X_surplus,sample_number,variable_number,low_bou,up_bou)
        % function describe distance between X and X_supply
        % x is colume vector and X_surplus is matrix which is num-1 x var
        % low_bou_limit__ and up_bou_limit__ is colume vector
        % variable in colume
        %
        a__=10;
        a_bou__=10;
        
        sign__=((x > X_surplus)-0.5)*2;
        
        xi__=-a__*(x-X_surplus).*sign__;
        psi__=a_bou__*(low_bou-x);
        zeta__=a_bou__*(x-up_bou);
        
        exp_psi__=exp(psi__);
        exp_zeta__=exp(zeta__);

%         sum_xi__=sum(xi__,2)/variable_number;
%         exp_sum_xi__=exp(sum_xi__);
%         % get fval
%         fval=sum(exp_sum_xi__,1)+...
%             sum(exp_psi__+exp_zeta__,2)/variable_number;

%         exp_xi__=exp(xi__);
%         sum_exp_xi__=sum(exp_xi__,2);
%         % get fval
%         fval=sum(sum_exp_xi__,1)/variable_number/sample_number+...
%             sum(exp_psi__+exp_zeta__,2)/variable_number;

        sum_xi__=sum(xi__,2)/variable_number;
        exp_sum_xi__=exp(sum_xi__);
        exp_xi__=exp(xi__);
        sum_exp_xi__=sum(exp_xi__,2)/variable_number;
        % get fval
        fval=(sum(sum_exp_xi__,1)+sum(exp_sum_xi__,1))/2/sample_number+...
            sum(exp_psi__+exp_zeta__,2)/variable_number*0.1;
        
%         % get gradient
%         gradient=sum(-a__*sign__.*exp_sum_xi__,1)/variable_number+...
%             (-a_bou__*exp_psi__+a_bou__*exp_zeta__)/variable_number;

%         % get gradient
%         gradient=sum(-a__*sign__.*exp_xi__,1)/variable_number/sample_number+...
%             (-a_bou__*exp_psi__+a_bou__*exp_zeta__)/variable_number;

        % get gradient
        gradient=(sum(-a__*sign__.*exp_sum_xi__,1)+sum(-a__*sign__.*exp_xi__,1))/2/variable_number/sample_number+...
            (-a_bou__*exp_psi__+a_bou__*exp_zeta__)/variable_number*0.1;
        
    end

    function distance_min__=getMinDistance(x_list__)
        % get distance min from x_list
        %
        if isempty(x_list__)
            distance_min__=[];
            return;
        end

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
