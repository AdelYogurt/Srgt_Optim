function model_MS=clusterMS(X,bandwidth)
% generate mean shift clustering model
% do not need to input clustering number
% clustering method is move point to value center
%
if nargin < 2
    bandwidth=[];
end

% normalize data
[x_num,vari_num]=size(X);
aver_X=mean(X);
stdD_X=std(X);stdD_X(stdD_X == 0)=1;
X_nomlz=(X-aver_X)./stdD_X;
data=X_nomlz;
index=zeros(x_num,1);

if isempty(bandwidth)
    bandwidth=1/vari_num;
end

iter_max=1000;
torl=1e-4;
identify_torl=1e-2;

center_list=[];
for x_idx=1:x_num
    x=data(x_idx,:);
    shift=inf;iter=0;
    
    while norm(shift) > torl && iter < iter_max
        dist_data=pdist2(x,data);
        weights=exp(-(dist_data.^2)/(2*bandwidth^2));
        shift=sum(data.*weights',1)/sum(weights)-x; % calculate Mean Shift vector
        x=x+shift;

        iter=iter+1;
    end
    
    if isempty(center_list)
        % new center
        center_list=[center_list;x];
        index(x_idx)=size(center_list,1);
    else
        dis_cen=vecnorm(x-center_list,2,2);
        if ~any(dis_cen < identify_torl)
            % new center
            center_list=[center_list;x];
            index(x_idx)=size(center_list,1);
        else
            index(x_idx)=find(dis_cen < identify_torl,1);
        end
    end
end

% normalize data
center_list=center_list.*stdD_X+aver_X;

model_MS.X=X;
model_MS.aver_X=aver_X;
model_MS.stdD_X=stdD_X;

model_MS.index=index;
model_MS.center_list=center_list;
end
