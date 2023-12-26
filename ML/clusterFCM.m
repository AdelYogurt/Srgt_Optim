function model_FCM=clusterFCM(X,classify_num,m)
% generate fuzzy cluster model
% X is x_number x vari_num matrix
% center_list is classify_num x vari_num matrix
%
if nargin < 3
    m=[];
end

iter_max=100;
torl=1e-6;

% normalize data
[x_num,vari_num]=size(X);
aver_X=mean(X);
stdD_X=std(X);stdD_X(stdD_X == 0)=1;
X_nomlz=(X-aver_X)./stdD_X;

if isempty(m)
    m=2;
end

% if x_number equal 1,clustering cannot done
if x_num == 1
    center_list=X;

    model_FCM.X=X;
    model_FCM.aver_X=aver_X;
    model_FCM.stdD_X=stdD_X;

    model_FCM.center_list=center_list;
    model_FCM.U=0;
    model_FCM.sumd=0;
    model_FCM.index=ones(x_num,1);
    model_FCM.fval_loss_list=[];
    return;
end

U=zeros(x_num,classify_num);
center_list=randn(classify_num,vari_num);
iter=0;done=0;
fval_loss_list=zeros(iter_max,1);

% get X_center_dis_sq
sumd_sq=zeros(x_num,classify_num); % X_center_dis_sq
for classify_idx=1:classify_num
    for x_idx=1:x_num
        X_nomlz_C=X_nomlz-center_list(classify_idx,:);
        sumd_sq(:,classify_idx)=sum(X_nomlz_C.*X_nomlz_C,2);
    end
end

while ~done
    % updata classify matrix U
    for classify_idx=1:classify_num
        for x_idx=1:x_num
            U(x_idx,classify_idx)=...
                1/sum((sumd_sq(x_idx,classify_idx)./sumd_sq(x_idx,:)).^(1/(m-1)));
        end
    end
    
    % updata center_list
    center_list_old=center_list;
    for classify_idx=1:classify_num
        center_list(classify_idx,:)=...
            sum((U(:,classify_idx)).^m.*X_nomlz)./...
            sum((U(:,classify_idx)).^m);
    end
    
    % updata X_center_dis_sq
    sumd_sq=zeros(x_num,classify_num);
    for classify_idx=1:classify_num
        for x_idx=1:x_num
            X_nomlz_C=X_nomlz-center_list(classify_idx,:);
            sumd_sq(:,classify_idx)=sum(X_nomlz_C.*X_nomlz_C,2);
        end
    end
    
%     scatter(X_nomlz(:,1),X_nomlz(:,2));
%     line(center_list(:,1),center_list(:,2),'Marker','o','LineStyle','None','Color','r');
    
    % forced interrupt
    if iter > iter_max
        done=1;
    end
    
    % convergence judgment
    if sum(sum(center_list_old-center_list).^2) < torl
        done=1;
    end
    
    iter=iter+1;
    fval_loss_list(iter)=sum(sum(U.^m.*sumd_sq));
end
fval_loss_list(iter+1:end)=[];

% renormalize
center_list=center_list.*stdD_X+aver_X;

% find index
[~,index]=max(U,[],2);

model_FCM.X=X;
model_FCM.aver_X=aver_X;
model_FCM.stdD_X=stdD_X;

model_FCM.center_list=center_list;
model_FCM.U=U;
model_FCM.sumd=sqrt(sumd_sq);
model_FCM.index=index;

model_FCM.fval_loss_list=fval_loss_list;
end
