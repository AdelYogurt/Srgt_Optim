function model_FCMFS=clusterFCMFS(X,classify_num,m,kernel_fcn)
% generate fuzzy cluster model with feature space
% kernal function recommend kernal_function=@(sq) exp(-x_sq/vari);
% X is x_number x vari_num matrix
% center_list is classify_num x vari_num matrix
%
if nargin < 4
    kernel_fcn=[];
    if nargin < 3
        m=[];
    end
end

iter_max=100;
torl=1e-6;

% normaliz data
[x_num,vari_num]=size(X);
aver_X=mean(X);
stdD_X=std(X);stdD_X(stdD_X == 0)=1;
X_nomlz=(X-aver_X)./stdD_X;

if isempty(m)
    m=2;
end

% default kernal function
if isempty(kernel_fcn)
    sigma=1/vari_num;
    kernel_fcn=@(dis_sq) exp(-dis_sq*sigma);
end

% if x_number equal 1, clustering cannot done
if x_num == 1
    center_list=X;

    model_FCMFS.X=X;
    model_FCMFS.aver_X=aver_X;
    model_FCMFS.stdD_X=stdD_X;

    model_FCMFS.center_list=center_list;
    model_FCMFS.U=0;
    model_FCMFS.sumd=0;
    model_FCMFS.index=ones(x_num,1);

    model_FCMFS.fval_loss_list=[];
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
                1/sum(((2-2*kernel_fcn(sumd_sq(x_idx,classify_idx)))./...
                (2-2*kernel_fcn(sumd_sq(x_idx,:)))).^(1/(m-1)));
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
    fval_loss_list(iter)=sum(sum(U.^m.*(2-2*kernel_fcn(sumd_sq))));
end
fval_loss_list(iter+1:end)=[];

% renormalize
center_list=center_list.*stdD_X+aver_X;

% find index
[~,index]=max(U,[],2);

model_FCMFS.X=X;
model_FCMFS.aver_X=aver_X;
model_FCMFS.stdD_X=stdD_X;

model_FCMFS.center_list=center_list;
model_FCMFS.U=U;
model_FCMFS.sumd=sqrt(sumd_sq);
model_FCMFS.index=index;

model_FCMFS.fval_loss_list=fval_loss_list;
end
