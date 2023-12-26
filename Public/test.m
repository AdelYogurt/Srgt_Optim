clc;
clear
close all hidden;

algorithm_list={'NSGAII','CSEA','PBRVEA','SAPSMO','True PF'};
color_list={'r','b','g','m'};
marker_list={'>','s','d','p'};

% IGD=zeros(4,10,3);
% for problem_index=1:3
%     load(['ZDT',num2str(problem_index),'.mat']);
%     for algorithm_index=1:4
%         load(['ZDT',num2str(problem_index),'_',algorithm_list{algorithm_index},'.mat']);
%         
%         for repeat_index=1:10
%             fval_pareto=fval_pareto_list{repeat_index};
%             IGD(algorithm_index,repeat_index,problem_index) = mean(min(pdist2(fval_pareto,real_pareto),[],2));
%         end
%     end
% end

% f=figure();
% ax=gca;
% boxplot(ax,IGD(:,:,3)');
% ax.set('YScale','log');
% ax.set('Ylim',[0.001,10]);
% ax.set('XTickLabel',{'NSGA-II','CSEA','PB-RVEA','MSAPSO'})
% ax.set('YGrid','On')
% title('ZDT3 Problem');
% ylabel('IGD')
% saveas(f,'ZDT3.png','png');

for problem_index=1:3
    f=figure();
    load(['ZDT',num2str(problem_index),'.mat']);
    for algorithm_index=1:4
        load(['ZDT',num2str(problem_index),'_',algorithm_list{algorithm_index},'.mat']);
        
        line(fval_pareto(:,1),fval_pareto(:,2),...
            'Marker',marker_list{algorithm_index},...
            'LineStyle','none','Color',color_list{algorithm_index},...
            'MarkerFaceColor',color_list{algorithm_index},...
            'MarkerSize',4);
    end
    if problem_index == 3
       line(real_pareto(1:9,1),real_pareto(1:9,2));
       line(real_pareto(10:17,1),real_pareto(10:17,2));
       line(real_pareto(18:22,1),real_pareto(18:22,2));
       line(real_pareto(23:26,1),real_pareto(23:26,2));
       line(real_pareto(27:end,1),real_pareto(27:end,2));
    else
        line(real_pareto(:,1),real_pareto(:,2));
    end
    legend(algorithm_list);
    title(['ZDT',num2str(problem_index),' Pareto Front']);
    xlabel('f1')
    ylabel('f2')
    
    saveas(f,['ZDT',num2str(problem_index),'_PF.png'],'png');
end

function [pareto_index_list,feasible_index_list]=getParetoFront...
    (fval_nomlz_list,con_nomlz_list,coneq_nomlz_list,...
    pareto_torlance)
% distinguish pareto front of exist point
% use initial pareto definition(contrain compare first and fval compare)
%
if nargin < 4
    pareto_torlance=0;
end

pareto_index_list=[];% index of filter point list
feasible_index_list=[];% feasible point list

% select no domain filter
for fval_index=1:size(fval_nomlz_list,1)
    con_nomlz=[];
    if ~isempty(con_nomlz_list)
        con_nomlz=max(con_nomlz_list(fval_index,:));
    end
    coneq_nomlz=[];
    if ~isempty(coneq_nomlz_list)
        coneq_nomlz=max(abs(coneq_nomlz_list(fval_index,:)));
    end
    con_nomlz_max=max([con_nomlz;coneq_nomlz;0]); % con is large than 0 or 0
    
    if (con_nomlz_max <= pareto_torlance)
        feasible_index_list=[feasible_index_list;fval_index];
    end
    
    % notice con_nomlz_max is greater than or equal to zero
    % notice con_nomlz_pareto_max is greater than or equal to zero
    pareto_index=1;
    add_filter_flag=1;
    while pareto_index <= length(pareto_index_list)
        % compare x with exit pareto front point
        x_pareto_index=pareto_index_list(pareto_index,:);
        
        % contain constraint of x_filter
        con_pareto_nomlz=[];
        if ~isempty(con_nomlz_list)
            con_pareto_nomlz=max(con_nomlz_list(x_pareto_index,:));
        end
        coneq_pareto_nomlz=[];
        if ~isempty(coneq_nomlz_list)
            coneq_pareto_nomlz=max(coneq_nomlz_list(x_pareto_index,:));
        end
        con_nomlz_pareto_max=max([con_pareto_nomlz;coneq_pareto_nomlz;0]);
        
        % compare x with x_pareto
        % if this x is domain by pareto point, reject it
        if con_nomlz_max > con_nomlz_pareto_max
            % x constrain is large than pareto
            add_filter_flag=0;
            break;
        elseif con_nomlz_max <= pareto_torlance && con_nomlz_pareto_max <= pareto_torlance
            % both is feasiable point
            judge=fval_nomlz_list(fval_index,:)+pareto_torlance >= fval_nomlz_list(x_pareto_index,:);
            if ~sum(~judge)
                add_filter_flag=0;
                break;
            end
        end
        
        % if better or equal than exit pareto point, reject pareto point
        if con_nomlz_pareto_max > con_nomlz_max
            % pareto constrain is large than x
            pareto_index_list(pareto_index)=[];
            pareto_index=pareto_index-1;
        elseif con_nomlz_max <= pareto_torlance && con_nomlz_pareto_max <= pareto_torlance
            judge=fval_nomlz_list(fval_index,:) <= fval_nomlz_list(x_pareto_index,:)+pareto_torlance;
            if ~sum(~judge)
                pareto_index_list(pareto_index)=[];
                pareto_index=pareto_index-1;
            end
        end
        
        pareto_index=pareto_index+1;
    end
    
    % add into pareto list if possible
    if add_filter_flag
        pareto_index_list=[pareto_index_list;fval_index];
    end
end
end