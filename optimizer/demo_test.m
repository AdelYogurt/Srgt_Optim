clc;
clear;
close all hidden;

%% test case

% benchmark=Benchmark();

% benchmark_type='single';
% benchmark_name='PVD4';

% benchmark_name_list={'SR7','PVD4','G01','G5MOD','G09','G16','G18'};
% benchmark_name_list={'G07','G09','G19','G23'};
% benchmark_name_list={'G04','G06','G09','PVD4','SR7'};
% benchmark_name_list={'BR','SC','RS','PK','HN','F16'};


%% single run

% [objcon_fcn,vari_num,low_bou,up_bou,obj_fcn,Aineq,Bineq,Aeq,Beq,nonlcon_fcn]=benchmark.get(benchmark_type,benchmark_name);

% NFE_max=500;iter_max=500;obj_torl=1e-6;con_torl=0;

% optimizer=OptimFSRBF(NFE_max,iter_max,obj_torl,con_torl);
% optimizer=OptimKRGCDE(NFE_max,iter_max,obj_torl,con_torl);
% optimizer=OptimKRGEGO(NFE_max,iter_max,obj_torl,con_torl);
% optimizer=OptimPAKMCA(NFE_max,iter_max,obj_torl,con_torl);
% optimizer=OptimRBFCDE(NFE_max,iter_max,obj_torl,con_torl);
% optimizer=OptimSACORS(NFE_max,iter_max,obj_torl,con_torl);
% optimizer=OptimTRARSM(NFE_max,iter_max,obj_torl,con_torl);

% optimizer=OptimSADEKTS(NFE_max,iter_max,obj_torl,con_torl);
% optimizer.dataKTS(load('source\G06.mat').datalib);

% load('optim','optimizer');

% [x_best,obj_best,NFE,output]=optimizer.optimize(objcon_fcn,vari_num,low_bou,up_bou);
% datalib=optimizer.datalib;
% plot(1:length(datalib.Obj),datalib.Obj(datalib.Best_idx),'o-')
% line(1:length(datalib.Obj),datalib.Obj,'Marker','o','Color','g')
% line(1:length(datalib.Vio),datalib.Vio,'Marker','o','Color','r')

% [x_best,obj_best,~,output]=fmincon(obj_fcn,rand(1,vari_num).*(up_bou-low_bou)+low_bou,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,nonlcon_fcn,optimoptions("fmincon",'Algorithm','sqp','Display','iter-detailed'));

%% repeat run

% for benchmark_idx=1:length(benchmark_name_list)
%     benchmark_name=benchmark_name_list{benchmark_idx};
% 
%     [objcon_fcn,vari_num,low_bou,up_bou,...
%         obj_fcn,A,B,Aeq,Beq,nonlcon_fcn]=benchmark.get(benchmark_type,benchmark_name);
%     repeat_num=25;
%     result_obj=zeros(repeat_num,1);
%     result_NFE=zeros(repeat_num,1);
%     NFE_max=200;iter_max=300;obj_torl=1e-6;con_torl=0;
% 
%     for repeat_idx=1:repeat_num
%         % optimizer=OptimFSRBF(NFE_max,iter_max,obj_torl,con_torl);
%         % optimizer=OptimKRGCDE(NFE_max,iter_max,obj_torl,con_torl);
%         % optimizer=OptimKRGEGO(NFE_max,iter_max,obj_torl,con_torl);
%         % optimizer=OptimPAKMCA(NFE_max,iter_max,obj_torl,con_torl);
%         % optimizer=OptimRBFCDE(NFE_max,iter_max,obj_torl,con_torl);
%         % optimizer=OptimSACORS(NFE_max,iter_max,obj_torl,con_torl);
%         % optimizer=OptimTRARSM(NFE_max,iter_max,obj_torl,con_torl);
% 
%         [x_best,obj_best,NFE,output]=optimizer.optimize(objcon_fcn,vari_num,low_bou,up_bou);
% 
%         result_obj(repeat_idx)=obj_best;
%         result_NFE(repeat_idx)=NFE;
% 
%         % datalib=output.datalib;
%         % plot(1:length(datalib.Obj),datalib.Obj(datalib.Best_idx),'o-')
%         % line(1:length(datalib.Obj),datalib.Obj,'Marker','o','Color','g')
%         % line(1:length(datalib.Vio),datalib.Vio,'Marker','o','Color','r')
%     end
% 
%     fprintf('Obj     : lowest=%4.4f,mean=%4.4f,worst=%4.4f,std=%4.4f \n',min(result_obj),mean(result_obj),max(result_obj),std(result_obj));
%     fprintf('NFE     : lowest=%4.4f,mean=%4.4f,worst=%4.4f,std=%4.4f \n',min(result_NFE),mean(result_NFE),max(result_NFE),std(result_NFE));
%     save([benchmark_name,'_FSRBF_',num2str(NFE_max),'.mat']);
% end

%% test case 

% benchmark_type='single';
% benchmark_name_list={'G07','G09','G19','G23'};
% benchmark_error_list={
%     [1,5,100;
%     2,0.5,1000;
%     3,0.5,1000;
%     4,0.5,1000;
%     1,0.5,1000;
%     2,0.5,1000;
%     3,0.5,1000;
%     4,0.5,1000;
%     1,0.5,1000];
%     [1,5,100;
%     2,0.5,1000;
%     3,0.5,1000;
%     4,0.5,1000;
%     1,0.5,1000;]
%     [1,2,200;
%     2,1,1000;
%     3,1,1000;
%     4,1,1000;
%     1,1,1000;
%     2,1,1000;];
%     [1,-2,1000;
%     2,0.05,100;
%     3,0.05,1000;
%     4,0.05,1000;
%     1,0.05,1000;
%     2,0.05,1000;
%     3,0.05,1000;];};
% 
% con_fcn_cheap=[];

%% single run

% Cost=[1,0.01];
% Ratio=[1,4];
% [objcon_fcn_list,vari_num,low_bou,up_bou,...
%         obj_fcn,A,B,Aeq,Beq,nonlcon_fcn]=Benchmark().getMF(benchmark_type,benchmark_name,benchmark_error);
% NFE_max=300;
% optimizer=OptSACOAMS(NFE_max,300,1e-6,0);
% [x_best,obj_best,NFE,output]=optimizer.optimize(objcon_fcn_list,Cost,Ratio,vari_num,low_bou,up_bou,con_fcn_cheap);
% result_x_best=output.result_x_best;
% result_obj_best=output.result_obj_best;
% 
% datalib_HF=output.datalib_HF;
% plot(datalib.Obj(datalib_HF.result_best_idx),'o-')
% line(1:length(datalib_HF.Obj),datalib_HF.Obj,'Marker','o','Color','g')
% line(1:length(datalib_HF.Vio),datalib_HF.Vio,'Marker','o','Color','r')

%% repeat run

% for benchmark_idx=1:length(benchmark_name_list)
%     benchmark_name=benchmark_name_list{benchmark_idx};
%     benchmark_error=benchmark_error_list{benchmark_idx};
% 
%     [objcon_fcn_list,vari_num,low_bou,up_bou,...
%         obj_fcn,A,B,Aeq,Beq,nonlcon_fcn]=Benchmark().getMF(benchmark_type,benchmark_name,benchmark_error);
% 
%     Cost=[1,0.01];
%     Ratio=[1,4];
%     repeat_number=25;
%     result_obj=zeros(repeat_number,1);
%     result_NFE=zeros(repeat_number,1);
%     NFE_max=200;
%     for repeat_idx=1:repeat_number
%         optimizer=OptSACOAMS(NFE_max,300,1e-6,0);
%         [x_best,obj_best,NFE,output]=optimizer.optimize(objcon_fcn_list,Cost,Ratio,vari_num,low_bou,up_bou,con_fcn_cheap);
% 
%         result_obj(repeat_idx)=obj_best;
%         result_NFE(repeat_idx)=NFE;
%         datalib_HF=output.datalib_HF;
% 
%         plot(datalib_HF.Obj(datalib_HF.result_best_idx),'o-')
%         line(1:length(datalib_HF.Obj),datalib_HF.Obj,'Marker','o','Color','g')
%         line(1:length(datalib_HF.Vio),datalib_HF.Vio,'Marker','o','Color','r')
%     end
% 
%     fprintf('Obj     : lowest=%4.4f,mean=%4.4f,worst=%4.4f,std=%4.4f \n',min(result_obj),mean(result_obj),max(result_obj),std(result_obj));
%     fprintf('NFE     : lowest=%4.4f,mean=%4.4f,worst=%4.4f,std=%4.4f \n',min(result_NFE),mean(result_NFE),max(result_NFE),std(result_NFE));
%     save([benchmark_name,'_SACO_AMS_',num2str(NFE_max),'.mat']);
% end
