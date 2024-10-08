clc;
clear;
close all;

%% path

% addpath ../benchmark/
% addpath ../common/
% addpath ../LHD/
% addpath ../ML/
% addpath ../surrogate/

%% single run

% prob='G07';
% [objcon_fcn,vari_num,low_bou,up_bou]=getProb(prob);

% NFE_max=200;iter_max=220;obj_tol=1e-6;con_tol=0;

% optimizer=OptimFSRBF(NFE_max,iter_max,obj_tol,con_tol);
% optimizer=OptimGLoSADE(NFE_max,iter_max,obj_tol,con_tol);
% optimizer=OptimKRGCDE(NFE_max,iter_max,obj_tol,con_tol);
% optimizer=OptimSKO(NFE_max,iter_max,obj_tol,con_tol);
% optimizer=OptimPAKMCA(NFE_max,iter_max,obj_tol,con_tol);
% optimizer=OptimRBFCDE(NFE_max,iter_max,obj_tol,con_tol);
% optimizer=OptimSACORS(NFE_max,iter_max,obj_tol,con_tol);
% optimizer=OptimTRARSM(NFE_max,iter_max,obj_tol,con_tol);

% optimizer=OptimSADEKTS(NFE_max,iter_max,obj_tol,con_tol);
% optimizer.dataKTS(load('source\G06.mat').datalib);

% load('optim','optimizer');

% optimizer.FLAG_CONV_JUDGE=true;
% optimizer.FLAG_DRAW_FIGURE=true;
% optimizer.datalib_filestr='lib.mat';
% optimizer.dataoptim_filestr='optim.mat';

% [x_best,obj_best,NFE,output]=optimizer.optimize(objcon_fcn,vari_num,low_bou,up_bou);
% datalib=optimizer.datalib;
% plot(1:length(datalib.Obj),datalib.Obj(datalib.Best_idx),'o-')
% line(1:length(datalib.Obj),datalib.Obj,'Marker','o','Color','g')
% line(1:length(datalib.Vio),datalib.Vio,'Marker','o','Color','r')

%% repeat run

% prob_list={'SR7','PVD4','G01','G5MOD','G09','G16','G18'};
% 
% for prob_idx=1:length(prob_list)
%     prob=prob_list{prob_idx};
%     [objcon_fcn,vari_num,low_bou,up_bou]=getProb(prob);
% 
%     repeat_num=25;
%     result_obj=zeros(repeat_num,1);
%     result_NFE=zeros(repeat_num,1);
%     NFE_max=200;iter_max=250;obj_tol=1e-6;con_tol=0;
% 
%     for repeat_idx=1:repeat_num
%         % optimizer=OptimFSRBF(NFE_max,iter_max,obj_tol,con_tol);
%         % optimizer=OptimKRGCDE(NFE_max,iter_max,obj_tol,con_tol);
%         % optimizer=OptimSKO(NFE_max,iter_max,obj_tol,con_tol);
%         % optimizer=OptimPAKMCA(NFE_max,iter_max,obj_tol,con_tol);
%         % optimizer=OptimRBFCDE(NFE_max,iter_max,obj_tol,con_tol);
%         % optimizer=OptimSACORS(NFE_max,iter_max,obj_tol,con_tol);
%         % optimizer=OptimTRARSM(NFE_max,iter_max,obj_tol,con_tol);
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
%     save([prob,'_optim_',num2str(NFE_max),'.mat']);
% end
