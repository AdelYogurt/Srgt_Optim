clc;
clear;
close all hidden;

sample_num=6;
vari_num=2;

%% lhd sample

% [X_sample,dist_min_nomlz]=lhdESLHS(sample_num,vari_num);
% [X_sample,dist_min_nomlz]=lhdO(sample_num,vari_num);
% [X_sample,dist_min_nomlz]=lhdPS(sample_num,vari_num);
% [X_sample,dist_min_nomlz]=lhdSLE(sample_num,vari_num);
% [X_sample,dist_min_nomlz]=lhdSSLE(sample_num,vari_num);

% scatter(X_sample(:,1),X_sample(:,2))

%% Nlhd sample

% X_base=lhsdesign(sample_num*2,vari_num);
% [X_sample,dist_min_nomlz]=lhdNSLE(X_base,sample_num,vari_num);
% 
% scatter(X_base(:,1),X_base(:,2));hold on;
% scatter(X_sample(:,1),X_sample(:,2));hold off;