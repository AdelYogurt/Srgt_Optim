clc;
clear;
close all hidden;

%% generate data

x_num=100;
vari_num=2;
cluster_num=2;
low_bou=zeros(vari_num,1);
up_bou=ones(vari_num,1);
X=[rand(x_num,vari_num)*0.5;0.5+rand(x_num,vari_num)];

%% calculate clustering and draw picture

m=2;

model_cluster=clusterFCM(X,cluster_num,m);
model_cluster=clusterFCMFS(X,cluster_num,m);
model_cluster=clusterMS(X);

displayCluster(model_cluster);
