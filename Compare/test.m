clc;
clear;
close all hidden;

x_list=getLatinHypercube(10,2);
scatter(x_list(:,1),x_list(:,2));
