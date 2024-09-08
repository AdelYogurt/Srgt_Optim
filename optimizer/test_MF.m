clc;
clear;
close all hidden;

% objcon_fcn_2=@(x) deal(-sin(x)-exp(x/100)+10.3+0.03*(x-3).^2,[],[]);
% objcon_fcn_1=@(x) deal(-sin(x)-exp(x/100)+10,[],[]);
% objcon_fcn_list={objcon_fcn_2,objcon_fcn_1};
% vari_num=1;
% low_bou=0;up_bou=10;
% cost_list=[0.25,1];

obj_fcn_1=@(x) (4-2.1*x(1)^2+x(1)^4/3)*x(1)^2+x(1)*x(2)+(-4+4*x(2)^2)*x(2)^2;
obj_fcn_5=@(x) 1.5*obj_fcn_1(x+[0.1,-0.1])-0.5;
obj_fcn_2=@(x) obj_fcn_1(x)+0.25*(obj_fcn_5(x)-obj_fcn_1(x));
obj_fcn_3=@(x) obj_fcn_1(x)+0.5*(obj_fcn_5(x)-obj_fcn_1(x));
obj_fcn_4=@(x) obj_fcn_1(x)+0.75*(obj_fcn_5(x)-obj_fcn_1(x));

objcon_fcn_1=@(x) deal(obj_fcn_1(x),[],[]);
objcon_fcn_5=@(x) deal(obj_fcn_5(x),[],[]);
objcon_fcn_2=@(x) deal(obj_fcn_2(x),[],[]);
objcon_fcn_3=@(x) deal(obj_fcn_3(x),[],[]);
objcon_fcn_4=@(x) deal(obj_fcn_4(x),[],[]);

objcon_fcn_list={objcon_fcn_2,objcon_fcn_1};
vari_num=2;
low_bou=[-2,-1];up_bou=[2,1];
cost_list=[0.5,1];
NFE_max=50;iter_max=300;obj_torl=1e-6;con_torl=0;

optimizer=OptimMFSKO(NFE_max,iter_max,obj_torl,con_torl);

% X_LF=[0;2;4;6;8;10];
% X_HF=[3.5;6.5];
% optimizer.X_init={X_LF;X_HF};

[x_best,obj_best,NFE,output]=optimizer.optimize(objcon_fcn_list,vari_num,low_bou,up_bou,cost_list);
datalib=optimizer.datalib;
dataoptim=optimizer.dataoptim;
plot(1:length(datalib{end}.Obj),datalib{end}.Obj(datalib{end}.Best_idx),'o-')
line(1:length(datalib{end}.Obj),datalib{end}.Obj,'Marker','o','Color','g')
line(1:length(datalib{end}.Vio),datalib{end}.Vio,'Marker','o','Color','r')

% objcon_fcn=@(x) deal(-sin(x)-exp(x/100)+10,[],[]);
% vari_num=1;
% low_bou=0;up_bou=10;

% objcon_fcn=@(x) deal((4-2.1*x(1)^2+x(1)^4/3)*x(1)^2+x(1)*x(2)+(-4+4*x(2)^2)*x(2)^2,[],[]);
% vari_num=2;
% low_bou=[-2,-1];up_bou=[2,1];
% NFE_max=50;iter_max=300;obj_torl=1e-6;con_torl=0;
% 
% optimizer=OptimSKO(NFE_max,iter_max,obj_torl,con_torl);
% 
% [x_best,obj_best,NFE,output]=optimizer.optimize(objcon_fcn,vari_num,low_bou,up_bou);
% datalib=optimizer.datalib;
% plot(1:length(datalib.Obj),datalib.Obj(datalib.Best_idx),'o-')
% line(1:length(datalib.Obj),datalib.Obj,'Marker','o','Color','g')
% line(1:length(datalib.Vio),datalib.Vio,'Marker','o','Color','r')

