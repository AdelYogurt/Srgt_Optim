clc;
clear;
close all hidden;

func_low=@(x) lowFunction(x);
func_high=@(x) highFunction(x);

func_low_bou=@(x) 0.45+sin(2.2*pi*x)/2.5;
func_high_bou=@(x) 0.5+sin(2.5*pi*x)/3;

variable_number=2;

% generate X
x_number=45;
X=zeros(x_number,2);
for x_index=1:30
    x=x_index/30-1/60;
    y=func_low_bou(x)+(rand()-0.5)*0.3;
    if y < 0
        y=0;
    end
    if y > 1
        y=1;
    end
    X(x_index,:)=[x,y];
end
X(31:45,:)=lhsdesign(15,variable_number);
% get class
Y=zeros(x_number,1);
for x_index=1:x_number
    Y(x_index)=func_low(X(x_index,:)');
end

% generate X_H
x_number_H=10;
X_H=zeros(x_number_H,2);
for x_index=1:x_number_H
    x=x_index/x_number_H-1/x_number_H/2;
    X_H(x_index,:)=[x,func_high_bou(x)+(rand()-0.5)*0.3];
end
% get class
Y_H=zeros(x_number_H,1);
for x_index=1:x_number_H
    Y_H(x_index)=func_high(X_H(x_index,:)');
end

low_bou=[0;0];
up_bou=[1;1];

GMFM_model=classifyGaussMultiFidelity...
    (X,Y,X_H,Y_H,low_bou,up_bou);
GMFM_predict_function=GMFM_model.predict_function;

GM_model=classifyGauss...
    (X_H,Y_H,low_bou,up_bou);
GM_predict_function=GM_model.predict_function;

add_point_number=25;
GMFM_data=zeros(add_point_number,1);
GM_data=zeros(add_point_number,1);

real_function=func_high;

for point_index=1:add_point_number
    correct_rate=evluatePredictFunction...
        (GMFM_predict_function,real_function,variable_number);
    GMFM_data(point_index)=correct_rate;
    correct_rate=evluatePredictFunction...
        (GM_predict_function,real_function,variable_number);
    GM_data(point_index)=correct_rate;
    
    save();
    
    x_initial=rand(variable_number,1).*(up_bou-low_bou)+low_bou;
    x_add=getPointActiveLearn(GMFM_predict_function,x_initial,low_bou,up_bou);
    
    y_add=func_high(x_add);
    
    X_H=[X_H;x_add'];
    Y_H=[Y_H;y_add];
    
    GMFM_model=classifyGaussMultiFidelity...
        (X,Y,X_H,Y_H,low_bou,up_bou);
    GMFM_predict_function=GMFM_model.predict_function;
    
    GM_model=classifyGauss...
        (X_H,Y_H,low_bou,up_bou);
    GM_predict_function=GM_model.predict_function;
end

function x=getPointActiveLearn(predict_function,x_initial,low_bou,up_bou)
object_function=@(x) objectFunctionActive(predict_function,x);
fmincon_options = optimoptions('fmincon','Display','none','Algorithm','sqp');
x=fmincon(object_function,x_initial,...
    [],[],[],[],low_bou,up_bou,[],fmincon_options);

    function fval=objectFunctionActive(predict_function,x)
       [~,fval,~]=predict_function(x);
       fval=(fval-0.5)^2;
    end
end

function correct_rate=evluatePredictFunction...
    (predict_function,real_function,variable_number)
% test predict function
%
test_number=100;
X_test=lhsdesign(test_number,variable_number);

correct_number=0;
for x_index=1:test_number
    x=X_test(x_index,:)';
    if predict_function(x)==real_function(x)
        correct_number=correct_number+1;
    end
end

correct_rate=correct_number/test_number;
end

function fval=lowFunction(x)
if 0.45+sin(2.2*pi*x(1))/2.5-x(2) > 0
    fval=1;
else
    fval=-1;
end
end
function fval=highFunction(x)
if 0.5+sin(2.5*pi*x(1))/3-x(2) > 0
    fval=1;
else
    fval=-1;
end
end