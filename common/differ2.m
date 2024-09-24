function [fval,gradient,hessian]=differ2(differ_function,x,fval,step)
% differ function to get gradient and hessian
%
variable_number=length(x);
if nargin < 4
    step=1e-5;
end
if nargin < 3
    fval=differ_function(x);
end
fval_list=zeros(variable_number,2); % backward is 1, forward is 2
gradient=zeros(variable_number,1);
hessian=zeros(variable_number);

% fval and gradient
for variable_index__=1:variable_number
    x_forward=x;
    x_backward=x;
    x_backward(variable_index__)=x_backward(variable_index__)-step;
    fval_list(variable_index__,1)=differ_function(x_backward);
    
    x_forward(variable_index__)=x_forward(variable_index__)+step;
    fval_list(variable_index__,2)=differ_function(x_forward);
    
    gradient(variable_index__)=...
        (fval_list(variable_index__,2)-fval_list(variable_index__,1))/2/step;
end

% hessian
for variable_index__=1:variable_number
    hessian(variable_index__,variable_index__)=...
        (fval_list(variable_index__,2)-2*fval+fval_list(variable_index__,1))/step/step;
    for variable_index_next__=variable_index__+1:variable_number
        x_for_for=x;
        x_for_for(variable_index__)=x_for_for(variable_index__)+step;
        x_for_for(variable_index_next__)=x_for_for(variable_index_next__)+step;
        
        hessian(variable_index__,variable_index_next__)=(...
            differ_function(x_for_for)-...
            fval_list(variable_index__,2)-fval_list(variable_index_next__,2)+...
            fval...
            )/step/step;
        hessian(variable_index_next__,variable_index__)=...
            hessian(variable_index__,variable_index_next__);
    end
end
end
