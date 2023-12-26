function [fval,gradient]=differ(differ_function,x,fval,step)
% differ function to get gradient
% support matrix output
%
variable_number=length(x);
if nargin < 4
    step=1e-5;
end
if nargin < 3
    fval=differ_function(x);
end
[rank_num,colume_num]=size(fval);
if ((rank_num ~= 1) || (colume_num ~= 1))
    multi_flag=1; % matrix output
else
    multi_flag=0; % numeral output
end

% gradient
if multi_flag
    gradient=zeros(rank_num,colume_num,variable_number);
else
    gradient=zeros(variable_number,1);
end

for variable_index=1:variable_number
    x_forward=x;
    x_forward(variable_index)=x_forward(variable_index)+step;

    if multi_flag
        gradient(:,:,variable_index)=...
            (differ_function(x_forward)-fval)/step;
    else
        gradient(variable_index)=...
            (differ_function(x_forward)-fval)/step;
    end
end

end
