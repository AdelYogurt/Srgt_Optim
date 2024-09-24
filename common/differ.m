function [fval,grad] = differ(differ_fcn,x,fval,step)
% differ function to get gradient
% support matrix output
%
vari_num = length(x);
if nargin < 4 || isempty(step)
    step = 1e-5;
end
if nargin < 3 || isempty(fval)
    fval = differ_fcn(x);
end
[rank_num,colu_num] = size(fval);
if ((rank_num ~= 1) || (colu_num ~= 1))
    multi_flag = 1; % matrix output
else
    multi_flag = 0; % numeral output
end

% gradient
if multi_flag
    grad = zeros(rank_num,colu_num,vari_num);
else
    grad = zeros(vari_num,1);
end

for vari_idx = 1:vari_num
    x_for = x;
    x_for(vari_idx) = x_for(vari_idx)+step;

    if multi_flag
        grad(:,:,vari_idx) = (differ_fcn(x_for)-fval)/step;
    else
        grad(vari_idx) = (differ_fcn(x_for)-fval)/step;
    end
end

end
