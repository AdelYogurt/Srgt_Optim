function objcon_fcn=getLFProb(objcon_fcn,bias_define)
% get low fidelity model by adding error bias to high fidelity
%
objcon_fcn=@(x)warpBias(objcon_fcn,x,bias_define);
end

function [obj,con,coneq]=warpBias(objcon_fcn,x,bias_define)
% wrap bias to function
%
[obj,con,coneq]=objcon_fcn(x);

bias_param=bias_define{1}; % obj
obj=addbias(x,obj,bias_param);

if ~isempty(con)
    if length(bias_define) < 2
        error('getLF.warpBias: bias_define is not enough');
    end
    bias_param=bias_define{2}; % con
    con=addbias(x,con,bias_param);
end

if ~isempty(coneq)
    if length(bias_define) < 3
        error('getLF.warpBias: bias_define is not enough');
    end
    bias_param=bias_define{3}; % coneq
    coneq=addbias(x,coneq,bias_param);
end

end

function value=addbias(x,value,bias_param)
if length(value) ~= size(bias_param,1)
    error('getLFProb.warpBias.addbias: function output do not match bias');
end

for idx=1:length(value)
    bias=bias_param(idx,:);
    switch bias(1)
        case 1
            err=bias(2)*biasE1(x,bias(3));
        case 2
            err=bias(2)*biasE2(x,bias(3));
        case 3
            err=bias(2)*biasE3(x,bias(3));
        case 4
            err=bias(2)*biasE4(x,bias(3));
    end
    value(idx)=value(idx)+err;
end
end