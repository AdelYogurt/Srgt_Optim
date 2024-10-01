function fcn=getLFFFcn(fcn,bias_param)
% add bias error to function
%
fcn=@(x)warpBias(fcn,x,bias_param);
end

function value=warpBias(fcn,x,bias_param)
% wrap bias to function
%
value=fcn(x);
if length(value) ~= size(bias_param,1)
    error('addBias.warpBias: function output do not match bias')
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
