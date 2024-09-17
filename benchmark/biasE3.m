function bias=biasE3(x,fail)
if nargin < 2
    fail=1000;
end
if fail < 1000
    theta=1-0.0002*fail;
elseif fail < 2000
    theta=0.8;
elseif fail < 3000
    theta=1.2-0.0002*fail;
elseif fail < 4000
    theta=0.6;
elseif fail < 5000
    theta=1.4-0.0002*fail;
elseif fail < 6000
    theta=0.4;
elseif fail < 7000
    theta=1.6-0.0002*fail;
elseif fail < 8000
    theta=0.2;
elseif fail < 9000
    theta=1.8-0.0002*fail;
else
    theta=0;
end

bias=theta*sum(cos(10*pi*theta*x+0.5*pi*theta+pi),2);
end