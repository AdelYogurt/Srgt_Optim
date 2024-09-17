function bias=biasE1(x,fail)
if nargin < 2
    fail=1000;
end
theta=1-0.0001*fail;
bias=theta*sum(cos(10*pi*theta*x+0.5*pi*theta+pi),2);
end