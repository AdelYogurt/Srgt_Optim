function [fval,con,coneq] = modelFunction(x,object_function,nonlcon_function)
% model function,concertrate fval,con,coneq into one function
%
if nargin < 3 || isempty(nonlcon_function)
    con = [];
    coneq = [];
else
    [con,coneq] = nonlcon_function(x);
end
fval = object_function(x);
end