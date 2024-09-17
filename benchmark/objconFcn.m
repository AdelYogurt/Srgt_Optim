function [obj,con,coneq]=objconFcn(x,obj_fcn,con_fcn)
% objective and constraint function, concertrate obj,con,coneq into one function
%
if nargin < 3 || isempty(con_fcn)
    con=[];
    coneq=[];
else
    [con,coneq]=con_fcn(x);
end
obj=obj_fcn(x);
end