function [con,coneq] = constrainFunction(x,A,B,Aeq,Beq,cheapcon_function)
% convert A, B, Aeq, Beq to total cheapcon function
% x input is rank vector
%
if nargin < 6
    cheapcon_function = [];
    if nargin < 5
        Beq = [];
        if nargin < 4
            Aeq = [];
            if nargin < 3
                B = [];
                if nargin < 2
                    A = [];
                end
            end
        end
    end
end
con = [];
coneq = [];
if ~isempty(A)
    if isempty(B)
        con = [con,x*A'];
    else
        con = [con,x*A'-B'];
    end
end
if ~isempty(Aeq)
    if isempty(Beq)
        coneq = [coneq,x*Aeq'];
    else
        coneq = [coneq,x*Aeq'-Beq'];
    end
end
if ~isempty(cheapcon_function)
    [lincon,linconeq] = cheapcon_function(x);
    con = [con,lincon];
    coneq = [coneq,linconeq];
end
end
