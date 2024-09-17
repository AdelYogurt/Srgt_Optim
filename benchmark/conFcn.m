function [con,coneq]=conFcn(x,A,b,Aeq,beq,nonlcon_fcn)
% convert A,b,Aeq,beq,nonlcon_fcn to constraints function
% x input is rank vector
%
if nargin < 6
    nonlcon_fcn=[];
    if nargin < 5
        beq=[];
        if nargin < 4
            Aeq=[];
            if nargin < 3
                b=[];
                if nargin < 2
                    A=[];
                end
            end
        end
    end
end
con=[];
coneq=[];
if ~isempty(A)
    if isempty(b)
        con=[con,x*A'];
    else
        con=[con,x*A'-b'];
    end
end
if ~isempty(Aeq)
    if isempty(beq)
        coneq=[coneq,x*Aeq'];
    else
        coneq=[coneq,x*Aeq'-beq'];
    end
end
if ~isempty(nonlcon_fcn)
    [nonlcon,nonlconeq]=nonlcon_fcn(x);
    con=[con,nonlcon];
    coneq=[coneq,nonlconeq];
end
end