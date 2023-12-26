function [Model_function,Cost,variable_number,low_bou,up_bou,x_best,fval_best]=...
    getBenchmarkMultiFidelity(benchmark_name,benchmark_type)
% obtain problem
%
problem = [benchmark_type,benchmark_name];

% obtain Par
parameter_function=[problem,'().getPar'];
[variable_number,low_bou,up_bou,x_best,fval_best] = eval(parameter_function);

% obtain model
Model_function={
    str2func(['@(x) ',problem,'().calModelHF(x)']);
    str2func(['@(x) ',problem,'().calModelLF(x)']);};
Cost=[1;0.25];

end