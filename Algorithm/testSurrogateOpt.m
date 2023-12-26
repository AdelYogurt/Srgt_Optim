clc;
clear;
close all hidden;

benchmark = BenchmarkFunction();

benchmark_type = 'single';
benchmark_name = 'PK';
% benchmark_name = 'EP20';
% benchmark_name = 'Forrester';
% benchmark_name = 'PVD4';
% benchmark_name = 'G01';
% benchmark_name = 'G06';

[model_function,variable_number,low_bou,up_bou,...
    object_function,A,B,Aeq,Beq,nonlcon_function] = benchmark.getBenchmarkFunction(benchmark_type,benchmark_name);

repeat_number = 10;
result_fval = zeros(repeat_number, 1);
max_NFE = 30;

objconstr = @(x) modelFunction(x,object_function,nonlcon_function);

for repeat_index = 1:repeat_number
    x_initial_list = lhsdesign(2*variable_number,variable_number,'iterations',100,'criterion','maximin').*(up_bou-low_bou)+low_bou;
    surrogateopt_option = optimoptions('surrogateopt', 'MaxFunctionEvaluations', max_NFE, 'Display', 'none','PlotFcn','surrogateoptplot','InitialPoints',x_initial_list);

    [x_best, fval_best, exitflag, output] = surrogateopt...
        (objconstr, low_bou, up_bou, surrogateopt_option);

    result_fval(repeat_index) = fval_best;
end

fprintf('Fval     : lowest = %4.4f, mean = %4.4f, worst = %4.4f, std = %4.4f \n', min(result_fval), mean(result_fval), max(result_fval), std(result_fval));
object_function_name=char(object_function);
save([object_function_name(15:end-3),'_',num2str(max_NFE),'_surrogatopt','.mat']);


function objcon = modelFunction(x,object_function,nonlcon_function)
fval = object_function(x);
if isempty(nonlcon_function)
    objcon = fval;
else
    [con,coneq] = nonlcon_function(x);
    objcon.Fval = fval;
    objcon.Ineq = con;
end
end
