# Surrogate_Model_Algorithm

Algorithm/: Surrogate base optimization algorithm.
Common/: Benchmark and common use debug function.
LHD/: Latin hypercubic design method.
Machine_Learning/: Light machine learning algorithm.
Surrogate_Model/: Surrogate model.

Surrogate base optimization algorithm(SBOA) is used to solve expensive black-box problem.
Expensive black-box problem is time consume problem. The fewer model calls, the better.
During optimize, number of function to evaluate(NFE) is importance.

Surrogate base optimization algorithm basic input is model_function, variable_number, low_bou, up_bou.

For object_function and nonlcon_function
Difference from convention optimization algorithm, object_function and nonlcon_function(expensive)
should be packed into model_function by modelFunction.m in order to desearce NFE.
For example:
model_function = @(x) modelFunction(x,@(x) object_function,@(x) nonlcon_function)

For model_function:
The model_function should output [fval, con(can be []), coneq(can be [])]

 