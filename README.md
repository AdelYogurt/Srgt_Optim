# Surrogate Model and Optimize Algorithm

## dir

`common/`: Benchmark and common use debug function.  
`LHD/`: Latin hypercubic design method.  
`ML/`: Light machine learning algorithm.  
`optimizer/`: Surrogate base optimization algorithm.  
`surrogate/`: Surrogate model.  

## about optimizer

Surrogate base optimization algorithm(SBOA) is used to solve expensive black-box problem.  
Expensive black-box problem is time consume problem. The fewer model calls, the better.  
During optimize, number of function to evaluate(NFE) is important.

## how to use optimizer

Optimizer basic input is `objcon_fcn`, `vari_num`, `low_bou`, `up_bou`.  
Function `objcon_fcn` should return `[obj, con, coneq]`  
 