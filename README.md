# pr_func
Implementation of a 'function' container class containing a numpy array and knowledge about its variables.
It was developed to simplify calculus with multi-dimensional tensors/matrices. For example, when modelling discrete probability distributions of multiple variables in terms of multi-dimensional arrays, then multiplication and addition requires to specify the axes that correspond to the same dimensions. When using the class `func` defined in `pr_func.py` we only have to specify the variables for each distribution, and then do the calculations naively by using the standard operators.


