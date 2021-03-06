# pr_func

Implementation of a 'function' container class containing a numpy array and knowledge about its variables.
It was developed to simplify calculus with multi-dimensional tensors/matrices. For example, when modelling discrete probability distributions of multiple variables in terms of multi-dimensional arrays, then multiplication and addition requires to specify the axes that correspond to the same dimensions. When using the class `func` defined in `pr_func.py` we only have to specify the variables for each distribution, and then do the calculations naively by using the standard operators.

## Simple example

Here is an example of how to calculate the conditional probability `p(x|y,z)` from given distributions `p(z|x,y)` and `p(x,y)`, i.e. `p(x|y,z) = p(z|x,y)*p(x,y)/p(y,z)`, where `p(y,z) = sum_x p(z|x,y)*p(x,y)`.

The __numpy variant__ using the powerful `einsum` method would be:
```python
X,Y,Z = 10,15,20             # fixing the possible number of values for each of the random variables
pxy = np.ones((X,Y))/(X*Y)   # uniform distributions for simplicty                  
pz_xy = np.ones((X,Y,Z))/Z   # normalizing a non-uniform distribution would require another sum/einsum
px_yz = np.einsum('ijk,ij,jk->ijk', pz_xy, pxy, 1.0/np.einsum('ijk,ij->jk',pz_xy, pxy)  
```

And here is the __pr_func variant__:
```python
pr.set_dims([('x',10),('y',15),('z',20)])              # setting up the dimensions
pxy = pr.func(vars=['x','y'], val='unif').normalize()  # an instance of `func` depending on x and y
pz_xy = pr.func(vars=['x','y','z'], val='unif').normalize(['z'])    # ... on x,y,z, and normalizing
px_yz = pz_xy*pxy/pr.sum(pz_xy*pxy,['x'])              # simple multiplication, division, and sums
```
As we can see, the setup requires to define the dimensions, but then the distributions can be multiplied like numbers, since they are instances of the `pr_func` class knowing which dimensions correspond to each other. Note that, since we implemented a `normalize()` method, the last row of the `pr_func` variant could have been even simpler:
```python
px_yz = (pz_xy*pxy).normalize(['x'])
```


Examples of how to use `pr_func` for efficiently implementing Blahut-Arimoto type algorithms can be found [here](https://github.com/sgttwld/blahut-arimoto).


## Overview

### Setup
```python
import pr_func as pr
pr.set_dims([('x',10),('y',15),('z',5)])    # setting up the dimensions
```

### Defining instances
```python
F = pr.func('f(x,y)', val='rnd')            # an instance depending on x and y with random values
G = pr.func('f(z)', val=np.array([1,1,2]))  # an instance depending on z with given values
H = 5*pr.func(vars=['x','z'], val='unif')   # an instance depending on x and z with the same value for each entry
```

### Basic calculus
```python
F*G, F/G, 2*H, F+H, 1+G-F, ...              # results of basic operations are also func instances  
```

### Summation and normalization
```python
pr.sum(F)                                   # summation over all variables of F
pr.sum(['z'],G*H)                           # summation over z
pr.sum(F+H,['z'])                           # summation over all variables of F+H except z
F.normalize()                               # normalization with respect to all variables
(F+G).normalize(['y','z'])                  # normalization with respect to y and z
```

### Instance properties
```python
F.val                                       # numpy array, here: F.val = np.random.rand(10,15)
F.vars                                      # list of the variables, here: F.vars = ['x','y']
F.r                                         # positions of the vars in dims, here: F.r = [0,1]
```

### Evaluation/Slicing
```python
f = F.eval('x',4)                         # f(y) = F(x=4,y)
```
