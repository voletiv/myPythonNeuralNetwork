# Neural Network in Python

An implementation of a Multi-Layer Perceptron, with forward propagation, back propagation using Gradient Descent, training usng Batch or Stochastic Gradient Descent

Use: myNN = MyPyNN(nOfInputDims, nOfHiddenLayers, sizesOfHiddenLayers, nOfOutputDims, alpha, regLambda)
Here, alpha = learning rate of gradient descent, regLambda = regularization parameter

## Example 1

```
from myPyNN import *
X = [0, 0.5, 1]
y = [0, 0.5, 1]
myNN = MyPyNN(1, 1, 1, 1)
```
Input Layer    : 1-dimensional (Eg: X)

1 Hidden Layer : 1-dimensional

Output Layer   : 1-dimensional (Eg. y)

Learning Rate  : 0.05 (default)
``` 
print myNN.predict(0.2)
```


## Example 2
```
X = [[0,0], [1,1]]
y = [0, 1]
myNN = MyPyNN(2, 1, 3, 1, 0.8)
```
Input Layer    : 2-dimensional (Eg: X)

1 Hidden Layer : 3-dimensional

Output Layer   : 1-dimensional (Eg. y)

Learning rate  : 0.8
``` 
print myNN.predict(X)
#myNN.trainUsingGD(X, y, 899)
myNN.trainUsingSGD(X, y, 1000)
print myNN.predict(X)
```

## Example 3

```
X = [[2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6], [7,7,7], [8,8,8], [9,9,9], [10,10,10], [11,11,11]]
y = [.2, .3, .4, .5, .6, .7, .8, .9, 0, .1]
myNN = MyPyNN(3, 3, [10, 10, 5], 0.9, 0.5)
```
Input Layer    : 3-dimensional (Eg: X)

3 Hidden Layers: 10-dimensional, 10-dimensional, 5-dimensional

Output Layer   : 1-dimensional (Eg. y)

Learning rate  : 0.9

Regularization parameter : 0.5
``` 
print myNN.predict(X)
#myNN.trainUsingGD(X, y, 899)
myNN.trainUsingSGD(X, y, 1000)
print myNN.predict(X)
```

## References
- [Machine Learning Mastery's excellent tutorial](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)

- [Mattmazur's example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)

- [Welch Lab's excellent video playlist on neural networks](https://www.youtube.com/playlist?list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU)

