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

## Requirements for interactive tutorial (myPyNN.ipynb)

I ran this in OS X, after installing brew for command-line use, and pip for python-related stuff.

### Python

I designed the tutorial on Python 2.7, can be run on Python 3 as well.

### Packages

- numpy
- matplotlib
- ipywidgets

### Jupyter

The tutorial is an iPython notebook. It is designed and meant to run in Jupyter. To install Jupyter, one can install Anaconda which would install Python, Jupyter, along with a lot of other stuff. Or, one can install only Jupyter using:
```
pip install jupyter
```

### ipywidgets

ipywidgets comes pre-installed with Jupyter. However, widgets might need to be actived using:
```
jupyter nbextension enable --py widgetsnbextension
jupyter nbextension enable --py --sys-prefix widgetsnbextension
```

## References
- [Machine Learning Mastery's excellent tutorial](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)

- [Mattmazur's example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)

- [Welch Lab's excellent video playlist on neural networks](https://www.youtube.com/playlist?list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU)

- [Michael Nielsen's brilliant hands-on interactive tutorial on the awesome power of neural networks as universal approximators](https://neuralnetworksanddeeplearning.com/chap4.html)

- [CS321n's iPython tutorial](https://cs231n.github.io/ipython-tutorial/)

- [Karlijn Willem's definitive Jupyter guide](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook#gs.SJPul58)

- [matplotlib](https://matplotlib.org/)

- [Tutorial on using Matplotlib in Jupyter](https://nbviewer.jupyter.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-4-Matplotlib.ipynb)

- [Interactive dashboards in Jupyter](https://blog.dominodatalab.com/interactive-dashboards-in-jupyter/)

- [ipywidgets - for interactive dashboards in Jupyter](http://ipywidgets.readthedocs.io/)

- [drawing-animating-shapes-matplotlib](https://nickcharlton.net/posts/drawing-animating-shapes-matplotlib.html)

- [RISE - for Jupyter presentations](https://github.com/damianavila/RISE)

- [MNIST dataset and results](http://yann.lecun.com/exdb/mnist/)

- [MNIST dataset .npz file (Amazon AWS)](https://s3.amazonaws.com/img-datasets/mnist.npz)

- [NpzFile doc](http://docr.it/numpy/lib/npyio/NpzFile)

- [matplotlib examples from SciPy](http://scipython.com/book/chapter-7-matplotlib/examples/simple-surface-plots/)

- [Yann LeCun's backprop paper, containing tips for efficient backpropagation](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

- [Mathematical notations for LaTeX, which can also be used in Jupyter](https://en.wikibooks.org/wiki/LaTeX/Mathematics)

- [JupyterHub](http://jupyterhub.readthedocs.io/en/latest/getting-started.html)

- [Optional code visibility in iPython notebooks](https://chris-said.io/2016/02/13/how-to-make-polished-jupyter-presentations-with-optional-code-visibility/)

- [Ultimate iPython notebook tips](https://blog.juliusschulz.de/blog/ultimate-ipython-notebook)

- [Full preprocessing for medical images tutorial](https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial)

- [Example ConvNet for a kaggle problem (cats vs dogs)](https://www.kaggle.com/sentdex/dogs-vs-cats-redux-kernels-edition/full-classification-example-with-convnet)

- Fernando Pérez, Brian E. Granger, IPython: A System for Interactive Scientific Computing, Computing in Science and Engineering, vol. 9, no. 3, pp. 21-29, May/June 2007, doi:10.1109/MCSE.2007.53. URL: (http://ipython.org)

