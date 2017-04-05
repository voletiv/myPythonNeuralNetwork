from myPyNN import *

X = [[0,0], [1,1]]
y = [0, 1]
myNN = MyPyNN(2, 1, 3, 1, 0.8)
myNN.predict(X)
#myNN.trainUsingGD(X, y, 899)
myNN.trainUsingSGD(X, y, 1000)


X = [[2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6], [7,7,7], [8,8,8], [9,9,9], [10,10,10], [11,11,11]]
y = [.2, .3, .4, .5, .6, .7, .8, .9, 0, .1]
myNN = MyPyNN(3, 1, 10, 1)
yHat = myNN.forwardProp(2)
myNN.backPropGradDescent(0.5)




