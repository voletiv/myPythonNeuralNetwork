from myPyNN import *

# RANDOM
X = [[2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6], [7,7,7], [8,8,8], [9,9,9], [10,10,10], [11,11,11]]
y = [.2, .3, .4, .5, .6, .7, .8, .9, 0, .1]
myNN = MyPyNN([3, 10, 1])

X = np.random.random((10,3))
y = np.reshape(np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0]), (10,1))
myNN = MyPyNN(3, 1, 4, 1)
myNN.trainUsingSGD(X, y, alpha=0.8)

# MANUAL CALCULATIONS
def addBiasTerms(X):
    if X.ndim==0 or X.ndim==1:
        X = np.insert(X, 0, 1)
    elif X.ndim==2:
        X = np.insert(X, 0, 1, axis=1)
    return X

def sigmoid(z):
    return 1/(1 + np.exp(-z))

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [1]])
myNN = MyPyNN([2, 1, 1])
alpha = 0.9
nIterations = 600
W01 = myNN.weights[0]
W02 = myNN.weights[1]
W1 = W01
W2 = W02
for i in range(nIterations):
    yPred = sigmoid(np.dot(addBiasTerms(sigmoid(np.dot(addBiasTerms(X), W1))), W2))
    err2 = yPred - y
    output1 = sigmoid(np.dot(addBiasTerms(X), W1))
    del2 = np.multiply(np.multiply(yPred, (1-yPred)), err2)/len(yPred)
    err1 = np.dot(del2, W2[1:].T)
    deltaW2 = alpha*np.dot(addBiasTerms(output1).T, del2)
    newW2 = W2 - deltaW2
    del1 = np.multiply(np.multiply(output1, 1-output1), err1)/len(output1)
    deltaW1 = alpha*np.dot(addBiasTerms(X).T, del1)
    newW1 = W1 - deltaW1
    W1 = newW1
    W2 = newW2

myNN.trainUsingGD(X, y, alpha=alpha, nIterations=nIterations)
newW1 == myNN.weights[0]
newW2 == myNN.weights[1]

# COMPARING LEARNING RATES
myNN1 = MyPyNN([2, 3, 1])
myNN2 = MyPyNN([2, 3, 1])
myNN3 = MyPyNN([2, 3, 1])
myNN4 = MyPyNN([2, 3, 1])
myNN5 = MyPyNN([2, 3, 1])
myNN2.weights[0] = myNN1.weights[0]
myNN2.weights[1] = myNN1.weights[1]
myNN3.weights[0] = myNN1.weights[0]
myNN3.weights[1] = myNN1.weights[1]
myNN4.weights[0] = myNN1.weights[0]
myNN4.weights[1] = myNN1.weights[1]
myNN5.weights[0] = myNN1.weights[0]
myNN5.weights[1] = myNN1.weights[1]
myNN1.trainUsingGD(X, y, alpha=0.1, nIterations=1000)
myNN2.trainUsingGD(X, y, alpha=0.5, nIterations=600)
myNN3.trainUsingGD(X, y, alpha=1, nIterations=300)
myNN4.trainUsingGD(X, y, alpha=2, nIterations=200)
myNN5.trainUsingGD(X, y, alpha=200, nIterations=1000)


