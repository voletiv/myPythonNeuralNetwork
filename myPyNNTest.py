from myPyNN import *

# RANDOM
X = [[2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6], [7,7,7], [8,8,8], [9,9,9], [10,10,10], [11,11,11]]
y = [.2, .3, .4, .5, .6, .7, .8, .9, 0, .1]
myNN = MyPyNN([3, 10, 1])


# MANUAL CALCULATIONS TO CHECK NETWORK
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
lr = 1.5
nIterations = 1
W01 = myNN.weights[0]
W02 = myNN.weights[1]
W1 = W01
W2 = W02
X = X.astype('float')
inputs = X - np.reshape(np.mean(X, axis=0), (1, X.shape[1]))
for i in range(nIterations):
    yPred = sigmoid(np.dot(addBiasTerms(sigmoid(np.dot(addBiasTerms(inputs), W1))), W2))
    err2 = yPred - y
    output1 = sigmoid(np.dot(addBiasTerms(inputs), W1))
    del2 = np.multiply(np.multiply(yPred, (1-yPred)), err2)
    err1 = np.dot(del2, W2[1:].T)
    deltaW2 = lr*np.dot(addBiasTerms(output1).T, del2)/len(yPred)
    newW2 = W2 - deltaW2
    del1 = np.multiply(np.multiply(output1, 1-output1), err1)
    deltaW1 = lr*np.dot(addBiasTerms(inputs).T, del1)/len(yPred)
    newW1 = W1 - deltaW1
    W1 = newW1
    W2 = newW2

myNN.trainUsingGD(X, y, learningRate=lr, nIterations=nIterations, visible=True)
newW1 == myNN.weights[0]
newW2 == myNN.weights[1]

yPred == myNN.outputs[1]
output1 == myNN.outputs[0]


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
myNN1.trainUsingGD(X, y, learningRate=0.1, nIterations=2500)
myNN2.trainUsingGD(X, y, learningRate=0.5, nIterations=600)
myNN3.trainUsingGD(X, y, learningRate=1, nIterations=400)
myNN4.trainUsingGD(X, y, learningRate=2, nIterations=200)
myNN5.trainUsingGD(X, y, learningRate=200, nIterations=1000)


# Make network
myNN = MyPyNN([784, 30, 10])
lr = 3
nIterations = 30
minibatchSize = 10

# MNIST DATA
'''
f = np.load(path)

# To check files stored in .npz file
f.files

# Saving the files
x_train = f['x_train']
y_train = f['y_train']
x_test = f['x_test']
y_test = f['y_test']
f.close()

# Preprocess inputs
x_train_new = np.array([x.flatten() for x in x_train])
y_train_new = np.zeros((len(y_train), 10))
for i in range(len(y_train)):
    y_train_new[i][y_train[i]] = 1

x_test_new = np.array([x.flatten() for x in x_test])
y_test_new = np.zeros((len(y_test), 10))
for i in range(len(y_test)):
    y_test_new[i][y_test[i]] = 1
'''

[x_train_new, y_train_new, x_test_new, y_test_new] = myNN.loadMNISTData()

myNN.trainUsingGD(x_train_new, y_train_new, nIterations=nIterations, learningRate=lr)
myNN.trainUsingSGD(x_train_new, y_train_new, nIterations=nIterations, minibatchSize=minibatchSize, learningRate=lr)
myNN.trainUsingSGD(x_train_new, y_train_new, nIterations=nIterations, minibatchSize=minibatchSize, learningRate=lr, printTestAccuracy=True, testX=x_test_new, testY=y_test_new)

# Make network
myNN = MyPyNN([784, 5, 5, 10])
lr = 1.5
nIterations = 1000
minibatchSize = 100
myNN.trainUsingSGD(x_train_new, y_train_new, nIterations=nIterations, minibatchSize=minibatchSize, learningRate=lr)

# To check type of the dataset
type(x_train)
type(y_train)
# To check data
x_train.shape
y_train.shape
fig = plt.figure(figsize=(10, 2))
for i in range(20):
    ax1 = fig.add_subplot(2, 10, i+1)
    ax1.imshow(x_train[i], cmap='gray');
    ax1.axis('off')

