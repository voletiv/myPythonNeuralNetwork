import numpy as np
DEBUG = 0

class MyPyNN(object):

    def __init__(self, layers=[3, 4, 2]):

        self.layers = layers

        # Network
        self.weights = [np.random.randn(x+1, y) 
                        for x, y in zip(layers[:-1], layers[1:])]

    def predict(self, X, visible=False):
        self.visible = visible
        inputs = self.preprocessInputs(X)

        if inputs.ndim!=1 and inputs.ndim!=2:
            print "X is not one or two dimensional, please check."
            return

        if DEBUG or self.visible:
            print "PREDICT:"
            print inputs

        for w in self.weights:
            inputs = self.addBiasTerms(inputs)
            inputs = self.sigmoid(np.dot(inputs, w))
            if DEBUG or self.visible:
                print "Layer "+str(l+1)
                print inputs
        
        return inputs

    def trainUsingGD(self, X, y, nIterations=1000, learningRate=0.05,
                        regLambda=0, visible=False):
        self.learningRate = learningRate
        self.regLambda = regLambda
        self.visible = visible
        yPred = self.predict(X, visible=self.visible)
        print "accuracy = " + str((np.sum([np.all((yPred[k]>0.5)==y[k])
                                        for k in range(len(y))])).astype(float)/len(y))
        print "cost = " + str(0.5*np.sum((yPred-y)**2)/len(y))
        for i in range(nIterations):
            print "Iteration "+str(i)+" of "+str(nIterations)
            self.forwardProp(X)
            self.backPropGradDescent(X, y)
            yPred = self.predict(X, visible=self.visible)
            print "accuracy = " + str((np.sum([np.all((yPred[k]>0.5)==y[k])
                                        for k in range(len(y))])).astype(float)/len(y))
            print "cost = " + str(0.5*np.sum((yPred-y)**2)/len(y))

    def trainUsingSGD(self, X, y, nIterations=1000, minibatchSize=100,
                        learningRate=0.05, regLambda=0, visible=False):
        self.learningRate = float(learningRate)
        self.regLambda = regLambda
        self.visible = visible
        X = self.preprocessInputs(X)
        y = self.preprocessOutputs(y)
        yPred = self.predict(X, visible=self.visible)
        if yPred.shape != y.shape:
            print "Shape of y ("+str(y.shape)+") does not match what shape of y is supposed to be: "+str(yPred.shape)
            return
        print "accuracy = " + str((np.sum([np.all((yPred[k]>0.5)==y[k])
                                        for k in range(len(y))])).astype(float)/len(y))
        print "cost = " + str(0.5*np.sum((yPred-y)**2)/len(y))
        idx = range(len(X))
        if minibatchSize > len(X):
            minibatchSize = int(len(X)/10)+1
        for i in range(nIterations):
            print "Iteration "+str(i)+" of "+str(nIterations)
            np.random.shuffle(idx)
            idx = idx[:minibatchSize]
            miniX = X[idx]
            miniY = y[idx]
            a = self.forwardProp(miniX)
            if a==True:
                self.backPropGradDescent(miniX, miniY)
            else:
                return
            yPred = self.predict(X, visible=self.visible)
            if self.visible:
                print yPred
            print "accuracy = " + str((np.sum([np.all((yPred[k]>0.5)==y[k])
                                        for k in range(len(y))])).astype(float)/len(y))
            print "cost = " + str(0.5*np.sum((yPred-y)**2)/len(y))

    def forwardProp(self, inputs):
        inputs = self.preprocessInputs(inputs)
        print "Forward..."

        if inputs.ndim!=1 and inputs.ndim!=2:
            print "Input argument " + str(inputs.ndim) + \
                "is not one or two dimensional, please check."
            return False

        if (inputs.ndim==1 and len(inputs)!=self.layers[0]) or \
            (inputs.ndim==2 and inputs.shape[1]!=self.layers[0]):
            print "Input argument does not match input dimensions (" + \
                str(self.layers[0]) + ") of network."
            return False
        
        if DEBUG or self.visible:
            print inputs

        self.outputs = []
        for l, w in enumerate(self.weights):
            inputs = self.addBiasTerms(inputs)
            self.outputs.append(self.sigmoid(np.dot(inputs, w)))
            inputs = np.array(self.outputs[-1])
            if DEBUG or self.visible:
                print "Layer "+str(l+1)
                print inputs
        del inputs

        return True

    def backPropGradDescent(self, X, y):
        X = self.preprocessInputs(X)
        y = self.preprocessOutputs(y)
        print "...Backward"
        # Compute first error
        error = self.outputs[-1] - y

        if DEBUG or self.visible:
            print "error = self.outputs[-1] - y:"
            print error

        for l, w in enumerate(reversed(self.weights)):
            if DEBUG or self.visible:
                print "LAYER "+str(len(self.weights)-l)
            
            predOutputs = self.outputs[len(self.weights)-l-1]

            if DEBUG or self.visible:
                print "predOutputs"
                print predOutputs

            # delta = (z*(1-z))*(z - zHat) === nxneurons
            delta = np.multiply(np.multiply(predOutputs, 1 - predOutputs),
                    error)

            if DEBUG or self.visible:
                print "To compute error to be backpropagated:"
                print "del = predOutputs*(1 - predOutputs)*error :"
                print delta
                print "weights:"
                print w

            # Compute new error to be propagated back (bias term neglected in backpropagation)
            error = np.dot(delta, w[1:,:].T)

            if DEBUG or self.visible:
                print "backprop error = np.dot(del, w[1:,:].T) :"
                print error

            # inputs === outputs from previous layer
            if l==len(self.weights)-1:
                inputs = np.array(X)
            else:
                inputs = np.array(self.outputs[len(self.weights)-l-2])
            inputs = self.addBiasTerms(inputs)
            
            if DEBUG or self.visible:
                print "To compute errorTerm:"
                print "inputs:"
                print inputs
                print "del:"
                print delta

            # errorTerm = (inputs.T).*(delta)
            # delta === nxneurons, inputs === nxprev, W === prevxneurons
            errorTerm = np.dot(inputs.T, delta)/len(y)
            if errorTerm.ndim==1:
                errorTerm.reshape((len(errorTerm), 1))

            if DEBUG or self.visible:
                print "errorTerm = np.dot(inputs.T, del) :"
                print errorTerm
            
            # regularization term
            regWeight = np.zeros(w.shape)
            regWeight[1:,:] = self.regLambda

            if DEBUG or self.visible:
                print "To update weights:"
                print "learningRate*errorTerm:"
                print self.learningRate*errorTerm
                print "regWeight:"
                print regWeight
                print "weights:"
                print w
                print "regTerm = regWeight*w :"
                print regWeight*w

            # Update weights
            self.weights[len(self.weights)-l-1] = w - \
                (self.learningRate*errorTerm + np.multiply(regWeight,w))
            
            if DEBUG or self.visible:
                print "Updated 'weights' = learningRate*errorTerm + regTerm :"
                print self.weights[len(self.weights)-l-1]

    def preprocessInputs(self, X):
        X = np.array(X, dtype=float)
        # if X is int
        if X.ndim==0:
            X = np.array([X])
        # if X is 1D
        if X.ndim==1:
            if self.layers[0]==1: #if ndim=1
                X = np.reshape(X, (len(X),1))
            else: #if X is only 1 nd-ndimensional vector
                X = np.reshape(X, (1,len(X)))
        return X

    def preprocessOutputs(self, Y):
        Y = np.array(Y, dtype=float)
        # if Y is int
        if Y.ndim==0:
            Y = np.array([Y])
        # if Y is 1D
        if Y.ndim==1:
            if self.layers[-1]==1:
                Y = np.reshape(Y, (len(Y),1))
            else:
                Y = np.reshape(Y, (1,len(Y)))
        return Y

    def addBiasTerms(self, X):
        if X.ndim==0 or X.ndim==1:
            X = np.insert(X, 0, 1)
        elif X.ndim==2:
            X = np.insert(X, 0, 1, axis=1)
        return X

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
