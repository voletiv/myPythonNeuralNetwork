import numpy as np
DEBUG = 0

class MyPyNN(object):

    def __init__(self, nOfInputDims=3, nOfHiddenLayers=1, \
                    hiddenLayerSizes=4, nOfOutputDims=2, alpha=0.05, regLambda=0):

        if isinstance(hiddenLayerSizes, int):
            hiddenLayerSizes = [hiddenLayerSizes]

        if len(hiddenLayerSizes) != nOfHiddenLayers:
            print "Please specify sizes of hidden layers properly!!"
            return

        self.layerSizes = list(hiddenLayerSizes) #list() => deep copy
        self.layerSizes.insert(0, nOfInputDims)
        self.layerSizes.append(nOfOutputDims)

        self.alpha = alpha
        self.regLambda = regLambda

        # Network
        self.network = [{'weights':np.array(np.random.random(\
                        (self.layerSizes[layer-1]+1,self.layerSizes[layer])))} \
                        for layer in range(1,len(self.layerSizes))]

    def predict(self, X):
        inputs = self.preprocessInputs(X)

        if inputs.ndim!=1 and inputs.ndim!=2:
            print "X is not one or two dimensional, please check."
            return

        if DEBUG:
            print inputs

        for l, layer in enumerate(self.network):
            inputs = self.addBiasTerms(inputs)
            inputs = self.sigmoid(np.dot(inputs, layer['weights']))
            if DEBUG:
                print "Layer "+str(l+1)
                print inputs
        
        return inputs

    def trainUsingGD(self, X, y, nIterations=1000):
        for i in range(nIterations):
            print i
            self.forwardProp(X)
            self.backPropGradDescent(X, y)
            print self.predict(X)

    def trainUsingSGD(self, X, y, nIterations=100):
        X = self.preprocessInputs(X)
        y = self.preprocessOutputs(y)
        idx = range(len(X))
        for n in range(nIterations):
            print n
            np.random.shuffle(idx)
            for i in idx:
                print "  "+str(i)
                a = self.forwardProp(X[i])
                if a==True:
                    self.backPropGradDescent(X[i], y[i])
                else:
                    return
            print self.predict(X)

    def forwardProp(self, inputs):
        inputs = self.preprocessInputs(inputs)
        print "Forward..."

        if inputs.ndim!=1 and inputs.ndim!=2:
            print "Input argument " + str(inputs.ndim) + \
                "is not one or two dimensional, please check."
            return False

        if (inputs.ndim==1 and len(inputs)!=self.layerSizes[0]) or \
            (inputs.ndim==2 and inputs.shape[1]!=self.layerSizes[0]):
            print "Input argument does not match input dimensions (" + \
                str(self.layerSizes[0]) + ") of network."
            return False
        
        if DEBUG:
            print inputs

        for l, layer in enumerate(self.network):
            inputs = self.addBiasTerms(inputs)
            layer['outputs'] = self.sigmoid(np.dot(inputs, layer['weights']))
            inputs = np.array(layer['outputs'])
            if DEBUG:
                print "Layer "+str(l+1)
                print inputs
        del inputs

        return True

    def backPropGradDescent(self, X, y):
        X = self.preprocessInputs(X)
        y = self.preprocessOutputs(y)
        print "...Backward"
        # Compute first error
        error = self.network[-1]['outputs'] - y

        if DEBUG:
            print "error = self.network[-1]['outputs'] - y:"
            print error

        for l, layer in enumerate(reversed(self.network)):
            if DEBUG:
                print "LAYER "+str(len(self.layerSizes)-1-l)
            
            predOutputs = layer['outputs']

            if DEBUG:
                print "predOutputs"
                print predOutputs

            # delta = (z*(1-z))*(z - zHat) === nxneurons
            delta = np.multiply(np.multiply(predOutputs, 1 - predOutputs), \
                    error)/len(y)

            if DEBUG:
                print "To compute error to be propagated:"
                print "delta = predOutputs*(1 - predOutputs)*error :"
                print delta
                print "layer['weights']:"
                print layer['weights']

            # Compute new error (for next iteration)
            error = np.dot(delta, layer['weights'][1:,:].T)

            if DEBUG:
                print "error = np.dot(delta, layer['weights'][1:,:].T) :"
                print error

            # inputs === outputs from previous layer
            if l==len(self.network)-1:
                inputs = np.array(X)
            else:
                inputs = np.array(self.network[len(self.layerSizes)-2-l-1]['outputs'])
            inputs = self.addBiasTerms(inputs)
            
            if DEBUG:
                print "To compute errorTerm:"
                print "inputs:"
                print inputs
                print "delta:"
                print delta

            # errorTerm = inputs'.delta
            # delta === nxneurons, inputs === nxprev, W === prevxneurons
            errorTerm = np.dot(inputs.T, delta)
            if errorTerm.ndim==1:
                errorTerm.reshape((len(errorTerm), 1))

            if DEBUG:
                print "errorTerm = np.dot(inputs.T, delta) :"
                print errorTerm
            
            # regularization term
            regWeight = np.zeros(layer['weights'].shape)
            regWeight[1:,:] = self.regLambda

            if DEBUG:
                print "To update weights:"
                print "alpha*errorTerm:"
                print self.alpha*errorTerm
                print "regWeight:"
                print regWeight
                print "layer weights:"
                print layer['weights']
                print "regTerm = regWeight*layer['weights'] :"
                print regWeight*layer['weights']

            # Update weights
            layer['weights'] = layer['weights'] - \
                (self.alpha*errorTerm + np.multiply(regWeight,layer['weights']))
            
            if DEBUG:
                print "Updated layer['weights'] = alpha*errorTerm + regTerm :"
                print layer['weights']

    def preprocessInputs(self, X):
        X = np.array(X, dtype=float)
        # if X is int
        if X.ndim==0:
            X = np.array([X])
        # if X is 1D
        if X.ndim==1:
            if self.layerSizes[0]==1:
                X = np.reshape(X, (len(X),1))
            else:
                X = np.reshape(X, (1,len(X)))
        return X

    def preprocessOutputs(self, Y):
        Y = np.array(Y, dtype=float)
        # if Y is int
        if Y.ndim==0:
            Y = np.array([Y])
        # if Y is 1D
        if Y.ndim==1:
            if self.layerSizes[-1]==1:
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
