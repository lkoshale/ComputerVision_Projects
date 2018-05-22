import numpy as np

class NeuralNetwork:

    # layers is list of integers # of nodes in each layer
    #Wis list of weights for each layer
    def __init__(self,layers,alpha=0.1):
        self.W =[]
        self.layers = layers
        self.alpha = alpha

        #bias termas are added in weights
        for i in np.arange(0,len(layers)-2):
            w = np.random.randn(layers[i]+1,layers[i+1]+1)
            self.W.append(w/np.sqrt(layers[i]))

        # the last two layers are a special case where the input
        22  # connections need a bias term but the output does not
        w = np.random.randn(layers[-2]+1,layers[-1])
        self.W.append(w/np.sqrt(layers[-2]))


    def __repr__(self):
        # construct and return a string that represents the network
        # architecture
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_deriv(self,x):
        return x * (1-x)

    def fit(self,X,y,epochs=1000,displayUpdate=100):
        #add bias terms
        X = np.c_[X,np.ones((X.shape[0]))]

        for epoch in np.arange(0,epochs):

            for (x,target) in zip(X,y):
                self.fit_partial(x,y)

            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
            print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self,x,y):

        A = [np.atleast_2d(x)]

        for layer in np.arange(0,len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)


        #TODO BACKPROPOGATION
        error = A[-1] - y
        D = [error*self.sigmoid_deriv(A[-1])]

        for layer in np.arange(len(A)-2,0,-1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta* self.sigmoid_deriv(A[layer])
            D.append(delta)

        #reverse Delta
        D = D[::-1]

        #update wieght

        for layer in np.arange(0,len(self.W)):
            self.W[layer]+= -self.alpha*A[layer].T.dot(D[layer])





