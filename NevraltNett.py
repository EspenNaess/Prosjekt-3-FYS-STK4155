import numpy as np
import pickle,os
import sklearn.model_selection as skms

# sigmoidfunksjonen
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x));

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x)+1e-12)

def logCrossEntropy_derivative(y,t):
    '''
    logCrossEntropy_derivative:
        param:
            t: t.d ein prediksjon
            y: kategori - anten 0 eller 1

        ret: gradient som skildra i kap 3 av Michael A. Nielsen si bok; Neural networks and deep learning.
    '''
    return y-t

def standardCost(y,t):
    return 1/2*np.sum((y-t)**2)

def standardCost_derivative(y,t):
    return y-t

class NevraltNett:
    def __init__(self, train_x, train_y, goymde_celler, modus, cost):
        '''
            __init__:
            Initaliserer det nevrale nettverket

                param:
                    train_x, train_y: treningsdatamengder for det nevrale nettverket
                    goymde_celler: ein array der talet på kvar indeks l utgjer talet på nerveceller i lag l
                    modus: klassifisering: 'k' / regresjon: 'r'
                    cost: kostfunksjon, verdiar: 'standard'/'crossentropy'
        '''

        self.train_x = train_x
        self.train_y = train_y
        self.modus = modus

        if cost == 'standard':
            self.costfunc = standardCost
            self.costderivative = standardCost_derivative
        else:
            self.costfunc = logCrossEntropy
            self.costderivative = logCrossEntropy_derivative

        self.goymde_celler = np.insert(goymde_celler, 0, train_x.shape[1])
        self.goymdelag = np.empty(len(self.goymde_celler)-1, object)

        if self.goymdelag.size != 0:
            self.goymdevekter = np.empty(len(self.goymdelag), object)
            for i in range(len(self.goymdevekter)):
                self.goymdevekter[i]=np.random.randn(self.goymde_celler[i], self.goymde_celler[i+1])*0.1

        self.outputvekter = np.random.randn(self.goymde_celler[-1], 1)*0.1

        self.goymdebiases = np.empty(len(self.goymdelag), object)
        for i in range(len(self.goymdebiases)):
            self.goymdebiases[i] = np.random.randn(1,self.goymde_celler[i+1])+0.0001

        self.outputbiases = np.random.randn(1,1)+0.0001

    def feedforward(self, X):
        '''
            feedforward:
                X: matrise som skal feedforwardast

                ret: output til det nevrale nettverket
        '''
        if self.goymdelag.size != 0:
            for i in range(len(self.goymdelag)):
                self.goymdelag[i] = sigmoid((X if i == 0 else self.goymdelag[i-1]) @ self.goymdevekter[i]+self.goymdebiases[i])

        self.outputlag = (X if self.goymdelag.size == 0 else self.goymdelag[-1]) @ self.outputvekter+self.outputbiases

        if self.modus == 'k':
            self.outputlag = sigmoid(self.outputlag)

        return self.outputlag

    def trenNettverk(self,mb_size, lr, tal_epokar):
        '''
            trenNettverk:
                Trener nettverket med dei initialiserte parametrane.
                Vert nytta mini batch gradient descent.

                param:
                    mb_size: storleik på minibatches
                    lr: læringsrate
                    tal_epokar: talet på epokar som skal køyrast
        '''
        #cost = self.costfunc(self.feedforward(self.train_x).flatten(),self.train_y)

        #print("Talet på epokar: {}".format(tal_epokar))
        tal_mb = int(self.train_x.shape[0]/mb_size+1)
        #print("Tal på minibatches: {}".format(tal_mb))

        iter_epokar = 1
        while(iter_epokar < tal_epokar):
            indices = np.arange(0,len(self.train_y))
            np.random.shuffle(indices)
            X_shuffled, y_shuffled = (self.train_x[indices], self.train_y[indices])

            for i in range(0, self.train_x.shape[0], mb_size):
                X_batch, y_batch = (X_shuffled[i:i+mb_size], y_shuffled[i:i+mb_size])
                self.feedforward(X_batch)
                self.backprop(X_batch, y_batch, lr)

            iter_epokar += 1
            #cost = self.costfunc(self.feedforward(self.train_x).flatten(),self.train_y)
            #print(cost)

    def backprop(self,X_batch,y_batch, lr):
        '''
            backprop:
            Gjer ein backpropagation.

                param:
                    X_batch, y_batch: batchen det vert gjort ein backpropagation for
                    lr: læringsrate
        '''
        backprop_errors = np.zeros(len(self.goymdelag)+1, object) # kvart element av same dimensjon som den tilsvarande vektmatrisa

        if self.costderivative == standardCost_derivative:
            backprop_errors[-1] = self.costderivative(self.outputlag,np.array([y_batch]).T)
            if self.modus == 'k':
                backprop_errors[-1] *= sigmoid_derivative((X if self.goymdelag.size == 0 else self.goymdelag[-1]) @ self.outputvekter+self.outputbiases)
        else:
            backprop_errors[-1] = self.costderivative(self.outputlag, np.array([y_batch]).T) # ser kanskje litt gale ut, men skal ikkje vera det. Meir info i rapport

        for l in reversed(range(0,len(backprop_errors)-1)):
            vekter = self.goymdevekter[l+1].T if l+1 < len(self.goymdevekter) else self.outputvekter.T
            backprop_errors[l] = backprop_errors[l+1] @ vekter * sigmoid_derivative((X_batch if l == 0 else self.goymdelag[l-1]) @ self.goymdevekter[l]+self.goymdebiases[l])

        for i in range(0,len(self.goymdelag)):
            self.goymdevekter[i] -= lr/len(X_batch)*((X_batch if i == 0 else self.goymdelag[i-1]).T @ backprop_errors[i])
            self.goymdebiases[i] -= lr/len(X_batch)*np.sum(backprop_errors[i],axis=0)

        self.outputvekter -= lr/len(X_batch)*self.goymdelag[-1].T @ backprop_errors[-1]
        self.outputbiases -= lr/len(X_batch)*np.sum(backprop_errors[-1],axis=0)

        # alt dette kun for debugging og samanlikning med scikit
        '''
        self.derivative_weights = np.empty(len(self.goymdevekter)+1, object)
        self.derivative_biases = np.empty(len(self.goymdevekter)+1, object)
        for i in range(0,len(self.goymdevekter)+1):
            self.derivative_weights[i] = lr/len(X_batch)*((X_batch if i == 0 else self.goymdelag[i-1]).T @ backprop_errors[i])
            self.derivative_biases[i] = lr/len(X_batch)*np.sum(backprop_errors[i],axis=0)
        '''
