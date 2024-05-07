import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels


class NeuralNetwork(ClassifierMixin, BaseEstimator):
    """
    Neural network implemented by Giovanni Sequani

    Parameters
    ----------
    n_hidden_units : list, default=[10]
        List of number of units for each hidden layer.
    eta: float, default=0.1
        Learning rate.
    l2: float, default=0.01
        L2 regolarization parameter.
    epochs: int, default=100
        Number of epochs of training.
    seed: int, default=0
        Random seed for weights intialization.
        
    """
    def __init__(self,
                 n_hidden_units: list | None = [10],
                 eta: float | None = 0.1,
                 l2: float | None = 0.01,
                 epochs: int | None = 100,
                 seed: int | None = 0) -> None:
        
        self.n_hidden_units = n_hidden_units
        self.eta = eta
        self.l2 = l2
        self.epochs = epochs
        self.seed = seed

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit neural network

        Parameters
        ----------
        X_train : numpy.ndarray
            Input features dataset.
        y_train : numpy.ndarray
            Labels of the dataset.

        Returns
        -------
        self : object
            Returns self.
    
        """
        ## split del dataset
        ind = int(np.floor(0.7 * X_train.shape[0]))
        X_valid, X_train = X_train[ind:], X_train[:ind]
        y_valid, y_train = y_train[ind:], y_train[:ind]

        ## usefull info
        start = time.time()
        n_examples = X_train.shape[0] # number of obs.
        n_features = X_train.shape[1] # number of features
        n_output = np.unique(y_train).shape[0] # number of levels of the target variable
        self.classes_ = unique_labels(y_train) 

        ## weights initialization
        random = np.random.RandomState(self.seed)
        self.weights = []
        self.intercepts = []
        dimensions = [n_features]+self.n_hidden_units+[n_output]
        for i in range(len(dimensions)-1):
            self.weights.append(random.normal(loc=0,scale=0.1,
                                              size=(dimensions[i], dimensions[i+1])))
            self.intercepts.append(np.zeros(dimensions[i+1]))

        ## encoding of y: returns matrix of dummy [n_examples]x[n_output]
        y_enc = np.zeros(shape=(n_examples, n_output))
        for ind, val in enumerate(y_train.astype(int)):
            y_enc[ind,val] = 1

        ## dict for accuracy
        self.acc = {'train': [], 'valid': []}

        # repeat [self.epochs] times
        for j in range(self.epochs):

            # repeat [n_examples] times
            for idx in range(n_examples):
                
                X = X_train[[idx]] # select obs.

                a_hidden, z = self._forward_propagation(X)
                deltas = self._back_propagation(a_hidden, y_enc[idx])
                gradienti_weights, gradienti_intercepts = self._gradients(X, deltas, a_hidden)

                ## L2 regolarization
                delta_weights = []
                delta_intercepts = []
                for i in range(len(self.weights)):
                    delta_weights.append(gradienti_weights[i] + self.l2 * self.weights[i])
                    delta_intercepts.append(gradienti_intercepts[i]) # intercetta non regolarizzata

                ## weights update
                for i in range(len(self.weights)):
                    self.weights[i] -= self.eta * delta_weights[i]
                    self.intercepts[i] -= self.eta * delta_intercepts[i]


            ## final forward propagation
            a_hidden, z = self._forward_propagation(X)

            ## evaluate accuracy
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_accuracy = (np.sum(y_train == y_train_pred)).astype(np.float32) / n_examples
            valid_accuracy = (np.sum(y_valid == y_valid_pred)).astype(np.float32) / len(y_valid)

            self.acc['train'].append(train_accuracy)
            self.acc['valid'].append(valid_accuracy)

            sys.stderr.write('\repoch: %d/%d | valid accuracy: %.2f' % 
                             (j+1, self.epochs, valid_accuracy*100))
            sys.stderr.flush()
            
        s = time.time() - start
        print(f"\ntraining time: {int(s/60)} min {int(s%60)} s")

        return self

    def _forward_propagation(self, X: np.ndarray) -> tuple:
        """
        Internal method: forward propagation algorithm.

        Parameters
        ----------
        X : numpy.ndarray
            Array of features.
        
        Returns
        -------
        a_hidden : list
            List of arrays of activation values for each layer.
        """
        a_hidden = []
        layer_input = X
        for i in range(len(self.weights)):
                z_hidden = np.dot(layer_input, self.weights[i]) + self.intercepts[i]
                a_hidden.append(self._sigmoid_function(z_hidden))
                layer_input = a_hidden[-1]

        return a_hidden, z_hidden
    
    def _back_propagation(self, a_hidden: list, y: np.ndarray) -> list:
        """
        Internal method: back propagation algorithm.

        Parameters
        ----------
        a_hidden: list
            List of arrays of activation values for each layer.
        y: numpy.ndarray
            Array of target variable in onehot rappresentation.
        
        Returns
        -------
        deltas: list
            List of array of error of each layer.

        """
        deltas = [a_hidden[-1] - y] # errori
        sigmoid_derivative = []
        for i in range(len(a_hidden)-2, -1, -1):
                sigmoid_derivative.insert(0, a_hidden[i] * (1. - a_hidden[i])) # derivata della funzione sigmoide # VA TOLTA
                delta = np.dot(deltas[0], self.weights[i+1].T) * sigmoid_derivative[0] # VA TOLTA LA DERIV
                deltas.insert(0, delta)
        
        return deltas
    
    def _gradients(self, X: np.ndarray, deltas: list, a_hidden: list) -> tuple:
        """
        Internal method: evaluate gradients of each weight and intercept.

        Parameters
        ----------
        X : numpy.ndarray
            Array of features.
        deltas: list
            List of array of error of each layer.
        a_hidden: list
            List of arrays of activation values for each layer.

        Returns
        -------
        gradient_weights: list
            List of array of weights gradients for each layer
        gradient_intercepts: list
            List of array of intercepts gradients for each layer

        """
        gradient_weights = []
        gradient_intercepts = []
        layer_input = X
        for i in range(len(self.weights)):
            gradient_weights.append(np.dot(layer_input.T, deltas[i]))
            gradient_intercepts.append(np.sum(deltas[i], axis=0))
            layer_input = a_hidden[i]
        
        return gradient_weights, gradient_intercepts

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels

        Parameters
        ----------
        X : numpy.ndarray
            Input features dataset.

        Returns
        -------
        y_pred : numpy.ndarray
            Predicted labels.

        """
        a_hidden, z = self._forward_propagation(X)

        return np.argmax(z, axis=1)

    def _sigmoid_function(self, x):
        """
        Internal method: sigmoid function.
        """
        return 1. / (1. + np.exp(-np.clip(x, -250,250)))
    
    def plot_accuracy(self, set: str | None = "valid", save: bool | None = False) -> None:
        """
        Plot of accuracy curve for each epochs.
        
        Parameters
        ----------
        set : str (default: "valid")
            "train" => plot accuracy evaluated on the train set.

            "valid" => plot accuracy evaluated on the validation set.

            "both" => plot both accuracy.
        
        """
        if set == "both":
            plt.plot(range(1,self.epochs+1), self.acc["valid"], label="valid")
            plt.plot(range(1,self.epochs+1), self.acc["train"], label="train")
        else:
            plt.plot(range(1,self.epochs+1), self.acc[set], label=set)
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.title("Accuracy plot")
        plt.legend(loc='lower right')
        if save:
            plt.savefig(f"img/n{self.n_hidden_units} eta{self.eta} l2{self.l2}.png")
        plt.show()

    def accuracy(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate neural network's accuracy.

        Parameters
        ----------
        X_test : numpy.ndarray
            Input features dataset.
        y_test : numpy.ndarray
            Labels of the dataset.

        Returns
        -------
        accuracy: float
            Accuracy of the neural network.

        """
        y_pred = self.predict(X_test)
        accuracy = (np.sum(y_test == y_pred)).astype(np.float32) / len(y_test)
    
        return accuracy