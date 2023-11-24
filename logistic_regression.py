import numpy as np
from model_abstract import Model

class LogisticRegression(Model):
    def __init__(self, num_of_features: int) -> None:
        """ Initialises class LogisticRegression with random weights and bias

            Parameters:
            num_of_features (int)

            Returns:
            None
        """
        self.weights: np.ndarray = np.random.rand(num_of_features)
        self.w0: float = np.random.rand()


    def predict(self, x: np.ndarray) -> int:
        """ Gives a prediction for an input vector x based on the current hyperparameters

            Parameters:
            x (np.ndarray)

            Returns:
            int: A prediction based on the current hyperparameters
        """
        a: float = LogisticRegression.sigmoid(self.weights @ x + self.w0)
        return 1 if a >= 0.5 else 0
    
    def fit(
            self, 
            X: np.ndarray, 
            y: np.ndarray,
            epochs: int,
            lr: float=0.0001,
            print_epoch_num: bool = False
        ) -> None:
        """ Fit the model's parameters on the given data using gradient ascent

            Parameters:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            epochs: (int): Number of epochs
            lr (float): Learning rate
            print_epoch_num (bool): whether to print the epoch numbers or not
         
            Returns:
            None
        """
        for epoch in range(1, epochs + 1):
            weight_gradient: np.ndarray = self.calculate_gradient_of_weights(X, y)
            bias_gradient: float = self.calculate_gradient_of_bias(X, y)

            self.weights += lr * weight_gradient
            self.w0 += lr * bias_gradient

            if print_epoch_num:
                print(f'Epoch #{epoch}: Completed')

    def reset_hyperparameters(self) -> None:
        """ Resets model's hyperparameters to random

            Parameters:
            None

            Returns:
            None
        """
        self.weights = np.random.rand(self.weights.shape[0])
        self.w0 = np.random.rand()

    def calculate_gradient_of_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Calculates the gradient of the weights logistic function

            Parameters:
            X (np.ndarray): Collection of all feature vectors
            Y (np.ndarray): Collection of corresponding labels

            Returns:
            np.ndarray: Gradient of the weight vector
        """
        gradient: np.ndarray = np.zeros(self.weights.shape)

        for i in range(X.shape[0]):
            a: float = LogisticRegression.sigmoid(self.weights @ X[i] + self.w0)
            sig_der: float = LogisticRegression.sigmoid_der(a)

            for j in range(X.shape[1]):
                if a > 0:
                    gradient[j] += (y[i] / a) * sig_der * X[i, j]
                if a < 1:
                    gradient[j] += ((y[i] - 1) / (1 - a)) * sig_der * X[i, j]


        return gradient

    def calculate_gradient_of_bias(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Calculates the gradient of the weights logistic function

            Parameters:
            X (np.ndarray): Collection of all feature vectors
            Y (np.ndarray): Collection of corresponding labels

            Returns:
            np.ndarray: Gradient of the weight vector
        """
        gradient: np.ndarray = 0

        for i in range(X.shape[0]):
            a: float = LogisticRegression.sigmoid(self.weights @ X[i] + self.w0)
            sig_der: float = LogisticRegression.sigmoid_der(a)

            if a > 0:
                gradient += (y[i] / a) * sig_der
            if a < 1:
                gradient += ((y[i] - 1) / (1 - a)) * sig_der

        return gradient

    def sigmoid(z: float) -> float:
        """ Calculates the sigmoid given an input z

            Parameters:
            z (float)

            Returns:
            float:The value of the sigmoid function for the given input
        """
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_der(a: float) -> float:
        """ Calculates the derivative of sigmoid, given the sigmoid output 

            Parameters:
            a (float): value of sigmoid at some point

            Returns:
            float:The value of sigmoid at some point
        """
        return a * (1 - a)