import numpy as np
from model_abstract import Model

class SVM(Model):
    def __init__(self, num_of_features: int, C: float=1) -> None:
        """ Initialises class LogisticRegression with random weights and bias

            Parameters:
            num_of_features (int)

            Returns:
            None
        """
        self.weights: np.ndarray = np.random.rand(num_of_features)
        self.w0: float = np.random.rand()
        self.C = C

    def predict(self, x: np.ndarray) -> int:
        """ Gives a prediction for an input vector x based on the current hyperparameters

            Parameters:
            x (np.ndarray)

            Returns:
            int: A prediction based on the current hyperparameters
        """
        y: float = self.weights @ x + self.w0

        return 1 if y >= 0 else -1
    
    def fit(
            self, 
            X: np.ndarray, 
            y: np.ndarray,
            epochs: int,
            lr: float=0.0001,
            print_epoch_num: bool = False
        ) -> None:
        """ Fit the model's parameters on the given data using gradient descent

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
            for x, y_label in zip(X, y):
                weight_gradient: np.ndarray = self.calculate_gradient_of_weights(x, y_label)
                bias_gradient: float =  self.calculate_gradient_of_bias(x, y_label)

                self.weights -= lr * weight_gradient 
                self.w0 -= lr * bias_gradient

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

    def calculate_gradient_of_weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Calculates the gradient of the weights logistic function

            Parameters:
            X (np.ndarray): Collection of all feature vectors
            Y (np.ndarray): Collection of corresponding labels

            Returns:
            np.ndarray: Gradient of the weight vector
        """
        y_pred: float = 1 - y * (self.weights @ x + self.w0)

        return (
            self.weights - self.C * y * x  if y_pred >= 0 
            else self.weights
        )
    
    def calculate_gradient_of_bias(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Calculates the gradient of the weights logistic function

            Parameters:
            X (np.ndarray): Collection of all feature vectors
            Y (np.ndarray): Collection of corresponding labels

            Returns:
            np.ndarray: Gradient of the weight vector
        """
        y_pred: float = 1 - y * (self.weights @ x + self.w0)

        return (
            - self.C * y if y_pred >= 0
            else 0
        )