import numpy as np

class Model:
    def predict(self, x: np.ndarray) -> int:
        """ Gives a prediction for an input vector x based on the current hyperparameters

            Parameters:
            x (np.ndarray)

            Returns:
            float: A prediction based on the current hyperparameters
        """
        raise NotImplementedError('Subclasses must override method predict!')
    
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
        raise NotImplementedError('Subclasses must override method fit!')
    
    def reset_hyperparameters(self) -> None:
        """ Resets model's hyperparameters to random

            Parameters:
            None

            Returns:
            None
        """
        raise NotImplementedError('Subclasses must override method reset_hyperparameters!')