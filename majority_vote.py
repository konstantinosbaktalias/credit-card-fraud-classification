import numpy as np
from model_abstract import Model

class MajorityVote(Model):
    def __init__(
                self, 
                 models: list, 
                 class_vals: list, 
                 cap_in_training_samples: list, 
                 features_considered: list
                ) -> None:
        """ Initialiser of class KNN

            Parameters:
            models (list): models that make up the multi-model
            class_vals (list): list of each models classes values
            
            Returns:
            None
        """
        super().__init__()
        self.models: list = models
        self.class_vals: list = class_vals
        self.cap_in_training_samples: list = cap_in_training_samples
        self.features_considered: list = features_considered

    def predict(self, x: np.ndarray) -> int:
        """ Gives a prediction for an input vector x based on the current hyperparameters

            Parameters:
            x (np.ndarray)

            Returns:
            float: A prediction based on the current hyperparameters
        """
        class_1_count: int = 0
        class_2_count: int = 0

        for model, class_vals, features in zip(self.models, self.class_vals, self.features_considered):
            if model.predict(x[features]) == class_vals[0]:
                class_1_count += 1
            else:
                class_2_count += 1
        
        if class_1_count > class_2_count:
            return 0
        
        return 1
    
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
        for model, class_vals, cap, features in zip(self.models, self.class_vals, self.cap_in_training_samples, self.features_considered):
            y_labels: np.ndarray = np.array(y)[:cap]

            y_labels[y_labels == 0] = class_vals[0]
            y_labels[y_labels == 1] = class_vals[1]

            model.fit(X[:cap, features], y_labels, epochs, lr, print_epoch_num)
    
    def reset_hyperparameters(self) -> None:
        """ Resets model's hyperparameters to random

            Parameters:
            None

            Returns:
            None
        """
        for model in self.models:
            model.reset_hyperparameters()