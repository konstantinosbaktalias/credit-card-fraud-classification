import numpy as np

from model_abstract import Model
from k_dtree import KDTree, HeapEntry

class KNN(Model):
    def __init__(self, k: int) -> None:
        """ Initialiser of class KNN

            Parameters:
            k (int): number of neighbours 

            Returns:
            None
        """
        self.tree: KDTree = KDTree()
        self.k: int = k

    def predict(self, x: np.ndarray) -> int:
        """ Gives a prediction for an input vector x based on the current hyperparameters

            Parameters:
            x (np.ndarray)

            Returns:
            float: A prediction based on the current hyperparameters
        """
        k_nearest_neighbours: list = self.tree.get_k_nearest_points(x, self.k)

        counts: dict = {}

        for neighbour in k_nearest_neighbours:
            if not neighbour.label in counts:
                counts[neighbour.label] = 0
            counts[neighbour.label] += 1
        
        max_count_label: int = k_nearest_neighbours[0].label

        for label in counts:
            if counts[label] > counts[max_count_label]:
                max_count_label = label

        return max_count_label
    

    def fit(
            self, 
            X: np.ndarray, 
            y: np.ndarray,
            epochs: int=0,
            lr: float=0,
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
        for x, y_i in zip(X, y):
            self.tree.insert_point(x, y_i)

    def reset_hyperparameters(self) -> None:
        """ Resets model's hyperparameters to random

            Parameters:
            None

            Returns:
            None
        """
        self.tree = KDTree()