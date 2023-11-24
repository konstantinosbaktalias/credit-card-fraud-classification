import numpy as np
from model_abstract import Model

def k_fold_cross_validation(
        model: Model,
        k: int, 
        X: np.ndarray, 
        Y: np.ndarray, 
        epochs: int=10, 
        lr: float=0.001,
        y_class_1_val: int=1,
        y_class_2_val: int=0,
        reset_for_each_fold: bool=False
    ) -> None:
    """ Permorms the k fold validation procedure on the given model

        Parameters:
        model (Model)
        k (int)
        X (np.ndarray)
        Y (np.ndarray) 
        epochs (int)
        lr (float)
        y_class_1_val (int)
        y_class_2_val (int)
        train_per_fold (bool)
        
        Returns:
        float: average accuracy
    """
    fold_size: int = X.shape[0] // k
    acc: float = 0
    idx: int = 0

    for i in range(k):
        if idx - fold_size >= 0:
            model.fit(X=X[idx - fold_size:idx], y=Y[idx - fold_size:idx], epochs=epochs, lr=lr)

        if idx + fold_size < X.shape[0]:
            model.fit(X=X[idx+fold_size:], y=Y[idx+fold_size:], epochs=epochs, lr=lr)
        
        confusion_matrix: np.ndarray = np.zeros((2, 2), dtype=np.int32)

        for x, y in zip(X[idx:idx + fold_size], Y[idx:idx + fold_size]):
            y_pred: int = model.predict(x)
            
            if y_pred == y_class_2_val and y == y_class_2_val:
                confusion_matrix[0, 0] += 1
            elif y_pred == y_class_1_val and y == y_class_2_val:
                confusion_matrix[0, 1] += 1            
            elif y_pred == y_class_2_val and y == y_class_1_val:
                confusion_matrix[1, 0] += 1
            else:
                confusion_matrix[1, 1] += 1
        
        print(f'Fold #{i+1} Confusion Matrix:')
        print_confusion_matrix(confusion_matrix)
        print()
        idx += fold_size

        acc += (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)

        if reset_for_each_fold:
            model.reset_hyperparameters()
    
    return acc / k

def print_confusion_matrix(confusion_matrix: np.ndarray) -> None:
    """ Prints the given confusion matrix in a nice format

        Parameters:
        confusion_matrix (np.ndarray)

        Returns:
        None
    """
    cell_size: int = 7

    for row in confusion_matrix:
        for val in row:
            if len(str(val)) > cell_size:
                cell_size = len(str(val))

    print(''.join(' ' for i in  range(cell_size + 2)), end='')
    print(f'| Class 1', end ='')
    if (cell_size > 7):
        print(''.join(' ' for i in  range(cell_size - 7)), end='')
    print(f' | Class 2', end='')
    if (cell_size > 7):
        print(''.join(' ' for i in  range(cell_size - 7)), end='')
    print(' |')

    print(''.join('-' for i in range(4 * cell_size)))

    print(' Class 1', end='')
    if (cell_size > 7):
        print(''.join(' ' for i in  range(cell_size - 7)), end='')
    print(f' | {confusion_matrix[0, 0]}', end='')
    if (cell_size > len(str(confusion_matrix[0, 0]))):
        print(''.join(' ' for i in  range(cell_size - len(str(confusion_matrix[0, 0])))), end='')
    print(f' | {confusion_matrix[0, 1]}', end='')
    if (cell_size > len(str(confusion_matrix[0, 1]))):
        print(''.join(' ' for i in  range(cell_size - len(str(confusion_matrix[0, 1])))), end='')
    print(' |')

    print(''.join('-' for i in range(4 * cell_size)))

    print(' Class 2', end='')
    if (cell_size > 7):
        print(''.join(' ' for i in  range(cell_size - 7)), end='')
    print(f' | {confusion_matrix[1, 0]}', end='')
    if (cell_size > len(str(confusion_matrix[0, 0]))):
        print(''.join(' ' for i in  range(cell_size - len(str(confusion_matrix[1, 0])))), end='')
    print(f' | {confusion_matrix[1, 1]}', end='')
    if (cell_size > len(str(confusion_matrix[0, 1]))):
        print(''.join(' ' for i in  range(cell_size - len(str(confusion_matrix[1, 1])))), end='')
    print(' |')

    print(''.join('-' for i in range(4 * cell_size)))