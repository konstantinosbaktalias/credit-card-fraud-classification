import numpy as np

def produce_synthetic_data(
        num_of_samples_to_produce: int, 
        existing_data: np.ndarray
    ) -> np.ndarray:
    """ Produce synthetic data based on existing samples
        
        Parameters:
        num_of_samples_to_produce (int)
        existing_data (np.ndarray)

        Returns:
        np.ndarray:Synthetic data produced
    """
    list_of_means: list = calculate_means_of_all_features(existing_data)
    list_of_standard_deviations: list = (
        calculate_standard_deviations_of_all_features(existing_data, list_of_means)
    )

    synthetic_data: list = []

    for i in range(num_of_samples_to_produce):
        synthetic_datapoint: list = []

        for j in range(existing_data.shape[1]):
            synthetic_datapoint.append(np.random.normal(list_of_means[j], list_of_standard_deviations[j]))
        
        synthetic_data.append(synthetic_datapoint)
    
    return np.array(synthetic_data)
    

def calculate_standard_deviations_of_all_features(
        dataset: np.ndarray, 
        means_of_features: list
    ) -> list:
    """ Calculate the standard deviations of all features in the given dataset

        Parameters:
        dataset (np.ndarray)

        Returns:
        list:Standard deviations of all features
    """
    list_of_standard_deviations: list = []

    for j in range(dataset.shape[1]):
        list_of_standard_deviations.append(
            np.sqrt(
                calculate_variance_of_feature(dataset, j, means_of_features[j])
            )
        )
    
    return list_of_standard_deviations

def calculate_variance_of_feature(
        dataset: np.ndarray, 
        feature_idx: int, 
        mean_value: float
    ) -> float:
    """ Calculate the variance of a single feature in the given dataset

        Parameters:
        dataset (np.ndarray)
        feature_idx (int)

        Returns:
        float:Variance of selected feature
    """
    sum_of_squared_differences: float = 0

    for i in range(dataset.shape[0]):
        sum_of_squared_differences += np.square(dataset[i, feature_idx] - mean_value)
    
    return sum_of_squared_differences / dataset.shape[0]
    

def calculate_means_of_all_features(dataset: np.ndarray):
    """ Calculate the means of all features in the given dataset

        Parameters:
        dataset (np.ndarray)

        Returns:
        list:Means of all features
    """    
    list_of_means: list = []

    for j in range(dataset.shape[1]):
        list_of_means.append(calculate_mean_of_feature(dataset, j))
    
    return list_of_means

def calculate_mean_of_feature(dataset: np.ndarray, feature_idx: int) -> float:
    """ Calculate the mean of a single feature in the given dataset

        Parameters:
        dataset (np.ndarray)
        feature_idx (int)

        Returns:
        float:Mean of selected feature
    """
    sum_of_values: float = 0

    for i in range(dataset.shape[0]):
        sum_of_values += dataset[i, feature_idx]

    return sum_of_values / dataset.shape[0]