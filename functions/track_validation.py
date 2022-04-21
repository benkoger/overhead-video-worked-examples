import numpy as np
import os

def _has_validation(estimate_file):
    """ Does the estimate file have completed validation info. """
    ests = np.load(estimate_file)
    if np.all(ests != -1):
        return True
    return False


def get_observations_with_validations(validation_files):
    """ Return list of file names that contains humam validation information.
    
    Args:
        estimites_files: list of .npy files that record validation data
    """
    
    good_obs = set()
    for file in validation_files:
        if _has_validation(file):
            name = os.path.basename(file).split('-')[0]
            good_obs.add(name)
            
    return sorted(list(good_obs))   

def get_number_of_validations(validation_files):
    """ Return number of files that contains humam validation information.
    
    Args:
        estimites_files: list of .npy files that record validation data
        """
    
    num_validations = 0
    for file in validation_files:
        if _has_validation(file):
            num_validations += 1
            
    return num_validations