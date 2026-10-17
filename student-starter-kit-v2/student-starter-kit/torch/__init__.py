import numpy as np

def tensor(data, *args, **kwargs):
    """Return a NumPy array mimicking torch.tensor.
    Only the minimal functionality required by the test suite is provided.
    """
    return np.array(data, *args, **kwargs)
