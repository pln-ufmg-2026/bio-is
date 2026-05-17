import numpy as np
from src.main.python.iSel.base import InstanceSelectionMixin

class NoSel(InstanceSelectionMixin):
    """
    A No-Op instance selection method that returns the original dataset without modifications.
    """
    def __init__(self):
        pass

    def select_data(self, X, y):
        self.sample_indices_ = np.arange(len(y))
        return X, y
