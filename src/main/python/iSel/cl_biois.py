import numpy as np
from src.main.python.iSel.base import InstanceSelectionMixin
from src.main.python.iSel.biois import BIOIS

class CLBIOIS(InstanceSelectionMixin):
    """
    Curriculum Learning Wrapper for BIO-IS.
    Runs BIO-IS to perform instance selection, and then classifies the 
    REMAINING (selected) instances into easy, medium, and hard buckets 
    based on their entropy (proxy for difficulty).
    """

    def __init__(self, beta=0.25, theta=0.50, p_easy=50, p_med=80):
        """
        Parameters:
        ===========
        beta : float
            Reduction rate for BIO-IS
        theta : float
            Noise removal rate for BIO-IS
        p_easy : int
            Percentile boundary for easy examples (e.g., 50 means bottom 50% are easy)
        p_med : int
            Percentile boundary for medium examples (e.g., 80 means 50% to 80% are medium, above 80% are hard)
        """
        self.selector = BIOIS(beta=beta, theta=theta)
        self.p_easy = p_easy
        self.p_med = p_med

    def select_data(self, X, y):
        # 1. Run base BIO-IS
        X_sel, y_sel = self.selector.select_data(X, y)
        
        # Inherit attributes from base selector
        self.sample_indices_ = self.selector.sample_indices_
        self.entropy_ = getattr(self.selector, 'entropy_', None)
        self.reduction_ = getattr(self.selector, 'reduction_', 0.0)

        # 2. Curriculum Learning Difficulty Assignment
        if self.entropy_ is not None:
            # We only calculate percentiles based on the SELECTED instances
            # because the removed ones are already filtered out.
            # Low entropy = high confidence = easy.
            selected_entropy = self.entropy_[self.sample_indices_]
            
            p_easy_val = np.percentile(selected_entropy, self.p_easy)
            p_med_val = np.percentile(selected_entropy, self.p_med)

            # Assign labels parallel to sample_indices_
            self.difficulty_ = np.empty(len(self.sample_indices_), dtype=object)
            
            for i, ent in enumerate(selected_entropy):
                if ent < p_easy_val:
                    self.difficulty_[i] = 'easy'
                elif ent < p_med_val:
                    self.difficulty_[i] = 'medium'
                else:
                    self.difficulty_[i] = 'hard'
        else:
            self.difficulty_ = None

        return X_sel, y_sel
