from abc import ABC, abstractmethod
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split

class Evaluator(ABC):
    """
    Base class for all evaluators
    Extend this class and implement __call__ for custom evaluators.
    """
    def __init__(self, seed=42, **kwargs):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    @abstractmethod
    def __call__(self, model):
        """
        This is called during training to evaluate the model.
        It returns scores.

        Parameters
        ----------
        model:
            the model to evaluate
        """
        pass

    def _matched_random_sample(self, *arrays, size):
        """
        Do a random sample without replacement
        """
        if size >= len(arrays[0]):
            return arrays # do nothing
        splits = train_test_split(*arrays, train_size=size, random_state=self.seed)
        return [x for i, x in enumerate(splits) if i % 2 == 0]