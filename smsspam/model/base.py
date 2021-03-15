import numpy as np
from abc import ABC, abstractclassmethod
from typing import Dict, List, Optional

from utils.features import Datapoint


class Model(ABC):

    @abstractclassmethod
    def train(self, 
              train_datapoints: List[Datapoint],
              val_datapoints: List[Datapoint],
              cache_featurrizer: Optional[bool] = False) -> None:
        """
        performs Model training. implementations are model specific
        Return: None
        train_datapoints: list of train datapoints
        val_datapoints: list of validation datapoints
        cache_featurizer: (boolean) to cache model featurizer
        """
        pass

    @abstractclassmethod
    def predict(self, datapoints: List[Datapoint]) -> np.array:
        """
        performs inference on Model after training.
        return: numpy array of predictions
        datapoints: List of datapoints
        """
        pass

    @abstractclassmethod
    def compute_metrics(self, eval_datapoints: List[Datapoint]) -> Dict:
        """
        compute model sepcific metrics
        Return: Dict {metric: value}
        eval_datapoints: Datapoints to compute metircs for
        """
        pass

    @abstractclassmethod
    def get_params(self)-> Dict:
        """
        return params for specific model like number of tree of random forest, 
        hidden units incaes its a neural network 
        Return: Model Specific Paramaters
        """
        pass