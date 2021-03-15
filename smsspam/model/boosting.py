import os
import json
import pickle
import logging
from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional

import numpy as np
from lightgbm import  LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from model.base import Model
from utils.features import Datapoint, LGBMFeaturizer

logging.basicConfig(format='%(levelname)s--%(asctime)s--%(filename)s--%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LGBMModel(Model):
    def __init__(self, config: Optional[Dict]=None):
        self.config = config
        model_cache_path = os.path.join(config["model_output_path"], "model.pkl")
        self.featurizer = LGBMFeaturizer(os.path.join(config['featurizer_output_path'], 'featurizer.pkl'), config)

        if "evaluate" in config and config["evaluate"] and not os.path.exists(model_cache_path):
            raise ValueError("Non Existant Model output path in Evaluation Mode!")

        if model_cache_path and os.path.exists(model_cache_path):
            logger.info("Loading Model from Cache")
            with open(model_cache_path, "rb") as f:
                self.model = pickle.load(f)

        else: 
            logger.info("Initializing Model from scratch....")
            self.model = LGBMClassifier(**self.config['params'])

    def train(self, 
              train_datapoints: List[Datapoint], 
              val_datapoints: List[Datapoint], 
              cache_featurizer: Optional[bool]=False)->None:
        self.featurizer.fit(train_datapoints)
        
        # caching(if True) we don't have go through more featurizing steps in features/LGBMFeaturizer
        if cache_featurizer:
            feature_names = self.featurizer.get_all_feature_names()
            with open(os.path.join(self.config['featurizer_output_path'], "feature_names.pkl"), "wb") as f:
                pickle.dump(feature_names, f)
            self.featurizer.save(os.path.join(self.config["featurizer_output_path"], "featurizer.pkl"))
        
        logger.info("Featurizing From Scratch")
        train_features = self.featurizer.featurizer(train_datapoints) #transform

        targets = [datapoint.target for datapoint in train_datapoints]
        
        self.model.fit(train_features, targets)

    def compute_metrics(self, eval_datapoints: List[Datapoint]) -> Dict:
        expected_labels = [datapoint.target for datapoint in eval_datapoints]
        predicted_proba = self.predict(eval_datapoints)
        predicted_labels = np.argmax(predicted_proba, axis=1)
        
        accuracy = accuracy_score(expected_labels, predicted_labels)
        f1 = f1_score(expected_labels, predicted_labels)
        auc = roc_auc_score(expected_labels, predicted_labels)
        confusion_matrix_ = confusion_matrix(expected_labels, predicted_labels)
        tn, fp, fn, tp = confusion_matrix_.ravel()
        
        return {
            "Accuracy": accuracy,
            "f1": f1,
            "AUC": auc,
            "True Negative": tn,
            "False Positive": fp,
            "False Negative": fn,
            "True Positive": tp
        }

    def predict(self, datapoints: List[Datapoint]) -> np.array:
        features = self.featurizer.featurizer(datapoints)
        return self.model.predict_proba(features)

    def get_params(self) -> Dict:
        return self.model.get_params()
    
    def save(self, model_cache_path: str) -> None:
        logger.info("Saving Model To Disk")
        with open(model_cache_path, "wb") as f:
            pickle.dump(self.model, f)