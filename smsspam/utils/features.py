import os
import sys
import json
import pickle
import logging
import numpy as np
from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional

import numpy as np
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from pydantic import validator
from pydantic import BaseModel


logging.basicConfig(format='%(levelname)s--%(asctime)s--%(filename)s--%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Datapoint(BaseModel):
    target: Optional[bool]
    sms: str
    sms_length: int
    # kfold: int

    @validator('sms_length', pre=True)
    def name_must_be_str(cls, v):
        if type(v) is not int:
            # print(type(v))
            # print(v + '❤❤❤😘😘😘😘❤❤❤😘😘🆕📝📝⚠⚠⚠')
            raise TypeError("'sms_length' must be int, not " + type(v).__name__)
        return v

def extract_sms(datapoints: List[Datapoint]) -> List[str]:
    return [datapoint.sms for datapoint in datapoints]

def extract_manual_features(datapoints: List[Datapoint]) -> List[Dict]:
    all_features = []
    for datapoint in datapoints:
        features = {}
        features['sms_length'] = datapoint.sms_length
        all_features.append(features)
    
    return all_features


class LGBMFeaturizer(object):
    def __init__(self, featurizer_cache_path: str, config: Optional[Dict]=None):
        "you can add feature caching if the expense of computing features from scratch is greater"
        if os.path.exists(featurizer_cache_path):
            logger.info("Loading featurizer from cache...")
            with open(featurizer_cache_path, 'rb') as f:
                self.combined_featurizer = pickle.load(f)
        
        else:
            logger.info("Creating Featurizer from scratch...")
            
            dict_featurizer = DictVectorizer()
            tfidf_featurizer = TfidfVectorizer()

            sms_transformer = FunctionTransformer(extract_sms)
            manual_feature_transformer = FunctionTransformer(extract_manual_features)

            manual_feature_pipeline = Pipeline([
                ("manual_features", manual_feature_transformer),
                ('manual_featurizer', dict_featurizer)
            ])

            ngram_feature_pipeline = Pipeline([
                ('sms', sms_transformer),
                ('ngram_featurizer', tfidf_featurizer)
            ])

            self.combined_featurizer = FeatureUnion([
                ('manual_feature_pipeline', manual_feature_pipeline),
                ('ngram_feature_pipeline', ngram_feature_pipeline)
            ])
    
    def get_all_feature_names(self) -> List[str]:
        all_feature_names = []
        
        for _, pipeline in self.combined_featurizer.transformer_list:
            _ , final_pipe_transformer = pipeline.steps[-1]
            all_feature_names.extend(final_pipe_transformer.get_feature_names())
        return all_feature_names

    def fit(self, datapoints: List[Datapoint]) -> None:
        self.combined_featurizer.fit(datapoints)

    def featurizer(self, datapoints: List[Datapoint]) -> np.array:
        return self.combined_featurizer.transform(datapoints)

    def save(self, featurizer_cache_path: str):
        logger.info("Saving Featurizer to disk...")
        with open(featurizer_cache_path, "wb") as f:
            pickle.dump(self.combined_featurizer, f)