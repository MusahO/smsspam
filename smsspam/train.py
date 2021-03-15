import os
import json
import random
import logging
import argparse
from shutil import copy

import torch
import mlflow
import numpy as np
import pandas as pd

from model.boosting import LGBMModel
from utils.reader import read_json_data
from utils.features import Datapoint

logging.basicConfig(format='%(levelname)s--%(asctime)s--%(filename)s--%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str)

    return parser.parse_args()

def set_random_seed(val: int=1) -> None:
    random.seed(val)
    np.random.seed(val)

if __name__ == "__main__":
    args = read_args()

    with open(args.config_file) as f:
        config = json.load(f)

    set_random_seed(42)
    mlflow.set_experiment(config['model'])

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_output_path = os.path.join(base_dir, config["model_output_path"])

    #update full model output path
    config["model_output_path"] = model_output_path
    os.makedirs(model_output_path, exist_ok=True)
    # config file -> model output dir
    copy(args.config_file, model_output_path)
    with mlflow.start_run() as run:
        with open(os.path.join(model_output_path, "meta.json"), "w") as f:
            json.dump({"mlflow_run_id": run.info.run_id}, f)
        mlflow.set_tags({
            "evaluate": config["evaluate"]
        })

        train_data_path = os.path.join(base_dir, config["train_data_path"])

        df = pd.read_csv(train_data_path, sep='\t', names=['target', 'sms', 'sms_length', 'kfold'], header=None, skiprows=1)
        df = df.sample(frac=1).reset_index(drop=True)

        df = df[df['sms'].notna()]

        for fold_ in range(5):
            train_df = df[df.kfold != fold_].reset_index(drop=True)
            val_df = df[df.kfold == fold_].reset_index(drop=True)

            train_df = train_df.drop(['kfold'], axis=1)
            val_df = val_df.drop(['kfold'], axis=1)

 
            train_df.to_json(f'train_df_{fold_}.json', orient="records")
            val_df.to_json(f'val_df_{fold_}.json', orient="records")
            

            train_datapoints = read_json_data(f'train_df_{fold_}.json')
            val_datapoints = read_json_data(f'val_df_{fold_}.json')


            if config["model"] == "lgbm":
                config["featurizer_output_path"] = os.path.join(base_dir, config["featurizer_output_path"])
                model = LGBMModel(config)
            elif config["model"] == "roberta":
                pass
            else:
                raise ValueError(f"Model type {config['model']} Provided is not Available")

            if not config["evaluate"]:
                logger.info("Training Model....")
                model.train(train_datapoints, val_datapoints, cache_featurizer=True)
                if config["model"] == "lgbm":
                    # cache model weights  to disk
                    model.save(os.path.join(model_output_path, "model.pkl"))

            mlflow.log_params(model.get_params())
            logger.info("Evaluating Model....")
            val_metrics = model.compute_metrics(val_datapoints)
            logger.info(f"validation metrics: {val_metrics}")
            mlflow.log_metrics(val_metrics)