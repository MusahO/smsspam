import os
import csv
import json
import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from typing import Dict, List

from clean import clean_text

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--output-dir", type=str)

    return parser.parse_args()


def read_datapoints(FILE_PATH: str) -> List[Dict]:
    with open(FILE_PATH) as f:
        reader = csv.DictReader(f, delimiter='\t', fieldnames=['target', 'sms'])

        return [row for row in reader]

if __name__ == "__main__":
    args = read_args()
    data = read_datapoints(args.data_path)
    df = pd.DataFrame(data)

    df['sms'] = df['sms'].apply(lambda x: clean_text(x))
    df['sms_length'] = df['sms'].apply(len)
    df['target'] = df['target'].replace({'ham':True, 'spam':False})
    
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    kf = StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=df.target)):
        df.loc[v_, 'kfold'] = f

    df.to_csv(os.path.join(args.output_dir, 'cleaned_data.csv'), index=False, sep='\t')