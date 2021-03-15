import ast
import json
import pandas as pd
from typing import List
from utils.features import Datapoint



def read_json_data(datapath: str) -> List[Datapoint]:
    with open(datapath) as f:
        datapoints = json.load(f)
        # [Datapoint.parse_obj((dict((k, int(float(v))) if "sms_length" in k else (k,v) for (k,v) in point.items()))) for point in datapoints]
        # [Datapoint.parse_obj(point) for point in datapoints]

        return [Datapoint.parse_obj(point) for point in datapoints]