import numpy as np
import pytest

from smsspam.model.boosting import LGBMModel
from smsspam.utils.features import Datapoint

@pytest.fixture
def config():
    return {
        "evaluate": False,
        "model_output_path": "",
        "featurizer_output_path": "",
        "params": {

        }
    }

@pytest.fixture
def sample_datapoints():
    return [
        Datapoint(
            target=True,
            sms="Lonely",
            sms_length=6        
        ),
        Datapoint(
            target=False,
            sms="WIn Lottery with these steps",
            sms_length=28
        ),
        Datapoint(
            target=True,
            sms="Hello",
            sms_length=5)
    ]

def text_lgbm_overfits_small_dataset(config, sample_datapoints):
    model = LGBMModel(config=config)
    train_labels = [True, False, True]

    model.train(sample_datapoints)
    predicted_labels = np.argmax(model.predict(sample_datapoints), axis=1)
    predicted_labels = list(map(lambda x: bool(x), predicted_labels))
    
    assert predicted_labels == train_labels

def test_lgbm_correct_predict_shape(config, sample_datapoints):
    model = LGBMModel(config=config)

    model.train(sample_datapoints)
    predicted_labels =np.argmax(model.predict(sample_datapoints), axis=1)
    
    assert predicted_labels.shape[0] == 3

def test_lgbm_corect_predict_range(config, sample_datapoints):
    model = LGBMModel(config=config)

    model.train(sample_datapoints)
    predicted_proba = model.predict(sample_datapoints)

    assert (predicted_proba <= 1).all()