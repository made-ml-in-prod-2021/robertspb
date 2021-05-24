import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from typing import List, Optional, Tuple

from .datamodel import RequestDataModel, ResponseDataModel
from src.features.transformer import CustomTransformer


COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


def load_model_from_path(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_predictions(data: List[RequestDataModel],
                    classifier: Optional[LogisticRegression],
                    transformer: CustomTransformer
                    ) -> List[ResponseDataModel]:
    data = [record.__dict__ for record in data]
    df = pd.DataFrame(data)
    ids = df['id']
    features = df.drop('id', axis=1)
    train_x = transformer.transform(features)
    predictions = classifier.predict(train_x)
    return [ResponseDataModel(id=idx, target=prediction)
            for idx, prediction in zip(ids, predictions)]


def validate_data(record: RequestDataModel) -> Tuple[bool, str]:
    try:
        if not 0 <= record.age <= 120:
            raise ValueError(f'age must be in range 0 to 120')
        if record.sex not in [0, 1]:
            raise ValueError(f'sex value must be 0 or 1')
        if not 0 <= record.cp <= 3:
            raise ValueError(f'cp must be in range 0 to 3')
        if not 50 <= record.trestbps <= 600:
            raise ValueError(f'trestbps must be in range 50 to 600')
        if not 50 <= record.chol <= 1000:
            raise ValueError(f'chol must be in range 50 to 1000')
        if record.fbs not in [0, 1]:
            raise ValueError(f'fbs must be 0 or 1')
        if not 0 <= record.restecg <= 2:
            raise ValueError(f'restecg must be in range 0 to 2')
        if not 50 <= record.thalach <= 600:
            raise ValueError(f'thalach must be in range 50 to 600')
        if record.exang not in [0, 1]:
            raise ValueError(f'exang must be 0 or 1')
        if not 0 <= record.oldpeak <= 10:
            raise ValueError(f'oldpeak must be in range 0 to 10')
        if not 0 <= record.slope <= 2:
            raise ValueError(f'slope must be in range 0 to 2')
        if not 0 <= record.ca <= 4:
            raise ValueError(f'ca must be in range 0 to 4')
        if not 0 <= record.thal <= 3:
            raise ValueError(f'thal must be in range 0 to 3')
        return True, 'ok'
    except ValueError as error:
        return False, str(error)
