import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from typing import List, Optional

from .datamodel import RequestDataModel, ResponseDataModel
from src.features.transformer import CustomTransformer


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
    features = edit_data(df.drop('id', axis=1))
    train_x = transformer.transform(features)
    predictions = classifier.predict(train_x)
    return [ResponseDataModel(id=idx, target=prediction)
            for idx, prediction in zip(ids, predictions)]


def edit_data(data: pd.DataFrame, target: str = 'target') -> pd.DataFrame:
    categorical_values = []
    for column in data.columns:
        if len(data[column].unique()) <= 10 and column != target:
            categorical_values.append(column)
    dataset = pd.get_dummies(data, columns=categorical_values)
    return dataset
