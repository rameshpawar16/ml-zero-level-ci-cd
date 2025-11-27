from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class EvenOddModel:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, max_number: int = 1000)-> None:
        x = np.arange(max_number).reshape(-1, 1)
        y = (x[:, 0] % 2).astype(int)
        self.model.fit(x, y)

    def predict(self, number: int) -> str:
        pred = self.model.predict([[number]])[0]
        if pred == 0:
            return "Even"
        return "Odd"


def train_and_save_model(model_path: str | Path = "model.pkl") -> Path:
    cls = EvenOddModel()
    cls.train()
    model_path = Path(model_path)
    joblib.dump(cls, model_path)
    return model_path
