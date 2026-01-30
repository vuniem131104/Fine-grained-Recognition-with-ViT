from __future__ import annotations

import os
from typing import Dict
from dotenv import load_dotenv

import kserve
import mlflow
import numpy as np
from kserve import Model

load_dotenv()

class BirdsModel(Model):

    def __init__(self, name: str = 'birds-model'):
        super().__init__(name)
        self.model = None
        self.ready = False

    def load(self) -> bool:
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME'))

        self.model = mlflow.pyfunc.load_model(
            model_uri=os.getenv('MODEL_URI'),
        )
        self.ready = True
        return self.ready

    def predict(
        self,
        payload: Dict,
        headers: Dict[str, str] = None,
    ) -> Dict:
        instances = payload.get('instances', [])
        if not instances:
            raise ValueError('No instances provided')

        input_tensor = np.array(instances, dtype=np.float32)

        predictions = self.model.predict(input_tensor)

        return {'predictions': predictions.tolist()}


def main():
    model = BirdsModel()
    model.load()

    kserve.ModelServer(
        http_port=int(os.getenv('HTTP_PORT')),
        grpc_port=int(os.getenv('GRPC_PORT')),
        workers=int(os.getenv('WORKERS')),
        enable_grpc=os.getenv('ENABLE_GRPC').lower() == 'true',
    ).start([model])
