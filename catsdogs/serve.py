import numpy as np
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request

from catsdogs.config import EXPERIMENT_NAME, mlflow
from catsdogs.model_fns import get_best_run_id

app = FastAPI()

@serve.deployment     # decorator that converts Python class to Ray's Deployment class
@serve.ingress(app)
class ModelDeployment:
    def __init__(self, run_id):
        self.run_id = run_id
        self.model_uri = 's3://cats-dogs-classifier/artifacts/1/{}/artifacts/model/MLmodel'.format(self.run_id)
        self.model = mlflow.tensorflow.load_model(self.model_uri)
    
    @app.get('/run_id/')
    def _run_id(self) -> dict:
        """Get run id

        Returns:
            dict: dict with single key; 'run_id'
        """
        return {'run_id': self.run_id}

    @app.post('/predict')
    async def _predict(self, request: Request) -> dict:
        arr = np.array((await request.json())['array'])
        arr = arr.reshape((1,200,200,3))
        result = self.model(arr)
        return {'result': result}
    
    
clf_app = ModelDeployment.bind(get_best_run_id(EXPERIMENT_NAME, 'accuracy', 'DESC'))