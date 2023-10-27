import mlflow

PROJECT_DIR = '/home/user/Training/Cats_Dogs'
DATA_DIR = PROJECT_DIR + '/data'
CATS_DIR = DATA_DIR + '/Cat'
DOGS_DIR = DATA_DIR + '/Dog'

# mlflow config remote
TRACKING_SERVER_HOST = 'ec2-16-171-206-244.eu-north-1.compute.amazonaws.com'
MLFLOW_TRACKING_URI = f'http://{TRACKING_SERVER_HOST}:5000'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = 'Cats Dogs Classification'
mlflow.set_experiment(EXPERIMENT_NAME)