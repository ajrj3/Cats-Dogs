import json
import os
import tempfile

import mlflow
import mlflow.keras
import numpy as np
import ray
import tensorflow as tf
import typer
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.air.integrations.mlflow import setup_mlflow
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.tensorflow import TensorflowTrainer
from typing_extensions import Annotated

#from catsdogs 
import config, data, utils

app = typer.Typer()

def create_model(layer1_nodes: int = 32,
                 layer2_nodes: int = None):
    """Specify and create model

    Args:
        layer1_nodes (int, optional): number of filters in first conv layer. Defaults to 32.
        layer2_nodes (int, optional): number of filters in second conv layer. Defaults to None.

    Returns:
        tf.keras.Model: CNN model
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(layer1_nodes, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    if layer2_nodes is not None:
        model.add(tf.keras.layers.Conv2D(layer2_nodes, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def compile_model(model: tf.keras.Model) -> None:
    """Compiles model

    Args:
        model (tf.keras.Model): model to be compiled
    """
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

@app.command()
def train_model(results_filepath: Annotated[str, typer.Option(help='Path to json file where results are to be saved')] = None,
                cats_dir: Annotated[str, typer.Option(help='Directory where cat images are saved')] = config.CATS_DIR,
                dogs_dir: Annotated[str, typer.Option(help='Directory where dog images are saved')] = config.DOGS_DIR,
                epochs: Annotated[int, typer.Option(help='Number of training epochs')] = 3,
                batch_size: Annotated[int, typer.Option(help='Batch size during training')] = 32) -> dict:
    """Performs model training, tracked with mlflow, training not distributed

    Args:
        results_filepath (str, optional): Path to json file where results are to be saved. Defaults to None.
        cats_dir (str, optional): path to directory where cat images are saved. Defaults to config.CATS_DIR.
        dogs_dir (str, optional): path to directory where dog images are saved. Defaults to config.DOGS_DIR.
        epochs (int, optional): number of epochs during training. Defaults to 3.
        batch_size (int, optional): batch size during training. Defaults to 32.

    Returns:
        dict: training and validation set loss and performance metrics
    """
    # set seeds
    utils.set_seed()

    # read-in and prepare data
    train_dataset, val_dataset = data.prepare_dataset(cats_dir, dogs_dir, batch_size)
    input_example = list(train_dataset.as_numpy_iterator())[0][0]

    # create model and fit it to the data
    model = create_model()
    compile_model(model)
    history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)

    train_metrics = {'train_loss': history.history['loss'][-1],
                     'train_accuracy': history.history['accuracy'][-1]}
    val_metrics = {'val_loss': history.history['val_loss'][-1],
                   'val_accuracy': history.history['val_accuracy'][-1]}
    
    # track experiment and log artefacts using mlflow
    mlflow.set_experiment(experiment_name=config.EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_param('epochs', epochs)
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(val_metrics)
        mlflow.keras.log_model(model, 'model', input_example=input_example)

    results = history.history

    if results_filepath is not None:
        with open(results_filepath, 'w') as f:
            json.dump(results, f)

    return results

def train_func(train_loop_config: dict):
    """Training function that each worker will execute (for distributed training).

    Args:
        config (dict): arguments to use for training.
    """
    # set seeds
    utils.set_seed()
    per_worker_batch_size = train_loop_config.get('batch_size', 32)
    epochs = train_loop_config.get('epochs', 3)
    layer1_nodes = train_loop_config.get('layer1_nodes', 32)
    layer2_nodes = train_loop_config.get('layer2_nodes', None)

    # this environment variable will be set by Ray Train
    tf_config = json.loads(os.environ['TF_CONFIG'])
    num_workers = len(tf_config['cluster']['worker'])

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    global_batch_size = per_worker_batch_size * num_workers

    # read-in and prepare data
    train_dataset, val_dataset = data.prepare_dataset(batch_size=global_batch_size)

    with strategy.scope():
        # model building/compiling needs to be within `strategy.scope()`.
        multi_worker_model = create_model(layer1_nodes=layer1_nodes,
                                          layer2_nodes=layer2_nodes)
        compile_model(multi_worker_model)
    
    for epoch in range(epochs):
        history = multi_worker_model.fit(train_dataset, 
                                         validation_data=val_dataset,
                                         callbacks=[ReportCheckpointCallback()])
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            ray.train.report(history.history, checkpoint=checkpoint)
    
    # track experiment and log artefacts using mlflow
    train_metrics = {'train_loss': history.history['loss'][-1],
                     'train_accuracy': history.history['accuracy'][-1]}
    val_metrics = {'val_loss': history.history['val_loss'][-1],
                   'val_accuracy': history.history['val_accuracy'][-1]}
    mlflow_ray = setup_mlflow(train_loop_config,
                              tracking_uri=config.MLFLOW_TRACKING_URI,
                              experiment_name=config.EXPERIMENT_NAME)
    mlflow_ray.log_param('epochs', epochs)
    mlflow_ray.log_metrics(train_metrics)
    mlflow_ray.log_metrics(val_metrics)

@app.command()
def train_model_distributed(results_filepath: Annotated[str, typer.Option(help='Path to json file where results are to be saved')] = None,
                            epochs: Annotated[int, typer.Option(help='Number of training epochs')] = 3,
                            batch_size: Annotated[int, typer.Option(help='Batch size during training')] = 32):
    """Performs model training in a distributed fashion, tracked with mlflow.

    Args:
        results_filepath (str, optional): path to json file where results are to be saved. Defaults to None.
        epochs (int, optional): number of training epochs. Defaults to 3.
        batch_size (int, optional): batch size during training. Defaults to 32.

    Returns:
        Result: stores train and validation set accuracy and loss metrics
    """
    train_loop_config = {'batch_size': batch_size,
                         'epochs': epochs}
    
    trainer = TensorflowTrainer(train_loop_per_worker=train_func,
                                train_loop_config=train_loop_config,
                                scaling_config=ScalingConfig(num_workers=2, use_gpu=False))
    
    results = trainer.fit()

    if results_filepath is not None:
        with open(results_filepath, 'w') as f:
            json.dump(results.metrics, f)

    return results

def make_pred(imgs: list,
              model: tf.keras.Model) -> np.ndarray:
    """Makes class prediction given input image(s)

    Args:
        imgs (list): list where each element is an np array of image data
        model (tf.keras.Model): trained model

    Returns:
        np.ndarray: class probability (between 0 and 1)
    """
    y_pred = model.predict(imgs)
    return y_pred

def probs_to_labels(probs: np.ndarray,
                    threshold: float=0.5) -> np.ndarray:
    """Converts prediction probabilities to labels

    Args:
        probs (np.ndarray): class probability (between 0 and 1)
        threshold (float, optional): threshold probability for class label. Defaults to 0.5.

    Returns:
        np.ndarray: predicted class labels (1 or 0)
    """
    return np.where(probs > threshold, 1, 0)

@app.command()
def get_best_run_id(experiment_name: Annotated[str, typer.Option(help="Name of experiment")] = "Cats Dogs Classification", 
                    metric: Annotated[str, typer.Option(help="Metric to filter by")] = "val_accuracy",
                    mode: Annotated[str, typer.Option(help="Direction of metric (ASC/DESC)")] = "DESC") -> str:
    """Get the best run_id from an MLflow experiment.

    Args:
        experiment_name (str): name of the experiment.
        metric (str): metric to filter by.
        mode (str): direction of metric (ASC/DESC).

    Returns:
        str: best run id from experiment.
    """
    sorted_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f'metrics.{metric}{mode}'],
    )
    run_id = sorted_runs.iloc[0].run_id
    return run_id

if __name__ == "__main__":
    ray.init()
    app()