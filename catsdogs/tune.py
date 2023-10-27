import ray
from ray import tune
from ray.tune.tune_config import TuneConfig
from ray.train import ScalingConfig
from ray.train.tensorflow import TensorflowTrainer

from catsdogs.model_fns import train_func


def tune_hyperparams():
    trainer = TensorflowTrainer(train_loop_per_worker=train_func,
                                scaling_config=ScalingConfig(num_workers=2, use_gpu=False))

    tuner = tune.Tuner(trainable=trainer,
                       tune_config=TuneConfig(num_samples=1, metric='accuracy', mode='max'),
                       param_space = {
                           # train_func only takes train_loop_config as an arg, hence this is how param_space should be defined
                           'train_loop_config': {
                               'batch_size': tune.choice([32, 64]), 
                               'epochs': 3,
                               'layer1_nodes': tune.choice([32, 64]),
                               'layer2_nodes': tune.choice([None, 64, 128])
                            }
                        }
    )

    results = tuner.fit()
    best_result = results.get_best_result()  # Get best result object
    best_config = best_result.config  # Get best trial's hyperparameters
    print(best_result)
    print(best_config)

if __name__ == "__main__":
    ray.init()
    tune_hyperparams()