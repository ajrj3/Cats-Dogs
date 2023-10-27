# Cats and Dogs Classifier

This personal project builds a simple classifier that determines whether an image is of a cat or a dog, when presented with one or the other.
The classifier uses a convolutional neural network architecture built in TensorFlow. 

The aim of this project was less-focused on building a high-performing classifier, as the problem is quite simple, but instead intends to develop and demonstrate good MLOps practices.

To this end, the repo showcases:
- Distributed model training using Ray Train
- Model hyperparameter optimisation using Ray Tune
- Experiment tracking using MLflow
- AWS integration:
  - EC2 for MLflow tracking server
  - RDS for MLflow run metrics
  - S3 bucket for artefacts
- Automated testing with Pytest
- Model deployment using Ray Serve and FastAPI
- CLI for script execution using Typer
- Complete training workloads executable with a single command, with automated logging of artefacts and test results to the cloud (AWS S3 bucket)

Data source:

https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset


