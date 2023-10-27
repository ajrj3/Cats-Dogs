import numpy as np
import pytest

from catsdogs.model_fns import train_model, train_model_distributed, probs_to_labels

@pytest.mark.training
def test_train_model():
    results = train_model(epochs=2)
    loss = results['loss']
    assert loss[1] < loss[0]

@pytest.mark.training
def test_train_model_distributed():
    results = train_model_distributed(epochs=2)
    loss = results.metrics['loss']
    assert loss[1] < loss[0]

def test_probs_to_labels():
    probs = np.array([0.8, 0.4, 0.9])
    labels = probs_to_labels(probs)
    assert np.array_equal(labels, np.array([1, 0, 1]))