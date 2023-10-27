import pytest
import great_expectations as ge

from catsdogs import data, config


@pytest.fixture(scope='module')
def df():
    raw_df = data.load_dataset(config.CATS_DIR, config.DOGS_DIR, n=10)
    return ge.dataset.PandasDataset(data.generate_labels(raw_df))
