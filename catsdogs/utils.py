import random

import numpy as np
import tensorflow as tf


def set_seed(seed: int=1234):
    """Sets project-wide seeds

    Args:
        seed (int, optional): seed. Defaults to 1234.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)