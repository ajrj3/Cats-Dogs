import glob
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as random_shuffle

from catsdogs import config


def show_img(img_array: np.ndarray) -> None:
    """Displays an image

    Args:
        img_array (np.ndarray): RGB image array
    """
    plt.figure()
    plt.imshow(img_array)
    plt.show()

def preprocess_img(img_array: np.ndarray) -> np.ndarray:
    """Scales RGB channel values between 0 and 1

    Args:
        img_array (np.ndarray): array of RGB values with dims (x_size, y_size, 3)

    Returns:
        np.ndarray: array of RGB values with dims (x_size, y_size, 3). Each element
        has a value between 0 and 1
    """
    return img_array/255.0

def load_dataset(cat_dir: str,
                 dog_dir: str,
                 n: int=400,
                 img_size: tuple[int, int]=(200,200),
                 scale=True,
                 save=False,
                 fname=''
                 ) -> pd.DataFrame:
    """Loads all image data into a dataframe from raw image files

    Args:
        cat_dir (str): path to directory where cat image data is saved
        dog_dir (str): path to directory where dog image data is saved
        n (int): number of files to load from each dir. If None, returns all. Defaults to 400.
        img_size (tuple[int, int], optional): target image size. Defaults to (200, 200)
        scale (bool, optional): scale pixel values between 0 and 1. Defaults to True.
        save (bool, optional): save dataframe to file for easier loading in future. Defaults to False.
        fname (str, optional): filename of dataframe to be saved. Defaults to ''

    Returns:
        pd.DataFrame: 2-column data frame:
            1. Filename (str)
            2. Image data (np.ndarray)
    """
    cat_files = glob.glob(cat_dir + '/*')
    if n is not None:
        random.shuffle(cat_files)
        cat_files = cat_files[:n]
    cat_imgs = [load_img(img_file, target_size=(img_size[0], img_size[1])) for img_file in cat_files]
    cat_imgs = [img_to_array(img) for img in cat_imgs]
    cat_filenames = ['cat_' + Path(img_file).stem for img_file in cat_files]

    dog_files = glob.glob(dog_dir + '/*')
    if n is not None:
        random.shuffle(dog_files)
        dog_files = dog_files[:n]
    dog_imgs = [load_img(img_file, target_size=(img_size[0], img_size[1])) for img_file in dog_files]
    dog_imgs = [img_to_array(img) for img in dog_imgs]
    dog_filenames = ['dog_' + Path(img_file).stem for img_file in dog_files]

    if scale:
        cat_imgs = [preprocess_img(img) for img in cat_imgs]
        dog_imgs = [preprocess_img(img) for img in dog_imgs]
    
    df = pd.DataFrame({'Filename': cat_filenames + dog_filenames,
                       'Image': cat_imgs + dog_imgs})
    
    if save:
        fpath = config.DATA_DIR +'/' + fname + '.json'
        df.to_json(fpath)

    return df

def generate_labels(data_df: pd.DataFrame) -> pd.DataFrame:
    """Labels each image with its true label (dog or cat)

    Args:
        data_df (pd.DataFrame): 2-column data frame:
            1. Filename (str)
            2. Image data (np.ndarray)

    Returns:
        pd.DataFrame: 3-column data frame:
            1. Filename (str)
            2. Image data (np.ndarray)
            3. Cat (1 if cat, 0 if not - i.e. dog)
    """
    data_df['Cat'] = data_df['Filename'].map(lambda x: 1 if 'cat' in x else 0)
    return data_df

def stratify_split(data_df: pd.DataFrame,
                   test_size: float,
                   stratify: str='Cat',
                   shuffle: bool=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets with equal amounts from each class.

    Args:
        data_df (pd.DataFrame): input dataframe
        test_size (float): fraction of data that will form the test set
        stratify (str, optional): class (column) to stratify on. Defaults to 'Cat'.
        shuffle (bool, optional): shuffle dataset once split. Defaults to True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: train and test datasets
    """
    train, test = train_test_split(data_df, test_size=test_size, shuffle=shuffle, 
                                   stratify=data_df[stratify])

    if shuffle:
        train = random_shuffle(train)
        test = random_shuffle(test)
    
    return train, test

def to_numpyarray(series: pd.Series) -> np.ndarray:
    """Convert pd.Series of NumPy arrays to a multi-dimensional array. TensorFlow cannot
    handle the former.

    Args:
        series (pd.Series): series of np arrays (i.e. series of the image data)

    Returns:
        np.ndarray: multi-dimensional array of image data 
    """
    return np.array([np.array(val) for val in series])

def prepare_dataset(cats_dir: str=config.CATS_DIR,
                    dogs_dir: str=config.DOGS_DIR,
                    batch_size: int=32):
    """Prepare train and validation datasets for training

    Args:
        cats_dir (str, optional): path to directory where cat image data is saved. Defaults to config.CATS_DIR.
        dogs_dir (str, optional): path to directory where dog image data is saved. Defaults to config.DOGS_DIR.
        batch_size (int, optional): batch size during training. Defaults to 32.

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: train and validation datasets
    """
    df = load_dataset(cats_dir, dogs_dir) 
    df = generate_labels(df)       
    train, val = stratify_split(df, test_size=0.25)
    
    x_train = to_numpyarray(train['Image'])
    y_train = train['Cat']
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).batch(batch_size)
    
    x_val = to_numpyarray(val['Image'])
    y_val = val['Cat']
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (x_val, y_val)).batch(batch_size)
    
    return train_dataset, val_dataset
