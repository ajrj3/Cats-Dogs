import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def plot_training_loss(history: tf.keras.callbacks.History,
                       save=False,
                       fname='') -> None:
    """Plots training loss from a call to tf.model.fit()

    Args:
        history (tf.keras.callbacks.History): history of training process
        save (bool ,optional): whether to save plot to file or not. Dafults to False.
        fname (str, optional): file name if save is True. Defaults to ''.
    """
    fig, axs = plt.subplots(2, 1)
    y_labels = ['loss', 'accuracy']

    for ax, y_label in zip(axs, y_labels):
        y1 = history.history[y_label]
        y2 = history.history['val_' + y_label]
        x = np.arange(1, len(y1)+1, 1)
        ax.plot(x, y1, 'r', label='training')
        ax.plot(x, y2, 'b', label='val')
        ax.legend(loc='best')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(y_label)

    fig.tight_layout()

    if save:
        fig.savefig(fname)
        
    plt.show()
    plt.close()

def plot_cm(y_pred: list, 
            y_true: list) -> None:
    """Plots confusion matrix given a set of predictions and corresponding true labels

    Args:
        y_pred (list): predicted labels
        y_true (list): true labels
    """
    confusion_mat = confusion_matrix(y_true, y_pred)
    sns.set(font_scale=1.4)
    sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='g', annot_kws={'size': 16})
    plt.show()
    plt.close()

def generate_eval_metrics(y_pred: list,
                          y_true: list,
                          label='train') -> dict():
    """For a given set of predictions, with corresponding ground truths, generate
    evaluation metrics such as accuracy, precision and recall

    Args:
        y_pred (list): predicted labels
        y_true (list): true labels

    Returns:
        Dict: dictionary of evaluation metrics with keys: ['accuracy', 'precision',
        'recall', 'f1_score']
    """
    label_types = ['train', 'val']
    assert label in label_types, f'Label must be one of {label_types}'

    metrics = dict()
    metrics[label+'_accuracy'] = sklearn.metrics.accuracy_score(y_true, y_pred)
    metrics[label+'_precision'] = sklearn.metrics.precision_score(y_true, y_pred)
    metrics[label+'_recall'] = sklearn.metrics.recall_score(y_true, y_pred)
    metrics[label+'_f1_score'] = sklearn.metrics.f1_score(y_true, y_pred)
    return metrics