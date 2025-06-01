#!/usr/bin/env python3

import keras
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

def visualise_learn_history(
    history: keras.callbacks.History,
    model_name: str,
    dest_file: str = ''):
    """make an graph showing history of the accuracy and loss values evolution 

    Args:
        history (keras.callbacks.History): history object returned by model.fit
        model_name (str): model name
        dest_file (str, optional): image destination file. image wiould not be 
            shown if this is not empty. Defaults to ''.
    """
    history_dict = history.history

    plt.figure(figsize=(15,10))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], marker = 'o', label = 'Train Loss')
    plt.plot(history_dict['val_loss'], marker = 'o', label = 'Validation Loss')
    plt.title(f'{model_name} Loss', fontsize = 20)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_dict['accuracy'],marker = 'o', label = 'Train Accuracy')
    plt.plot(history_dict['val_accuracy'],marker = 'o', label = 'Validation Accuracy')
    plt.title(f'{model_name} Accuracy',fontsize=20)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if len(dest_file):
        plt.savefig(dest_file, dpi=300, transparent=True)
    else:
        plt.show()

def show_confusion_matrix(
    model_name: str,
    y_true: np.ndarray[int],
    y_pred: np.ndarray[int],
    class_names: tuple[str],
    dest_file: str = ''):
    """make a graphical representation of the confusion matrix

    Args:
        model_name (str): model name
        y_true (np.ndarray[int]): true values
        y_pred (np.ndarray[int]): predircted values
        class_names (tuple[str]): class anmes
        dest_file (str, optional): image destination file. image wiould not be 
            shown if this is not empty. Defaults to ''.
    """    
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.4g',
        center = True,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix (%)')
    plt.tight_layout()

    if len(dest_file):
        plt.savefig(dest_file, dpi=300, transparent=True)
    else:
        plt.show()

