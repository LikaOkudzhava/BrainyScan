#!/usr/bin/env python3

import os, shutil
from datetime import datetime

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

from .colab import is_colab, mount_gdrive
from .ziply import zip_directory, unzip_directory

def fit_model(
    model: keras.Model,
    model_name: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int) -> keras.callbacks.History:
    """fit model. early stopping will be used

    Args:
        model (keras.Model): model to fit
        model_name (str): name of the model
        train_ds (tf.data.Dataset): train data source
        val_ds (tf.data.Dataset): val;idation data source
        epochs (int): max number of epochs

    Returns:
        keras.callbacks.History: history object
    """
    checkpoint_cb = ModelCheckpoint(
        f"model_{model_name}_checkpoint.keras",
        save_best_only = True)

    early_stopping_cb = EarlyStopping(
        patience = 10,
        restore_best_weights = True
    )

    model.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    hist = model.fit(
        train_ds,
        epochs = epochs,
        validation_data = val_ds,
        callbacks = [ checkpoint_cb, early_stopping_cb ]
    )
    return hist


def store_model(
    model: keras.Model,
    dst_path: str,
    model_name: str,
    tmp_dir: str = '.',
    comment: tuple[str] = ()) -> str:
    """ store the model in zip archive. adding necessary information

    Args:
        model (keras.Model): model, which data should be stored
        dst_path (str): destination directory
        model_name (str): model name. the archive will have it's name
        tmp_dir (str, optional): directory to use as temporary. Defaults to '.'.
        comment (tuple[str], optional): list of the comment lines in markdown format. Defaults to ().

    Returns:
        str: path to the archive file
    """
    tmp_model_dir = os.path.join(tmp_dir, model_name.lower())
    os.makedirs(tmp_model_dir, exist_ok=True)

    model.save(os.path.join(tmp_model_dir, f'{model_name}_model_fittable.keras'))
    model.export(os.path.join(tmp_model_dir, f'{model_name}_model_prediction_only'))

    with open(os.path.join(tmp_model_dir, 'README.md'), 'w') as file:
        lines = [
            f'# {model_name}',
            f'',
            f'#### Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'',
            f'Contents of this array allow to recover model state using tf.keras.models.load_model',
            f'',
            f'- {model_name}_model_fittable.keras - can be used to restore complete model, capable to fit',
            f'- {model_name}_model_precition_only - can be used to restore model, capable to predict only',
            '',
        ]
        lines.extend(comment)
        file.write('\n'.join(lines) + '\n')

    zip_directory(tmp_model_dir, os.path.join(dst_path, f'{model_name.lower()}.zip'))

    shutil.rmtree(tmp_model_dir, ignore_errors=True)

    return os.path.join(dst_path, f'{model_name.lower()}.zip')

def load_fittable_model(
    src_path: str,
    model_name: str = 'dummy',
    tmp_dir: str = '.'):
    """load model ready-to-fit state from previosly saved archive. model can be used 
        for prediction as well

    Args:
        src_path (str): model archive
        model_name (str, optional): model name. Defaults to 'dummy'.
        tmp_dir (str, optional): directory to be used as temporary. Defaults to '.'.

    Returns:
        _type_: None
    """

    tmp_model_dir = os.path.join(tmp_dir, model_name.lower())
    os.makedirs(tmp_model_dir, exist_ok=True)

    if os.path.isdir(src_path) and os.path.isfile(os.path.join(src_path, f'{model_name.lower()}.zip')):
        src_path = os.path.join(src_path, f'{model_name.lower()}.zip')

    extracted_dir = unzip_directory(src_path, tmp_model_dir)

    model_url = os.path.join(extracted_dir, f'{model_name}_model_fittable.keras')
    if not os.path.isfile(model_url):
        model_url = None
        for fname in os.listdir(path = extracted_dir):
            fname = os.path.join(tmp_model_dir, fname)
            if os.path.isfile(fname) and '_model_fittable.keras' in fname:
                model_url = fname
                break
    
    
    model = tf.keras.models.load_model(model_url)
    
    shutil.rmtree(tmp_model_dir, ignore_errors=True)

    return model


def predict_data(model: keras.Model, test_ds: tf.data.Dataset) -> tuple[np.ndarray[int], np.ndarray[int]]:
    """do a prediction for imput data

    Args:
        model (keras.Model): model to use
        test_ds (tf.data.Dataset): input data to make a prediction

    Returns:
        tuple[np.ndarray[int], np.ndarray[int]]: tuple (true, predicted) values
    """

    y_true = []
    y_pred = []
    for images, labels in tqdm(test_ds, desc="predict"):
        preds = model.predict(images, verbose = 0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis = 1))
    
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    return (y_true, y_pred)