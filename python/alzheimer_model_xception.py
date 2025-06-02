#!/usr/bin/env python3


import os, shutil
import cv2
import zipfile

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from tqdm import tqdm

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.utils import image_dataset_from_directory

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import brainy as brn


def create_model(class_num: int, summary: bool = False) -> keras.Model:
    inputs = Input(shape=(image_size[0], image_size[1], 3))
    base_model = tf.keras.applications.Xception(
        pooling = 'avg',
        weights = 'imagenet',
        include_top = False,
        input_tensor = inputs
    )
    base_model.trainable = False

    x = base_model.output

    x = keras.layers.Flatten()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(2048, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(4, activation='softmax')(x)

    model = keras.Model(inputs = inputs, outputs = outputs)

    if summary:
        model.summary()
    return model

def find_data_dir() -> str:
    ddir = '.'
    for d in ('data', '../data'):
        if os.path.isdir(d):
            ddir = d
            break
    return ddir
    
def find_model_dir() -> str:
    mdir = '.'
    for d in ('models', '../models'):
        if os.path.isdir(d):
            mdir = d
            break
    return mdir


if __name__ == '__main__':

    MODEL_NAME = 'Xception'
    SEED = 42
    image_size = (299, 299)
    batch_size = 25
    EPOCHS = 20

    dataset_dir = os.path.join(find_data_dir(), 'SmallPreprocessed')
    if brn.is_colab():
        dataset_dir = os.path.join(
            brn.unzip_directory(
                os.path.join(mount_gdrive('/content/drive')), 'smallpreprocessed.zip'),
            'data', 
            'SmallPreprocessed')

    class_names, train_ds, test_ds, val_ds = brn.create_data_sources(
        train_path = os.path.join(dataset_dir, 'train'),
        test_path = os.path.join(dataset_dir, 'test'),
        val_path = os.path.join(dataset_dir, 'val'),
        image_size = image_size,
        batch_size = batch_size,
        color_mode = 'rgb',
        preprocess = tf.keras.applications.xception.preprocess_input)

    model = create_model(len(class_names))
    
    hist = brn.fit_model(
        model = model,
        model_name = MODEL_NAME,
        train_ds = train_ds,
        val_ds = val_ds,
        epochs = EPOCHS)

    loss_score, acc = model.evaluate(test_ds)

    y_true, y_pred = brn.predict_data(model, test_ds)

    tmp_dir = '.'
    learn_curve_file = os.path.join(tmp_dir, f'{MODEL_NAME.lower()}_learn.png')
    confusion_matrix_file = os.path.join(tmp_dir, f'{MODEL_NAME.lower()}_confusion_matrix.png')

    brn.visualise_learn_history(
        history = hist,
        model_name = MODEL_NAME,
        dest_file = learn_curve_file)

    brn.show_confusion_matrix(
        model_name = MODEL_NAME,
        y_true = y_true,
        y_pred = y_pred,
        class_names = class_names,
        dest_file = confusion_matrix_file)

    ClassificationReport = classification_report(
        y_true,
        y_pred,
        target_names = class_names)
    
    print('Val Loss =', round(loss_score, 4))
    print('Val Accuracy =', round(acc, 4))
    print(f'Classification Report {MODEL_NAME}is : \n {ClassificationReport}')

    arch_path = brn.store_model(
        model = model,
        dst_path = find_model_dir(),
        model_name = MODEL_NAME,
        tmp_dir = '.',
        comment = (
            '',
            f'Loss score: {round(loss_score, 3)}',
            '',
            f'Accuracy  : {round(acc, 3)}',
            f'',
            "```",
            ClassificationReport,
            "```",
            f'',
            f'![learn curve]({MODEL_NAME.lower()}_learn.png)',
            f'![confusion matrix]({MODEL_NAME.lower()}_confusion_matrix.png)'
        ))

    brn.append_file_to_zip(learn_curve_file, arch_path)
    brn.append_file_to_zip(confusion_matrix_file, arch_path)

    os.remove(confusion_matrix_file)
    os.remove(learn_curve_file)
