#!/usr/bin/env python3

import os
import tensorflow as tf

def create_data_sources(
        train_path: str, test_path: str, val_path: str,
        image_size: tuple[int, int],
        batch_size: int,
        color_mode: str,
        preprocess
    ) -> tuple[list[str], tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """create tf.data.Dataset for train, test, val data based on the directories.

    Args:
        train_path (str): train data path. should have one subdirectory for each class
        test_path (str): test data path. should have one subdirectory for each class
        val_path (str): val data path. should have one subdirectory for each class
        image_size (tuple[int, int]): image size to use
        batch_size (int): size of the batch
        color_mode (str): color mode
        preprocess (_type_): function to use as preprocessing fucntion

    Returns:
        tuple[list[str], tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: 
            tulple with (class_names, train, test, val)
    """    
    class_names = sorted(os.listdir(train_path))
    
    res = [class_names]
    for folder, shuffle in ((train_path, True), (test_path, False), (val_path, True)):
        tmp_ds = tf.keras.utils.image_dataset_from_directory(
            folder,
            class_names = class_names,
            labels='inferred',
            label_mode='categorical',
            image_size=image_size,
            batch_size=batch_size,
            color_mode=color_mode,
            shuffle=shuffle
        )
        tmp_ds = tmp_ds.map(lambda img, lbl: (preprocess(img), lbl))
        res.append(tmp_ds)
    
    return res


if __name__ == '__main__':
    pass