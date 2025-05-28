#!/usr/bin/env python3

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_train_test_env_generators(
    img_generator: tensorflow.keras.preprocessing.image.ImageDatagenerator,
    dataset_dir: str,
    tgt_size: int,
    batch_size: int) -> tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
    
    return (
        img_generator.flow_from_directory(
            os.path.join(dataset_dir, 'train'),
            target_size=(tgt_size, tgt_size),
            batch_size=batch_size,
            class_mode='categorical'
        ),
        img_generator.flow_from_directory(
            os.path.join(dataset_dir, 'test'),
            target_size=(tgt_size, tgt_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        img_generator.flow_from_directory(
            os.path.join(dataset_dir, 'val'),
            target_size=(tgt_size, tgt_size),
            batch_size=batch_size,
            class_mode='categorical'
        ),
    )
