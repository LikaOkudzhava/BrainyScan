#!/usr/bin/env python3

import os
import shutil
from tqdm import tqdm

def create_small_train_test_val_set(src_dir: str, dst_dir: str, splits: tuple[str, int]):
    for split, files in splits:
        for cl in os.listdir(os.path.join(src_dir, split)):
            images = os.listdir(os.path.join(src_dir, split, cl))
            for image in tqdm(images[:files], desc=f"copy small set of {split} {cl}"):

                if not os.path.isdir(os.path.join(dst_dir, split, cl)):
                    os.makedirs(os.path.join(dst_dir, split, cl))

                shutil.copy(
                    os.path.join(src_dir, split, cl, image),
                    os.path.join(dst_dir, split, cl, image)
                )

if __name__ == '__main__':

    src_dir = 'data/Preprocessed224'
    dst_dir = 'data/SmallPreprocessed224'

    splits = (
        ('test', 500),
        ('train', 3000),
        ('val', 500)
    )
    
    create_small_train_test_val_set(src_dir, dst_dir, splits)



