import os
import random
from glob import glob

import numpy as np
from tqdm import tqdm


def copy_image_to_processed_dir(alpha_list, img_dir, desc):
    for alpha in tqdm(alpha_list, desc=desc):
        write_dir1 = img_dir + '/' + os.path.basename(alpha) + '_'
        for char in (os.listdir(alpha)):
            write_dir2 = (write_dir1 + char)
            char_path = os.path.join(alpha, char)
            os.makedirs(write_dir2)
            for drawer in os.listdir(char_path):
                drawer_path = os.path.join(char_path, drawer)
                os.rename(drawer_path, os.path.join(write_dir2, drawer))


def prepare_data():
    background_dir = "data/unzip/background"
    evaluation_dir = "data/unzip/evaluation"
    processed_dir = "data/processed"
    random.seed(5)

    if os.path.exists(processed_dir) is None:
        os.makedirs(processed_dir)

    # Move 10 of evaluation image for getting more train set.
    if len(glob(evaluation_dir + '/*')) >= 20:
        for d in random.sample(glob(evaluation_dir + '/*'), 10):
            os.rename(d, os.path.join(background_dir, os.path.basename(d)))

    back_alpha = [x for x in glob(background_dir + '/*')]
    back_alpha.sort()

    # Split background data into train, validation data and make test data
    train_alpha = list(np.random.choice(back_alpha, size=30, replace=False))
    val_alpha = [x for x in back_alpha if x not in train_alpha]
    test_alpha = [x for x in glob(evaluation_dir + '/*')]
    test_alpha.sort()

    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')
    test_dir = os.path.join(processed_dir, 'test')

    copy_image_to_processed_dir(train_alpha, train_dir, 'train')
    copy_image_to_processed_dir(val_alpha, val_dir, 'val')
    copy_image_to_processed_dir(test_alpha, test_dir, 'test')
