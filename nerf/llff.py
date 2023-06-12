from pathlib import Path
from typing import Optional
import numpy as np


def get_llff_scene_paths(llff_root: Path) -> list[Path]:
    return [d for d in llff_root.iterdir() if d.is_dir()]


def make_llff_train_test_split(num_images_in_scene: int, target_num_train_images: int):
    # Makes an LLFF train-test split following the standard convention of holding out every 8th image as test
    train_idxs = []
    test_idxs = []
    for idx in range(num_images_in_scene):
        if idx % 8 == 0:
            test_idxs.append(idx)
        else:
            train_idxs.append(idx)
    print(f'Made LLFF-style train set of {len(train_idxs)} images and test set of {len(test_idxs)} images')

    # Now evenly subsample the train set down to the desired number of images (if necessary)

    # regnerf-style
    idx_sub = np.linspace(0, len(train_idxs) - 1, target_num_train_images)
    train_idxs = [train_idxs[round(i)] for i in idx_sub]

    return train_idxs, test_idxs
