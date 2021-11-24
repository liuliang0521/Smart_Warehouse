import os
import glob
import numpy as np
from data.data_config import get_train_data
import shutil
if __name__ == "__main__":
    all_imgs = glob.glob(get_train_data() + "traindata/*.jpg")
    np.random.shuffle(all_imgs)
    k = 0
    for file in all_imgs:
        image_id = file.split("\\")[-1].split(".")[0]
        if k < 500:
            shutil.move(file, file.replace("traindata", "valdata"))
            shutil.move(file.replace(".jpg",".txt"), file.replace(".jpg",".txt").replace("traindata", "valdata"))
        k += 1
