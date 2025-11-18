import os
import numpy as np

def check_file( path):
    if os.path.exists(path):
        return True
    else:
        return False
    
def create_dir(path):
    if os.path.exists(path):
        return None
    else:
        os.mkdir(path)

def loc_mask(mask_img):
    ys, xs = np.where(mask_img>0)
    padding = 10
    x1, x2 = xs.min() - padding, xs.max() + padding
    y1, y2 = ys.min() - padding, ys.max() + padding
    return x1, x2, y1, y2