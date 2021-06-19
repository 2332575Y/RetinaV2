import cv2
import pickle
import numpy as np

def loadPickle(fname):
    with open(fname, "rb") as f:
        temp = pickle.load(f)
    return temp

def savePickle(fname, data):
    with open(fname, "wb") as f:
            pickle.dump(data, f)

def get_retinaBackProjected_GRAY(retina):
    h, w = retina.size
    return retina.backProjectedVector[:h*w].reshape((h,w)).astype('uint8')
    
def get_retinaBackProjected_RGB(retina):
    h, w = retina.size
    R = retina.backProjectedVector[0, :h*w]
    G = retina.backProjectedVector[1, :h*w]
    B = retina.backProjectedVector[2, :h*w]
    return np.dstack([R,G,B]).reshape((h,w,3)).astype('uint8')

def get_retinaBackProjected_BGR(retina):
    h, w = retina.size
    R = retina.backProjectedVector[0, :h*w]
    G = retina.backProjectedVector[1, :h*w]
    B = retina.backProjectedVector[2, :h*w]
    return np.dstack([B,G,R]).reshape((h,w,3)).astype('uint8')

def get_cortexBackProjected_GRAY(cortex):
    Limg = get_retinaBackProjected_GRAY(cortex.left_hemi)
    Rimg = get_retinaBackProjected_GRAY(cortex.right_hemi)
    return np.concatenate((Limg, Rimg), axis=1)

def get_cortexBackProjected_RGB(cortex):
    Limg = get_retinaBackProjected_RGB(cortex.left_hemi)
    Rimg = get_retinaBackProjected_RGB(cortex.right_hemi)
    return np.concatenate((Limg, Rimg), axis=1)

def get_cortexBackProjected_BGR(cortex):
    Limg = get_retinaBackProjected_BGR(cortex.left_hemi)
    Rimg = get_retinaBackProjected_BGR(cortex.right_hemi)
    return np.concatenate((Limg, Rimg), axis=1)

def resize(img, size):
    return cv2.resize(img, dsize=(size[1], size[0]), interpolation=cv2.INTER_CUBIC)