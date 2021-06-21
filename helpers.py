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
    cort_img = get_retinaBackProjected_GRAY(cortex)
    cort_img[cortex.left_hemi_size[0]:,:] = np.rot90(cort_img[cortex.left_hemi_size[0]:,:],2)
    return np.rot90(cort_img,1)

def get_cortexBackProjected_RGB(cortex):
    cort_img = get_retinaBackProjected_RGB(cortex)
    cort_img[cortex.left_hemi_size[0]:,:,:] = np.rot90(cort_img[cortex.left_hemi_size[0]:,:,:],2)
    return np.rot90(cort_img,1)

def get_cortexBackProjected_BGR(cortex):
    cort_img = get_retinaBackProjected_BGR(cortex)
    cort_img[cortex.left_hemi_size[0]:,:,:] = np.rot90(cort_img[cortex.left_hemi_size[0]:,:,:],2)
    return np.rot90(cort_img,1)

def resize(img, size):
    return cv2.resize(img, dsize=(size[1], size[0]), interpolation=cv2.INTER_CUBIC)