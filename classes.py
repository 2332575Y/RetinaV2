import numpy as np
from helpers import *
from functions import *

def loadConfig():
    try:
        globals()['types'] = loadPickle('config.pkl')
        return 0
    except:
        return -1

if loadConfig()!=0:
    raise Exception('Failed to load config.pkl! Please make sure it exists.')

#===============================================================================================================================
#===============================================================================================================================
#                                         ██████╗ ███████╗████████╗██╗███╗   ██╗ █████╗ 
#                                         ██╔══██╗██╔════╝╚══██╔══╝██║████╗  ██║██╔══██╗
#                                         ██████╔╝█████╗     ██║   ██║██╔██╗ ██║███████║
#                                         ██╔══██╗██╔══╝     ██║   ██║██║╚██╗██║██╔══██║
#                                         ██║  ██║███████╗   ██║   ██║██║ ╚████║██║  ██║
#                                         ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝
#===============================================================================================================================
#===============================================================================================================================

class Retina:
    def __init__(self, fname):
        self.backProjectedVector = None
        self.normalizationVector = None
        self.normalizationImage = None
        self.input_resolution = None
        self.scalingFactor = None
        self.sampledVector = None
        self.coeff_array = None
        self.index_array = None
        self.size_array = None
        self.kernel_map = None
        self.crop_coords = None
        self.ret_coords = None
        self.backProject = None
        self.getResult = None
        self.fixation = None
        self.sample = None
        self.Arrays = None
        self.fname = fname
        self.shape = None
        self.size = None
        self.RGB = None

        try:
            self.loadArrays()
            self.shape = np.copy(self.size)
            self.createNormImg()
        except:
            raise Exception('Could not find previously saved arrays! Please make sure to initlaize them.')

    def loadArrays(self):
        if self.Arrays is None:
            self.Arrays = loadPickle(self.fname)
        self.size,self.scalingFactor,self.coeff_array,self.index_array,self.size_array,self.kernel_map = self.Arrays

    def setInputResolution(self, w, h):
        self.input_resolution = np.array([h,w], dtype='int32')
        
    def createNormImg(self):
        ones = np.ones(self.size[0]*self.size[1],dtype=types['INPUT'])
        self.sampledVector = np.zeros(len(self.size_array), dtype=types['RESULT'])
        sample(ones,self.coeff_array,self.sampledVector,self.size_array,self.index_array)
        self.backProjectedVector = np.zeros(self.size[0]*self.size[1], dtype=types['BACK_PROJECTED'])
        backProject(self.sampledVector, self.coeff_array, self.backProjectedVector, self.size_array, self.index_array)
        self.normalizationImage = np.copy(self.backProjectedVector)
        self.normalizationImage = self.normalizationImage.reshape(self.size)

    ########################################
    ############### FIXATION ###############
    ########################################

    def setFixation(self, x, y):
        self.fixation = np.array([x,y], dtype='int32')
        img_x1, img_y1, img_x2, img_y2, ret_x1, ret_y1, ret_x2, ret_y2 = get_bounds(self.input_resolution, self.shape, self.fixation)
        new_ret_coords = (ret_y1,ret_y2,ret_x1,ret_x2)
        self.crop_coords = (img_x1, img_y1, img_x2, img_y2)
        new_size = np.array([img_y2 - img_y1,img_x2 - img_x1], dtype='int32')
        if new_ret_coords==self.ret_coords:
            if self.normalizationVector is None:
                self.normalizationVector = self.normalizationImage[ret_y1:ret_y2, ret_x1:ret_x2].ravel()
            return
        self.ret_coords = new_ret_coords
        self.loadArrays()
        # Normalization vector
        self.normalizationVector = self.normalizationImage[ret_y1:ret_y2, ret_x1:ret_x2].ravel()
        # Size array
        new_size_array = np.zeros(self.size_array.shape,dtype=types['Kernel_Size'])
        for m in self.kernel_map:
            generateSizes(new_size_array,m.reshape(self.size)[ret_y1:ret_y2, ret_x1:ret_x2].ravel())
        self.size_array = new_size_array
        # Coeff and index arrays
        validRange =np.zeros(self.size,dtype='bool')
        validRange[ret_y1:ret_y2, ret_x1:ret_x2]=True
        mask = np.zeros(self.index_array.shape,dtype='bool')
        generateMask(self.index_array,validRange.ravel(),mask)
        new_index = np.zeros(self.size,dtype=types['INDEX'])
        new_index[ret_y1:ret_y2, ret_x1:ret_x2]=np.arange(new_size[0]*new_size[1]).reshape(new_size)
        self.index_array = self.index_array[mask]
        self.index_array = new_index.ravel()[self.index_array]
        self.coeff_array = self.coeff_array[mask]
        # Update size
        self.size = new_size

    ########################################
    ############## GRAY SCALE ##############
    ########################################

    def sample_gray(self, img):
        self.sampledVector = np.zeros(len(self.size_array), dtype=types['RESULT'])
        x1, y1, x2, y2 = self.crop_coords
        img = img[y1:y2, x1:x2].ravel()
        sample(img,self.coeff_array,self.sampledVector,self.size_array,self.index_array)

    def backProject_gray(self):
        self.backProjectedVector = np.zeros(self.size[0]*self.size[1], dtype=types['BACK_PROJECTED'])
        backProject(self.sampledVector, self.coeff_array, self.backProjectedVector, self.size_array, self.index_array)
        normalize(self.backProjectedVector, self.normalizationVector)

    #########################################
    ################## RGB ##################
    #########################################

    def sample_rgb(self, img):
        self.sampledVector = np.zeros((3, len(self.size_array)), dtype=types['RESULT'])
        x1, y1, x2, y2 = self.crop_coords
        R = img[y1:y2, x1:x2, 0].ravel()
        G = img[y1:y2, x1:x2, 1].ravel()
        B = img[y1:y2, x1:x2, 2].ravel()
        sampleRGB(R, G, B, self.coeff_array, self.sampledVector[0], self.sampledVector[1], self.sampledVector[2], self.size_array, self.index_array)

    def backProject_rgb(self):
        self.backProjectedVector = np.zeros((3,self.size[0]*self.size[1]), dtype=types['BACK_PROJECTED'])
        backProjectRGB(self.sampledVector[0], self.sampledVector[1], self.sampledVector[2], self.coeff_array, self.backProjectedVector[0], self.backProjectedVector[1], self.backProjectedVector[2], self.size_array, self.index_array)
        normalizeRGB(self.backProjectedVector[0], self.backProjectedVector[1], self.backProjectedVector[2], self.normalizationVector)

    #########################################
    ######## CALIBRATE SIZE AND TYPE ########
    #########################################

    def calibrate(self, img): 
        self.RGB = (len(img.shape)==3) and (img.shape[-1]==3)
        if self.RGB:
            self.sample = self.sample_rgb
            self.backProject = self.backProject_rgb
            self.getResult = lambda: divideRGB(np.copy(self.sampledVector), self.scalingFactor)
        else:
            self.sample = self.sample_gray
            self.backProject = self.backProject_gray
            self.getResult = lambda: (self.sampledVector/self.scalingFactor).astype(types['RESULT'])
        
        self.setInputResolution(img.shape[1],img.shape[0])
        self.setFixation(img.shape[1]/2,img.shape[0]/2)

#===============================================================================================================================
#===============================================================================================================================
#                                      ██████╗ ██████╗ ██████╗ ████████╗███████╗██╗  ██╗
#                                     ██╔════╝██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝
#                                     ██║     ██║   ██║██████╔╝   ██║   █████╗   ╚███╔╝ 
#                                     ██║     ██║   ██║██╔══██╗   ██║   ██╔══╝   ██╔██╗ 
#                                     ╚██████╗╚██████╔╝██║  ██║   ██║   ███████╗██╔╝ ██╗
#                                      ╚═════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
#===============================================================================================================================
#===============================================================================================================================

class Cortex:
    def __init__(self, fname):
        self.backProjectedVector = None
        self.normalizationVector = None
        self.right_hemi_size = None
        self.left_hemi_size = None
        self.scalingFactor = None
        self.backProject = None
        self.coeff_array = None
        self.index_array = None
        self.size_array = None
        self.fname = fname
        self.size = None

        try:
            self.loadArrays()
            self.createNormVect()
            pass
        except:
            raise Exception('Could not find previously saved arrays! Please make sure to initlaize them.')

    def loadArrays(self):
        self.size,self.scalingFactor,self.coeff_array,self.index_array,self.size_array,hemi_sizes = loadPickle(self.fname)
        self.left_hemi_size, self.right_hemi_size = hemi_sizes
        
    def createNormVect(self):
        ones = np.ones(len(self.size_array), dtype=types['RESULT'])
        self.backProjectedVector = np.zeros(self.size[0]*self.size[1], dtype=types['BACK_PROJECTED'])
        backProject(ones, self.coeff_array, self.backProjectedVector, self.size_array, self.index_array)
        self.normalizationVector = np.copy(self.backProjectedVector)

    ########################################
    ############## GRAY SCALE ##############
    ########################################

    def backProject_gray(self, sampledVector):
        self.backProjectedVector = np.zeros(self.size[0]*self.size[1], dtype=types['BACK_PROJECTED'])
        backProject(sampledVector, self.coeff_array, self.backProjectedVector, self.size_array, self.index_array)
        normalize(self.backProjectedVector, self.normalizationVector)

    #########################################
    ################## RGB ##################
    #########################################

    def backProject_rgb(self, sampledVector):
        self.backProjectedVector = np.zeros((3,self.size[0]*self.size[1]), dtype=types['BACK_PROJECTED'])
        backProjectRGB(sampledVector[0], sampledVector[1], sampledVector[2], self.coeff_array, self.backProjectedVector[0], self.backProjectedVector[1], self.backProjectedVector[2], self.size_array, self.index_array)
        normalizeRGB(self.backProjectedVector[0], self.backProjectedVector[1], self.backProjectedVector[2], self.normalizationVector)

    #########################################
    ######## CALIBRATE SIZE AND TYPE ########
    #########################################

    def calibrate(self, retina):
        if retina.RGB:
            self.backProject = self.backProject_rgb
        else:
            self.backProject = self.backProject_gray

#===============================================================================================================================
#===============================================================================================================================
#             ██████╗ ██╗   ██╗██████╗  █████╗ ███╗   ███╗██╗██████╗     ██╗     ███████╗██╗   ██╗███████╗██╗     
#             ██╔══██╗╚██╗ ██╔╝██╔══██╗██╔══██╗████╗ ████║██║██╔══██╗    ██║     ██╔════╝██║   ██║██╔════╝██║     
#             ██████╔╝ ╚████╔╝ ██████╔╝███████║██╔████╔██║██║██║  ██║    ██║     █████╗  ██║   ██║█████╗  ██║     
#             ██╔═══╝   ╚██╔╝  ██╔══██╗██╔══██║██║╚██╔╝██║██║██║  ██║    ██║     ██╔══╝  ╚██╗ ██╔╝██╔══╝  ██║     
#             ██║        ██║   ██║  ██║██║  ██║██║ ╚═╝ ██║██║██████╔╝    ███████╗███████╗ ╚████╔╝ ███████╗███████╗
#             ╚═╝        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═════╝     ╚══════╝╚══════╝  ╚═══╝  ╚══════╝╚══════╝
#===============================================================================================================================
#===============================================================================================================================

class Pyramid_Level:
    def __init__(self, coeff_array, index_array, size_array, input_dimension, scalingFactor):
        self.scalingFactor = scalingFactor
        self.coeff_array = coeff_array
        self.index_array = index_array
        self.size_array = size_array
        self.out_dim = len(size_array)
        self.in_dim = input_dimension
        self.backProjectedVector = None
        self.normalizationVector = None
        self.sampledVector = None
        self.backProject = None
        self.getResult = None
        self.sample = None
        
        self.createNormVect()
        
    def createNormVect(self):
        ones = np.ones(self.in_dim ,dtype=types['INPUT'])
        self.sampledVector = np.zeros(self.out_dim, dtype=types['RESULT'])
        sample(ones,self.coeff_array,self.sampledVector,self.size_array,self.index_array)
        self.backProjectedVector = np.zeros(self.in_dim, dtype=types['BACK_PROJECTED'])
        backProject(self.sampledVector, self.coeff_array, self.backProjectedVector, self.size_array, self.index_array)
        self.normalizationVector = np.copy(self.backProjectedVector)

    ########################################
    ############## GRAY SCALE ##############
    ########################################

    def sample_gray(self, img_vect):
        self.sampledVector = np.zeros(self.out_dim, dtype=types['RESULT'])
        sample(img_vect,self.coeff_array,self.sampledVector,self.size_array,self.index_array)

    def backProject_gray(self):
        self.backProjectedVector = np.zeros(self.in_dim, dtype=types['BACK_PROJECTED'])
        backProject(self.sampledVector, self.coeff_array, self.backProjectedVector, self.size_array, self.index_array)
        normalize(self.backProjectedVector, self.normalizationVector)

    #########################################
    ################## RGB ##################
    #########################################

    def sample_rgb(self, img_vect):
        self.sampledVector = np.zeros((3, self.out_dim), dtype=types['RESULT'])
        R = img_vect[0]
        G = img_vect[1]
        B = img_vect[2]
        sampleRGB(R, G, B, self.coeff_array, self.sampledVector[0], self.sampledVector[1], self.sampledVector[2], self.size_array, self.index_array)

    def backProject_rgb(self):
        self.backProjectedVector = np.zeros((3, self.in_dim), dtype=types['BACK_PROJECTED'])
        backProjectRGB(self.sampledVector[0], self.sampledVector[1], self.sampledVector[2], self.coeff_array, self.backProjectedVector[0], self.backProjectedVector[1], self.backProjectedVector[2], self.size_array, self.index_array)
        normalizeRGB(self.backProjectedVector[0], self.backProjectedVector[1], self.backProjectedVector[2], self.normalizationVector)

    #########################################
    ######## CALIBRATE SIZE AND TYPE ########
    #########################################

    def calibrate(self, img_vect):
        rgb = (len(img_vect.shape)==2) and (img_vect.shape[0]==3)
        if rgb:
            self.sample = self.sample_rgb
            self.backProject = self.backProject_rgb
            self.getResult = lambda: divideRGB(np.copy(self.sampledVector), self.scalingFactor)
        else:
            self.sample = self.sample_gray
            self.backProject = self.backProject_gray
            self.getResult = lambda: (self.sampledVector/self.scalingFactor).astype(types['RESULT'])