# RetinaV2
 Faster and more simplified version of the CPU based retina

# Requirements:
Python 3
Numpy
Cython
OpenCV
C++ compiler (Visual studio on windows / GCC on linux)

# Instructions:
# 1) Run  the "Generator - GUI.ipynb" notebook
# 2) Select the data type and number of bits used for each data type (if float is selected for the coefficients, the quantization bits will be ignored)
# 3) After selecting the desired types and number of bits, generate the configuration file. This file saves the data types and will be used in all future steps. Keep in mind only the configuration file is used, so if the settings are changed without generating a new configuration file, the changes will be discarded.
# 4) Convert the Cython template to a Cython extension. This will replace the placeholder data types with the types loaded from the configuration file.
# 5) Compile the Cython extension (functions.pyx). This should create the functions.pyd on windows and functions.so file on linux.
# 6) Select the data files for retina and cortex and generate arrays (some example data can be found at https://github.com/2332575Y/Retina which is a previous version of the CPU based software retina).
# 7) You can now run the demo or the performance benchmark.
