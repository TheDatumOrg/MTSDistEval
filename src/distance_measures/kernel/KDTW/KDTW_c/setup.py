from distutils.core import setup, Extension
import numpy
# define the extension module
KDTW = Extension('KDTW', sources=['KDTW.c'])

# run the setup
setup(ext_modules=[KDTW])

