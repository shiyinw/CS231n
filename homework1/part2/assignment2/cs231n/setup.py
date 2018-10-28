from distutils.core import setup
from distutils.extension import Extension
import numpy
import sys
sys.path.append("/usr/local/bin/")
print(sys.path)

try:
    from Cython.Build import cythonize
except ImportError:
     def cythonize(*args, **kwargs):
         from Cython.Build import cythonize
         return cythonize(*args, **kwargs)


extensions = [
  Extension('im2col_cython', ['im2col_cython.pyx'],
            include_dirs = [numpy.get_include()]
  ),
]

setup(
    ext_modules = cythonize(extensions),
)
