# Para construir arquivo cython e permitir importar
#  python setup.py build_ext --inplace

import os

from distutils.core import setup
from Cython.Build import cythonize

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('shared', parent_package, top_path)

    cythonize('interpolation.pyx', working_path=base_path)
    config.add_extension('interpolation', sources=['interpolation.c'])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**(configuration(top_path='').todict())
          )
