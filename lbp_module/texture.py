"""
Methods to characterize image textures.
"""

import numpy as np
import warnings
from .utils._texture import _local_binary_pattern

def local_binary_pattern(image, P, R, method='default'):
    check_nD(image, 2)

    methods = {
        'default': ord('D'),
        'improved': ord('I'),
        'completed': ord('C'),
        'hamming': ord('H'),
        'extended': ord('E')
    }
    image = np.ascontiguousarray(image, dtype=np.double)
    output = _local_binary_pattern(image, P, R, methods[method.lower()])
    return output

def check_nD(array, ndim, arg_name='image'):
    array = np.asanyarray(array)
    msg_incorrect_dim = "The parameter `%s` must be a %s-dimensional array"
    msg_empty_array = "The parameter `%s` cannot be an empty array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if array.size == 0:
        raise ValueError(msg_empty_array % (arg_name))
    if not array.ndim in ndim:
        raise ValueError(msg_incorrect_dim % (arg_name, '-or-'.join([str(n) for n in ndim])))
