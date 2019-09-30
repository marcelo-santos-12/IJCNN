"""
Methods to characterize image textures.
"""

import numpy as np
import warnings
from .utils._texture import _local_binary_pattern
from .utils._texture_ilbp import _improved_local_binary_pattern
from .utils._texture_hlbp import _hamming_local_binary_pattern
from .utils._texture_elbp import _extended_local_binary_pattern
from .utils._texture_clbp import _completed_local_binary_pattern

def base_lbp(image, P, R, method='default'):
    check_nD(image, 2)

    methods = {
        'default': ord('D'),
        'ror': ord('R'),
        'uniform': ord('U'),
        'nri_uniform': ord('N'),
        'var': ord('V')
    }
    image = np.ascontiguousarray(image, dtype=np.double)
    output = _local_binary_pattern(image, P, R, methods[method.lower()])
    return output

def improved_lbp(image, P, R, method='default'):
    check_nD(image, 2)

    methods = {
        'default': ord('D'),
        'ror': ord('R'),
        'uniform': ord('U'),
        'nri_uniform': ord('N'),
        'var': ord('V')
    }
    image = np.ascontiguousarray(image, dtype=np.double)
    output = _improved_local_binary_pattern(image, P, R, methods[method.lower()])
    return output

def extendend_lbp(image, P, R, method='default'):
    check_nD(image, 2)

    methods = {
        'default': ord('D'),
        'ror': ord('R'),
        'uniform': ord('U'),
        'nri_uniform': ord('N'),
        'var': ord('V')
    }
    image = np.ascontiguousarray(image, dtype=np.double)
    output = _extended_local_binary_pattern(image, P, R, methods[method.lower()])
    return output

def hamming_lbp(image, P, R, method='default'):
    check_nD(image, 2)

    methods = {
        'default': ord('D'),
        'ror': ord('R'),
        'uniform': ord('U'),
        'nri_uniform': ord('N'),
        'var': ord('V')
    }
    image = np.ascontiguousarray(image, dtype=np.double)
    output = _hamming_local_binary_pattern(image, P, R, methods[method.lower()])
    return output

def completed_lbp(image, P, R, method='default'):
    check_nD(image, 2)

    methods = {
        'default': ord('D'),
        'ror': ord('R'),
        'uniform': ord('U'),
        'nri_uniform': ord('N'),
        'var': ord('V')
    }
    image = np.ascontiguousarray(image, dtype=np.double)
    output = _completed_local_binary_pattern(image, P, R, methods[method.lower()])
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