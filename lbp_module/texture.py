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
from skimage.feature import local_binary_pattern as lbp

DEFAULT = 'default'
ROR = 'ror'
UNIFORM = 'uniform' 
NRI_UNIFORM = 'nri_uniform'
VAR = 'var'

methods = {
    DEFAULT: ord('D'),
    ROR: ord('R'),
    UNIFORM: ord('U'),
    NRI_UNIFORM: ord('N'),
    VAR: ord('V')
}

# Foram Obtidos Experimentalmente
bins_ROR = {
'4': 6,
'5': 8,
'6': 14,
'7': 20,
'8': 36,
'9': 60,
'10': 108,
'11': 188,
'12': 352,
'13': 632,
'14': 1182,
'15': 2192,
'16': 4116,
'17': 7712,
'18': 14602,
'19': 27596,
'20': 52488,
'21': 99880,
'22': 190746,
'23': 364724,
'24': 699252
}

def base_lbp(image, P, R, method, block,):
    check_nD(image, 2)
    image = np.ascontiguousarray(image, dtype=np.double)
    output = lbp(image, P, R, method) # original

    if method == DEFAULT:
        bins = 2**P
    
    elif method == UNIFORM:
        bins = P + 2

    elif method == NRI_UNIFORM:
        bins = P * (P - 1) + 3
    
    elif method == ROR:
        bins = bins_ROR[str(P)]
    
    else: # method == VAR
        bins = None

    return histogram(output, bins, block)

def improved_lbp(image, P, R, method, block,):
    check_nD(image, 2)
    image = np.ascontiguousarray(image, dtype=np.double)
    output = _improved_local_binary_pattern(image, P, R, methods[method.lower()])    

    if method == DEFAULT:
        bins = 2**(P + 1)
    
    elif method == UNIFORM:
        bins = P + 3

    elif method == NRI_UNIFORM:
        bins = (P + 1) * P + 3
    
    elif method == ROR:
        bins = bins_ROR[str(P + 1)]
    
    else: # method == VAR
        bins = None

    return histogram(output, bins, block)

def extended_lbp(image, P, R, method, block,):
    check_nD(image, 2)

    image = np.ascontiguousarray(image, dtype=np.double)
    output = _extended_local_binary_pattern(image, P, R, methods[method.lower()])

    if method == DEFAULT:
        bins = 2**P
    
    elif method == UNIFORM:
        bins = P + 2

    elif method == NRI_UNIFORM:
        bins = P * (P - 1) + 3
    
    elif method == ROR:
        bins = bins_ROR[str(P)]
    
    else: # method == VAR
        bins = None

    return histogram(output, bins=bins, block=block)

def hamming_lbp(image, P, R, method, block,):
    assert method == UNIFORM or method == NRI_UNIFORM, 'Method --> {}. Dont permissed for this variant.'.format(method)
    check_nD(image, 2)
    image = np.ascontiguousarray(image, dtype=np.double)
    output = _hamming_local_binary_pattern(image, P, R, methods[method.lower()])
    
    if method == UNIFORM:
        bins = P + 1

    else: # method == NRI_UNIFORM:
        bins = P * (P - 1) + 2
   
    return histogram(output, bins=bins, block=block)

def completed_lbp(image, P, R, method, block,):
    check_nD(image, 2)
    image = np.ascontiguousarray(image, dtype=np.double)
    output = _completed_local_binary_pattern(image, P, R, methods[method.lower()])    
    
    if method == DEFAULT:
        bins = 2**P
    
    elif method == UNIFORM:
        bins = P + 2

    elif method == NRI_UNIFORM:
        bins = P * (P - 1) + 3
    
    elif method == ROR:
        bins = bins_ROR[str(P)]
    
    else: # method == VAR
        bins = None

    return histogram(output, bins=bins, block=block)

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

def histogram(output, bins, block):
    r_range = int(output.shape[0]/block[0])
    c_range = int(output.shape[1]/block[1])
    hist = []

    for r in range(0, output.shape[0], r_range):
        for c in range(0, output.shape[1], c_range):

            hist_roi = np.histogram(output[r:r + r_range, c:c + c_range], bins=bins)[0]
    
            hist.extend(list(hist_roi))
            
    return np.asarray(hist)