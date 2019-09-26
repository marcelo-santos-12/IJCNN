import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos, fabs
from .interpolation cimport bilinear_interpolation, round
from cpython cimport array


ctypedef fused np_ints:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t

ctypedef fused np_uints:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t

ctypedef fused np_anyint:
    np_uints
    np_ints

ctypedef fused np_floats:
    cnp.float32_t
    cnp.float64_t

ctypedef fused np_real_numeric:
    np_anyint
    np_floats

cdef extern from "numpy/npy_math.h":
    double NAN "NPY_NAN"

cdef inline int _bit_rotate_right(int value, int length) nogil:
    return (value >> 1) | ((value & 1) << (length - 1))

def to_bin(aux_absolute):
    n_bits=3
    binary = [0, 0, 0]
    for j in range(n_bits): # Decimal to Binary
        binary[j] = 1 if (aux_absolute & 1) == 1 else 0
        aux_absolute = aux_absolute >> 1
    return binary

def _local_binary_pattern(double[:, ::1] image,
                          int P, float R, char method=b'D'):

    # texture weights
    cdef int[::1] weights = 2 ** np.arange(P, dtype=np.int32)
    # local position of texture elements
    rr = - R * np.sin(2 * np.pi * np.arange(P, dtype=np.double) / P)
    cc = R * np.cos(2 * np.pi * np.arange(P, dtype=np.double) / P)
    cdef double[::1] rp = np.round(rr, 5)
    cdef double[::1] cp = np.round(cc, 5)

    # pre-allocate arrays for computation
    cdef double[::1] texture = np.zeros(P, dtype=np.double)
    cdef signed char[::1] signed_texture = np.zeros(P, dtype=np.int8)
    cdef int[::1] rotation_chain = np.zeros(P, dtype=np.int32)

    output_shape = (image.shape[0], image.shape[1])
    cdef double[:, ::1] output = np.zeros(output_shape, dtype=np.double)

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]

    cdef double lbp
    cdef Py_ssize_t r, c, changes, i
    cdef Py_ssize_t rot_index, n_ones
    cdef cnp.int8_t first_zero, first_one

    # To compute the variance features
    cdef double sum_, var_, texture_i

    # To compute Extended LBP
    cdef Py_ssize_t n_bits = 3
    cdef int absolute_difference
    cdef int j, aux_absolute
    cdef signed char [::1] layer1_aux = np.zeros(P, dtype=np.int8) # Original LBP
    cdef signed char[::1] layer2_aux = np.zeros(P, dtype=np.int8)
    cdef signed char[::1] layer3_aux = np.zeros(P, dtype=np.int8)
    cdef signed char[::1] layer4_aux = np.zeros(P, dtype=np.int8)
    cdef signed char[::1] binary = np.zeros(n_bits, dtype=np.int8)
    cdef double[:, ::1] output_layer1 = np.zeros(output_shape, dtype=np.double)
    cdef double[:, ::1] output_layer2 = np.zeros(output_shape, dtype=np.double)
    cdef double[:, ::1] output_layer3 = np.zeros(output_shape, dtype=np.double)
    cdef double[:, ::1] output_layer4 = np.zeros(output_shape, dtype=np.double)
    cdef double layer1_lbp
    cdef double layer2_lbp
    cdef double layer3_lbp
    cdef double layer4_lbp

    with nogil:
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                for i in range(P):
                    bilinear_interpolation[cnp.float64_t, double, double](
                            &image[0, 0], rows, cols, r + rp[i], c + cp[i],
                            b'C', 0, &texture[i])
                # signed / thresholded texture
                for i in range(P): 
                    if texture[i] - image[r, c] >= 0:
                        layer1_aux[i] = 1
                    else:
                        layer1_aux[i] = 0

                    absolute_difference = int(fabs(texture[i] - image[r, c]))

                    aux_absolute =  absolute_difference if absolute_difference < 7 else 7 # According the original article
                    
                    for j in range(n_bits): # Decimal to Binary
                        binary[j] = 1 if (aux_absolute & 1) == 1 else 0
                        aux_absolute = aux_absolute >> 1

                    layer2_aux[i] = binary[0]
                    layer3_aux[i] = binary[1]
                    layer4_aux[i] = binary[2]
                    
                layer1_lbp = 0
                layer2_lbp = 0
                layer3_lbp = 0
                layer4_lbp = 0

                for i in range(P):
                    layer1_lbp += layer1_aux[i] * weights[i]
                    layer2_lbp += layer2_aux[i] * weights[i]
                    layer3_lbp += layer3_aux[i] * weights[i]
                    layer4_lbp += layer4_aux[i] * weights[i]

                
                output_layer1[r, c] = layer1_lbp
                output_layer2[r, c] = layer2_lbp
                output_layer3[r, c] = layer3_lbp
                output_layer4[r, c] = layer4_lbp

    return np.asarray(output_layer1), np.asarray(output_layer2), np.asarray(output_layer3), np.asarray(output_layer4)
