from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def _bit_rotate_right(value, length):
    return (value >> 1) | ((value & 1) << (length - 1))

def ROR(P):
    _ri = []
    rotation_chain = np.zeros(P, np.int)
    for lbp in tqdm(range(2**P)):
        rotation_chain[0] = lbp
        for i in range(1, P):
            rotation_chain[i] = \
                _bit_rotate_right(rotation_chain[i - 1], P)
        lbp = rotation_chain[0]
        for i in range(1, P):
            lbp = min(lbp, rotation_chain[i])
        _ri.append(lbp)

    return np.asarray(_ri)

def main():

    P = []
    #for i in range(25, 25):
    #    print('P = {} --> {}'.format(i, np.unique(ROR(i)).shape))
    #    P.append(np.unique(ROR(i)).shape)
    print('Iniciando teste')
    print(np.unique(ROR(25)).shape)
    
    #x1 = np.arange(4, 14)
    #y1 = np.array([6, 8, 14, 20, 36, 60, 108, 188, 352, 632])

    #plt.plot(x1, y1, color='green', label='Exato')

    #y2 = (x1) + np.sqrt(x1) * np.exp(x1) / x1**3 
    
    #plt.plot(x1, y2, color='red', label='Aproximado')

    #plt.legend()
    #plt.show()

    #print('y1: ', y1)
    #print('y2: ', np.array(y2, dtype=np.int))

if __name__ == '__main__':

    main()
