import numpy as np 
from numpy.linalg import inv
from CommPy.utils import hermitian, decimal2Qbase

__all__ = ['pseudo_inverse', 'maximum_likelihood', 'zero_forcing', 'MMSE_BPSK', 'MMSE_general']


def pseudo_inverse(y, H):
    
    z = inv(np.matmul(H, hermitian(H)))
    z = np.matmul(np.matmul(hermitian(H), z), y)
    
    return z


def maximum_likelihood(y, H, mod, constellation, N):
    # exhaustive search
    # y: y_symbols
    # N: number of user
    xhat = np.zeros([N, ]) + 1j * np.zeros([N, ])
    num_x = int(np.power(mod, N))
    squared_err = np.zeros(num_x)
    for i in range(0, num_x):
        coefficients = decimal2Qbase(i, mod, N)
        symbols = constellation[coefficients]
        squared_err[i] = np.sum(np.square(np.abs(y - np.matmul(H, symbols))))
    
    xhat_dec = np.argmin(squared_err)
    xhat_indices = decimal2Qbase(xhat_dec, mod, N)
    xhat_symbols = constellation[xhat_indices]

    return xhat_indices, xhat_symbols


def zero_forcing(H, y, constellation):
    z = inv(np.matmul(H, hermitian(H)))
    z = np.matmul(np.matmul(hermitian(H), z), y)
    return z


def MMSE_BPSK(H, y, lamda, constellation):
    m = H.shape[0]
    n = H.shape[1]

    lamda_mat = lamda * np.eye(m)
    z = inv(np.matmul(H, hermitian(H)) + lamda_mat)
    z = np.matmul(np.matmul(hermitian(H), z), y)
    return z


def MMSE_general(H, y, lamda, constellation):
    m = H.shape[0]
    n = H.shape[1]

    lamda_mat = 0.5 * lamda * (np.eye(m) + 1j* np.eye(m))
    z = inv(np.matmul(H, hermitian(H)) + lamda_mat)
    z = np.matmul(np.matmul(hermitian(H), z), y)
    return z