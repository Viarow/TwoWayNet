import numpy as np 
from numpy.linalg import inv
from CommPy.utils import hermitian

__all__ = ['estimate_initial']


def estimate_initial(H, y, constellation, lld_k_Initial):

    z = inv(np.matmul(H, hermitian(H)))
    z = np.matmul(np.matmul(hermitian(H), z), y)
    z_real = np.real(z)
    z_imag = np.imag(z)

    m = H.shape[0]
    n = H.shape[1]
    X_Initial = np.random.randn(n) + 1j * np.random.randn(n)
    var_Initial = np.zeros(n)

    real_max = np.max(np.real(constellation))
    real_min = np.min(np.real(constellation))
    imag_max = np.max(np.imag(constellation))
    imag_min = np.min(np.imag(constellation))

    for ii in range(0, n):

        if z_real[ii] < real_min:
            X_real = real_min
        elif z_real[ii] > real_max:
            X_real = real_max
        else:
            X_real = z_real[ii]
        
        if z_imag[ii] < imag_min:
            X_imag = imag_min
        elif z_imag[ii] > imag_max:
            X_imag = imag_max
        else:
            X_imag = z_imag[ii]

        X_Initial[ii] = X_real + 1j * X_imag

    for k in range(0, n):
        var_Initial[k] = np.sum(np.square(np.abs(constellation-X_Initial[k])) * np.exp(lld_k_Initial)) / np.sum(np.exp(lld_k_Initial))

    return X_Initial, var_Initial