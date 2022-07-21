import numpy as np 
from numpy.linalg import inv
from CommPy.utils import hermitian

__all__ = ['ZF_initial', 'MMSE_initial', 'BPSK_optimal', 'QPSK_optimal', 'QAM4_optimal']


def ZF_initial(H, y, constellation, lld_k_Initial):

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


def MMSE_initial(H, y, lamda, constellation, lld_k_Initial):

    m = H.shape[0]
    n = H.shape[1]

    lamda_mat = 0.5 * lamda * (np.eye(m) + 1j* np.eye(m))
    z = inv(np.matmul(H, hermitian(H)) + lamda_mat)
    z = np.matmul(np.matmul(hermitian(H), z), y)
    z_real = np.real(z)
    z_imag = np.imag(z)

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


def BPSK_optimal(y, H, constellation, bound):
    # draw the dicision boundary for BPSK
    
    num_symbols = y.shape[0]
    x_pred = np.zeros(y.shape) + 1j * np.zeros(y.shape)
    xhat_indices = np.zeros(y.shape, dtype=int)
    #z = inv(np.matmul(H, hermitian(H)))
    #z = np.matmul(np.matmul(hermitian(H), z), y)
    z = y
    
    for i in range(0, num_symbols):
        if np.real(z[i]) > bound:
            xhat_indices[i] = 0
            x_pred[i] = constellation[0]
        else:
            xhat_indices[i] = 1
            x_pred[i] = constellation[1]
    
    return xhat_indices, x_pred


def QPSK_optimal(y, H, constellation, offset):
    # draw the dicision boundary for QPSK
    # offset = [offset_0, offset_1]

    num_symbols = y.shape[0]
    x_pred = np.zeros(y.shape) + 1j * np.zeros(y.shape)
    #z = inv(np.matmul(H, hermitian(H)))
    #z = np.matmul(np.matmul(hermitian(H), z), y)
    z = y

    for i in range(0, num_symbols):
        real = np.real(z[i])
        imag = np.imag(z[i])
        if imag > (-(real + offset[1])):
            if imag < (real + offset[0]):
                x_pred[i] = constellation[0]
            else:
                x_pred[i] = constellation[1]
        else:
            if imag < (real + offset[0]):
                x_pred[i] = constellation[2]
            else:
                x_pred[i] = constellation[3]

    return x_pred


def QAM4_optimal(y, H, constellation, bound):
    # draw the dicision boundary for QAM4
    # bound = [bound0, bound1]

    num_symbols = y.shape[0]
    x_pred = np.zeros(y.shape) + 1j * np.zeros(y.shape)
    #z = inv(np.matmul(H, hermitian(H)))
    #z = np.matmul(np.matmul(hermitian(H), z), y)
    z = y

    for i in range(0, num_symbols):
        real = np.real(z[i])
        imag = np.imag(z[i])
        if real < bound[0]:
            if imag < bound[1]:
                x_pred[i] = constellation[0]
            else:
                x_pred[i] = constellation[1]
        else:
            if imag < bound[1]:
                x_pred[i] = constellation[2]
            else:
                x_pred[i] = constellation[3]
    
    return x_pred