import numpy as np 

__all__ = ['iterative_SIC']


def iterative_SIC(X_Initial, var_Initial, var_noise, y, H, constellation, num_iter):

    X = X_Initial
    e_squared = var_Initial
    H_squared = np.square(np.abs(H))
    C = constellation.shape[0]
    bound = 1e-100

    m = H.shape[0]
    n = H.shape[1]
    z = np.random.randn(m) + 1j * np.random.randn(m)
    delta = np.random.randn(m) + 1j * np.random.randn(m)
    x_predicted = np.zeros((num_iter, n), dtype = int)
    prob = np.zeros((C, n))

    for ll in range(0, num_iter):
        # Compute the log-likelihood of each element in X
        # Soft decision
        for k in range(0, n):
            for ii in range(0, m):
                field = np.concatenate((np.arange(0, k), np.arange(k+1, n)))
                z[ii] = y[ii] - np.sum( H[ii, field] * X[field] )
                delta[ii] = var_noise + np.sum( H_squared[ii, field] * e_squared[field] )

            lld_k = np.zeros((C, m))
            for alpha in range(0, C):
                # each possible value in the constellation
                for ii in range(0, m):
                    current_lld = np.log( 1/(np.pi * delta[ii]) * np.exp( -np.square(np.abs(z[ii]-H[ii, k]*constellation[alpha])) / delta[ii]))
                    lld_k[alpha, ii] = np.real(current_lld)
            
            lld_k = np.sum(lld_k, 1)
            prob[:, k] = np.exp(lld_k)

            if np.sum(np.exp(lld_k)) > bound:
                X[k] = np.sum(constellation * np.exp(lld_k)) / np.sum(np.exp(lld_k))
                e_squared[k] = np.sum(np.square(np.abs(constellation - X[k])) * np.exp(lld_k)) / np.sum(np.exp(lld_k))
            else:
                X[k] = 0
                e_squared[k] = np.sum(np.square(np.abs(constellation - X[k])) * np.exp(lld_k)) / np.sum(np.exp(lld_k))
  
        # hard decision
        x_predicted[ll] = np.argmax(prob, axis=0)
    
    # after ll iterations
    return x_predicted, prob