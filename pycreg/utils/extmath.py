import numpy as np

from ..constant import Convex, Concave


# Calculate yhat in testing sample
def yhat(alpha, beta, x_test, fun=Convex):
    '''
    function estimate the y_hat of convex functions.
    refers to equation (4.1) in journal article:
    "Representation theorem for convex nonparametric least squares. Timo Kuosmanen (2008)"
    input:
    alpha and beta are regression coefficients; x_test is the input of test sample.
    output:
    return the estimated y_hat.
    '''

    # check the dimension of input
    if beta.shape[1] != x_test.shape[1]:
        raise ValueError('beta and x_test should have the same number of dimensions.')
    else:
        # compute yhat for each testing observation
        yhat = np.zeros((len(x_test),))
        for i in range(len(x_test)):
            if fun == Concave:
                yhat[i] = (alpha + np.sum(np.multiply(beta, x_test[i]), axis=1)).min(axis=0)
            elif fun == Convex:
                yhat[i] = (alpha + np.sum(np.multiply(beta, x_test[i]), axis=1)).max(axis=0)

    return yhat