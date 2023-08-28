import numpy as np
from time import time

from cvxreg.models import CR, PCR
from cvxreg.datasets import load_elect_firms
from cvxreg.constant import convex, concave



# Generate data
# np.random.seed(0)
# n, d, SNR = 200, 3, 3
# x = np.random.uniform(low=-1, high=1, size=(n, d))
# y_true = np.linalg.norm(x, axis=1)**2 + 3"""  """

# sigma = np.sqrt(np.var(y_true, ddof=1, axis=0)/SNR)
# nse = np.random.normal(0, sigma, n)
# y = y_true + nse

# # real data
# x, y = load_elect_firms()

# # print the computation time
# cr = CR(solver='mosek')   # Fit the model
# t_start = time()
# cr.fit(x, y)
# t_delta = time() - t_start
# print('Seconds taken: %f' % t_delta)

# # predict the response
# y_pred = cr.predict([[0.1, 0.2, 0.3]])
# print(y_pred)