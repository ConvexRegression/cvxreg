import numpy as np
from cvxreg.models import CR
from time import time

# Generate data
np.random.seed(0)
n, d, SNR = 200, 5, 3
x = np.random.uniform(low=-1, high=1, size=(n, d))
y_true = np.linalg.norm(x, axis=1)**2 + 3

sigma = np.sqrt(np.var(y_true, ddof=1, axis=0)/SNR)
nse = np.random.normal(0, sigma, n)
y = y_true + nse

# print the computation time
cr = CR(solver='ecos')   # Fit the model
t_start = time()
cr.fit(x, y)
t_delta = time() - t_start
print('Seconds taken: %f' % t_delta)

y_pred = cr.predict(x)    # Make predictions
print(np.mean(np.sum(np.square(y - y_pred))))
# print(cr.predict([[0.1, 0.2, 0.3]]))