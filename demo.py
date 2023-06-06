import numpy as np
from cvxreg.models import PCR, CR
from time import time

# Generate data
np.random.seed(0)
n, d = 50, 3
x = np.random.uniform(low=-1, high=1, size=(n, d))
y_true = np.linalg.norm(x, axis=1)**2 + 3
sigma = np.sqrt(np.var(y_true, ddof=1, axis=0)/d)
nse = np.random.normal(0, sigma, n)

y = y_true + nse

# print the computation time
t_start = time()
cr = CR()   # Fit the model
cr.fit(x, y)
t_delta = time() - t_start
print('Seconds taken: %f' % t_delta)

print(cr.predict([[0.1, 0.2, 0.3]]))