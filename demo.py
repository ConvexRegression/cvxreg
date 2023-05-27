import numpy as np
from pycvr.ConvexRegression import CR

# Generate data
np.random.seed(0)
x = np.random.uniform(low=-1, high=1, size=(50, 1))
y_true = np.linalg.norm(x, axis=1)**2 + 3
sigma = np.sqrt(np.var(y_true, ddof=1, axis=0)/3)
nse = np.random.normal(0, sigma, 50)

y = y_true + nse

# Fit the model
cr = CR(x ,y)
cr.fit()

alpha = cr.intercept_
beta = cr.coef_
print(cr.predict(x))