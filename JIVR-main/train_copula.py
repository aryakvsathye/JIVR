import pandas as pd
import numpy as np
from copulae import GaussianCopula

residuals = pd.read_csv('saved_model/residuals.csv')
_, ndim = residuals.shape
copula = GaussianCopula(dim=ndim)
copula.fit(residuals)
np.save('saved_model/copula_corr.npy', copula.sigma)