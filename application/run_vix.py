import pandas as pd

from application.helper_functions import find_nearest_tau
from vix import vix

data = pd.read_csv('db.csv')

tau = 12
vix_test = vix(tau, data)
