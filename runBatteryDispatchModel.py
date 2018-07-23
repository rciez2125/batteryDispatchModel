import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize
from scipy.sparse import identity, tril, hstack, vstack, coo_matrix, find
from cvxopt import matrix, solvers, spmatrix
from scrips import makeLMPhist, setConstraintsA, plotDurationFrequency, plotDispatchCurve

# import PJM data
pjmdata = pd.read_csv('JCPL_DA_2017data.csv')
pjmdata.drop(['voltage', 'equipment', 'zone'], axis=1)

makeLMPhist('PJM', pjmdata.total_lmp_da)

d = 0 # degradation costs
p = pjmdata.total_lmp_da
p = p / 1000  # convert to $/kWh from $/MWh
#p = p.head(24*31*4)
eta = np.sqrt(0.6)
t = p.shape[0]
nameCap = 100 * 5 / 1000 # nameplate capacity in MWh
ub = np.ones([2*t,])*nameCap/5
lb = np.zeros([t*2,])
chargeCost = (p + np.ones([t,])*d)
dischargeCost = eta*(np.ones([t,])*d-p)

G = setConstraintsA(eta, t)
c = np.hstack((chargeCost, dischargeCost))
h = matrix(np.hstack((np.ones([t,])*nameCap, np.zeros([t, ]), 
	np.ones([2*t, ])*nameCap/5, np.zeros([2*t, ]))))

c = matrix(c)
ans3 = solvers.lp(c, G, h, solver = 'glpk')
outData = np.array(ans3['x'])
val = ans3['primal objective']
z = np.append(val, (outData))
df = pd.DataFrame(z)
df.to_csv('PJMdata.csv')

plotDispatchCurve(outData, eta, t, 'PJM')

plotDurationFrequency(outData, t, 'PJM')
