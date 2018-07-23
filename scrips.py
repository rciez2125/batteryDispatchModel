import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.optimize import linprog, minimize 
from scipy.sparse import identity, tril, hstack, vstack, coo_matrix, find
from cvxopt import matrix, solvers, spmatrix


def makeLMPhist(isoName, isoPriceData):
	plt.hist(isoPriceData, bins = 'auto')
	plt.title(isoName + '2017 Location Marginal Prices')
	plt.xlabel('Hourly LMP ($/MWh)')
	plt.savefig(isoName + 'pricehist.png')
	plt.close()

def setConstraintsA(eta, t):
	#Ihold = []
	#Jhold = []
	#for n in range(t):
	#	d = np.arange(t-n) + n
	#	Ihold = np.append(Ihold, d)
	#	d = np.ones([t, ])*n
	#	Jhold = np.append(Jhold, d)
	#I = np.append(Ihold, Ihold)
	#J = np.append(Jhold, (Jhold + t))
	#V = np.ones([sum(range(t+1)),])*eta
	#V = np.append(V, np.ones([sum(range(t+1)), ]))
	A1 = hstack([eta*tril(np.ones([t,t])), -1*tril(np.ones([t,t]))])
	f = find(A1)
	I = f[0]
	J = f[1]
	V = f[2]

	A2 = hstack([(-1*eta*tril(np.ones([t,t]))), tril(np.ones([t,t]))])
	f = find(A2)
	I = np.append(I, f[0]+t)
	J = np.append(J, f[1])
	V = np.append(V, f[2])
	A3 = hstack([identity(t), coo_matrix((t,t))])
	f = find(A3)
	I = np.append(I, f[0]+(2*t))
	J = np.append(J, f[1])
	V = np.append(V, f[2])
	A4 = hstack([coo_matrix((t,t)), identity(t)])
	f = find(A4)
	I = np.append(I, f[0]+(3*t))
	J = np.append(J, f[1])
	V = np.append(V, f[2])
	A5 = hstack([-1*identity(t), coo_matrix((t,t))])
	f = find(A5)
	I = np.append(I, f[0]+(4*t))
	J = np.append(J, f[1])
	V = np.append(V, f[2])
	A6 = hstack([coo_matrix((t,t)), -1*identity(t)])
	f = find(A6)
	I = np.append(I, f[0]+(5*t))
	J = np.append(J, f[1])
	V = np.append(V, f[2])
	G = spmatrix(V, I, J)
	return G

def plotDurationFrequency(outData, t, isoName):
	dischargeData = outData[t:2*t, ]
	z = 0
	d = np.zeros([t,])
	for n in range(t):
		if dischargeData[n]>0:
			z = z+1
			d[n] = z
		else:
			z = 0
			d[n] = z
	d = d[d !=0 ]
	plt.hist(d, bins=[0,1,2,3,5,6,7,8,9,10])
	plt.xlabel('Duration of Discharge Cycle (hours)')
	plt.ylabel('Frequency')
	plt.savefig(isoName + 'dischargeDurationFreq.png')
	plt.close()

def plotDispatchCurve(outData, eta, t, isoName):
	chargeData = outData[0:t, ]
	dischargeData = outData[t:2*t, ]
	A1 = hstack([eta*tril(np.ones([t,t])), -1*tril(np.ones([t,t]))])
	soc = A1*outData
	plt.plot(np.arange(0,t,1), chargeData, 'r-', np.arange(0,t,1), dischargeData, 'b-', np.arange(0,t,1), soc, 'g-')
	plt.xlabel('Hour')
	plt.legend(('Charging', 'Discharging', 'Stored Energy'))
	plt.savefig(isoName + 'dispatchCurve.png')
	plt.close()
