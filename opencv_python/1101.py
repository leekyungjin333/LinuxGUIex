#1101.py
import numpy as np
import matplotlib.pyplot as plt

#1
alpha = 1
beta  = 1
x = np.linspace(-10, 10, num=100)
y = beta*(1-np.exp(-alpha*x))/(1+np.exp(-alpha*x))
plt.plot(x, y, label='alpha=1, beta=1')

#2
alpha = 2/3
beta  = 1.7159
y = beta*(1-np.exp(-alpha*x))/(1+np.exp(-alpha*x))
plt.plot(x, y, label='alpha=2/3, beta=1.7159')

#3
#y = np.tanh(x)
y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
plt.plot(x, y, label='tanh')

#4
y = beta*np.tanh(x*alpha)
plt.plot(x, y, label='beta*np.tanh(x*alpha)')

plt.legend(loc='best')
plt.show()
