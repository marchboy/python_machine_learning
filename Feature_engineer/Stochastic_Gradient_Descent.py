#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def minimaFunction(theta):
    return np.cos(3 * np.pi * theta) / theta


def minimaFunctionDerivatite(theta):
    const1 = 3 * np.pi * np.sin(3 * np.pi * theta)
    const2 = np.cos(3 * np.pi * theta)
    return -(const1/theta) - (const2/theta**2)


theta = np.arange(0.1, 2.1, 0.01)
Jtheta = minimaFunction(theta)
dJtheta = minimaFunctionDerivatite(theta)


plt.plot(theta, Jtheta, label=r'$J({\theta})$')
plt.plot(theta, dJtheta/30, label='$dJ({\\theta})$')
plt.legend()
axes = plt.gca()

plt.xlabel('$\\theta$')
plt.ylabel('$J({\\theta}), dJ({\\theta})/30$')
plt.title('$J({\\theta}), dJ({\\theta})/30$ VS $\\theta$')

plt.show()

print(theta)
print(Jtheta)


