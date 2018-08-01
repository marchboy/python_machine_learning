#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def optimize(iteraions, OF, dOF, params, learningRate):
    oParams = [params]

    for i in range(iterations):
        dParams = dOF(params)
        params = params - learningRate*dParams
        oParams.append(params)

    return np.array(oParams)


def minimaFunction(theta):
    return np.cos(3 * np.pi * theta) / theta


def minimaFunctionDerivatite(theta):
    const1 = 3 * np.pi * np.sin(3 * np.pi * theta)
    const2 = np.cos(3 * np.pi * theta)
    return -(const1/theta) - (const2/theta**2)


theta = .6
iterations = 45
learningRate = .0007
optimizedParameters = optimize(iterations, minimaFunction, minimaFunctionDerivatite, theta, learningRate)
print(optimizedParameters)


thetaR = np.arange(0.1, 2.1, 0.01)
Jtheta = minimaFunction(thetaR)

JOptiTheta = minimaFunction(optimizedParameters)
fig, ax = plt.subplots()
line, = ax.plot(thetaR, Jtheta)

axes = plt.gca()
axes.set_ylim([-4,6])
axes.set_xlim([0,2])

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# Animate the updates
def animate(i):
    line, = ax.plot(optimizedParameters[i],JOptiTheta[i],'ob')  # update the data
    plt.title(r'Updating $\theta$ through SGD $\theta$ = %f J($\theta$) = %f' %(optimizedParameters[i],JOptiTheta[i]))
    return line,


ani = animation.FuncAnimation(fig, animate, np.arange(1, iterations), interval=1, blit=True)

ani.save('Stochastic_Gradient_Descent.mp4', writer=writer)
