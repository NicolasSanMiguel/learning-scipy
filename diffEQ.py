import numpy as np
import matplotlib.pyplot as plt
import scipy as sp2

from scipy.integrate import odeint

# first order - models speed change with a force on it
def dvdt(v, t):
    return 3*v**2 - 5
v0 = 0

t = np.linspace(0, 1, 100)
soln = odeint(dvdt, v0, t)
print("this is the ode solution:", soln)
deriv_soln = soln.T[0]

plt.figure(1)
plt.plot(t, deriv_soln)
# p1.show()
plt.grid()

# coupled first order
def dSdx(S, x):
    y1, y2 = S
    return [y1 + y2**2  + 3*x,
           3*y1 + y2**3 - np.cos(x)]
y1_0 = 0
y2_0 = 0
S_0 = (y1_0, y2_0)

x = np.linspace(0, 1, 100)
soln = odeint(dSdx, S_0, x)
print("this is the ode solution:",soln)

y1_soln = soln.T[0]
y2_soln = soln.T[1]

plt.figure(2)
plt.plot(x, y1_soln)
plt.plot(x, y2_soln)
# p2.show()
plt.grid()

# 2nd order ODE
def dSdt(S, t):
    theta, omega = S
    return [omega,
           np.sin(theta)]

# initial conditions
theta0 = np.pi/4
omega0 = 0
S0 = (theta0, omega0)

# solve the ode
t = np.linspace(0, 20, 100)
soln = odeint(dSdt, S0, t)
theta, omega = soln.T

plt.figure(3)
plt.plot(t, theta)
# p3.show()
plt.grid()


# Fourier transforms

x_series = np.linspace(0, 5*np.pi, 50)
y_series = np.sin(2*np.pi*x_series) + np.sin(3*np.pi*x_series) + 0.5*np.random.randn(len(x_series))
plt.figure(4)
plt.plot(x_series, y_series)
plt.grid()

from scipy.fft import fft, fftfreq
N = len(y_series)
yf = fft(y_series)[:N//2]
xf = fftfreq(N, np.diff(x)[0])[:N//2]

plt.figure(5)
plt.plot(xf, np.abs(yf))
plt.grid()
# p4.show()

# input()
plt.show()