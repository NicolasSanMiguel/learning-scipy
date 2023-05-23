# Nicolas San Miguel
# April 2023

# INTRO TO BASIC FUNCTIONS
# This is a simple sandbox to explore the scipy library and its functions

import scipy
from scipy.integrate import quad as integ1
from scipy.integrate import dblquad as integ2
from scipy.integrate import nquad as integN
import numpy as np
import matplotlib.pyplot as plt

# # # # # # # # Integration

def fun1(x, a, b): # integrates over x as its the first variable, a, b are defined as parameters
    return a*x**3 + b*x

# single integration
i = scipy.integrate.quad(lambda x:scipy.special.exp10(x),0,1)
print("single integral:", i)

area = integ1(fun1,-1,1,args=(-1,5))
print("single integral of fun1:", area)

# double integration
e = lambda x, y: x*y**2
f = lambda x: 81
g = lambda x: -5*x
i = scipy.integrate.dblquad(e,0,1,f,g) # returns two values, 1st is value of integral, 2nd is absolute error estimate
print("double integral:", i)

def fun2(x, y, a, b): # integrates over x as its the first variable, a, b are defined as parameters
    return a*x**3 + b*x*y
area = integ2(fun2,-1,1, 0,5, args=(-1,5))
print("double integral of fun2:", area)

# lambda functions are useful for holding bounds
y_lb = lambda y: y   # lower bound is y
y_ub = lambda y: 2*y # upper bound is 2y
area = integ2(fun2,-1,1, y_lb,y_ub, args=(-1,5))
print("double integral of fun2 with lambda bounds:", area)

def fun3(x, y, z, a, b): # integrates over x as its the first variable, a, b are defined as parameters
    return a*x**3 + b*x*y -z
area = integN(fun3,[[-1,1], [3,7], [1,2]], args=(-1,5))
print("triple integral of fun3:", area)

quit()

# # # # # # # # Fourier Transforms
x = np.array([1, 2, 3, 4])
y = scipy.fftpack.fft(x)

print("this is x:",x)
print("this is the fourier transform of x:",y)

# this is the inverse fourier transform
y = scipy.fftpack.ifft(x)
print("this is the inverse fourier transform of x:",y)

# # # # # # # # Linear Algebra
a = np.array([[1, 2],[3, 4]])
b = scipy.linalg.inv(a)

# # # # # # # # Interpolation
x = np.arange(5,20)
# y = np.exp(x)
# y = x**2 -3
y = np.sin(x)
f = scipy.interpolate.interp1d(x,y) # returns an interpolation function
x1 = np.arange(6,12)
y1 = f(x1)

x2 = np.arange(5,20,0.1)
y2 = np.sin(x2)
x3 = np.arange(15,19,0.1)
fnew = scipy.interpolate.interp1d(x2,y2) # returns an interpolation function
y3 = fnew(x3)

plt.plot(x, y, 'o', x1, y1, '--')
plt.plot(x2, y2, '.', x3, y3, '--') # using the new interpolated function

# plt.show()











