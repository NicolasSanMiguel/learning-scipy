
from scipy.optimize import minimize

def fun1(x): # simple quadratic function
    return (x-1)**2
out = minimize(fun1, x0=2)
print("x_minimized", out.x)
print("x_minimized",out) # more verbose output

# # # # # # multivariate quadratic function -- bounded area defined by constraints
# can use x, y or x[0], x[1]
fun2 = lambda x: (x[0] - 1)**2 + (x[1] -3)**2 # multivariate quadratic function
constr = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 1},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 0.6 * x[1] + 4.5},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 1 * x[1] + 7})
bds = ((0, None), (0, None))
out = minimize(fun2, (2, 0), bounds=bds, constraints=constr)
print("x_minimized",out.x)
print("x_minimized",out) # more verbose output



