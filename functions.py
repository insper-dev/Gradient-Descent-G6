# functions.py
# Function f and its gradient

from math import exp

def obj_f(x: float, y: float) -> float:
    '''
        Objective function f to be minimized.
    '''
    return x**2 + 4 * y**2 + x * y + 3 * x + y

def dfx(x: float, y: float) -> float:
    '''
        Partial derivative of f with respect to x.
    '''
    return 2 * x + y + 3

def dfy(x: float, y: float) -> float:
    '''
        Partial derivative of f with respect to y.
    '''
    return x + 8 * y + 1

def f_grad(x: float, y: float) -> float:
    '''
        Compute the gradient of f at (x, y).
    '''
    return dfx(x, y), dfy(x, y)


# Function g and its gradient

def obj_g(x: float, y: float) -> float:
    '''
        Objective function g to be minimized.
    '''
    return (x**2 + y**2 + 2)**0.5 + 2 * x**2 * exp(-y**2) + (x - 2)**2

def dgx(x: float, y: float) -> float:
    '''
        Partial derivative of g with respect to x.
    '''
    term1 = x
    term1_2 = ((x**2 + y**2 + 2)**0.5)
    term2 = - 4 * x**2 * exp(-y**2)
    term3 = 2 * (x - 2)
    if term1_2 == 0:
        raise ValueError("Division by zero in dgx calculation.")
    return term1/term1_2 + term2 + term3

def dgy(x: float, y: float) -> float:
    '''
        Partial derivative of g with respect to y.
    '''
    term1 = y / (x**2 + y**2 + 2)**0.5
    term2 = - 4 * x**2 * y * exp(-y**2)
    return term1 + term2
def g_grad(x: float, y: float) -> float:
    '''
        Compute the gradient of g at (x, y).
    '''
    return dgx(x, y), dgy(x, y)
# Function h and its gradient

def obj_h(x: float, y: float) -> float:
    '''
        Objective function h to be minimized.
    '''
    return 4 * exp(-x * x - y * y) + 3 * exp(-x * x - y * y + 4 * x + 6 * y - 13) - (x * x) / 4 - (y * y) / 6 + 2

def dhx(x: float, y: float) -> float:
    '''
        Partial derivative of h with respect to x.
    '''
    term1 = 4 * exp(-x * x - y * y) * (-2 * x)
    term2 = 3 * exp(-x * x - y * y + 4 * x + 6 * y - 13) * (-2 * x + 4)
    term3 = -x / 2
    return term1 + term2 + term3

def dhy(x: float, y: float) -> float:
    '''
        Partial derivative of h with respect to y.
    '''
    term1 = 4 * exp(-x * x - y * y) * (-2 * y)
    term2 = 3 * exp(-x * x - y * y + 4 * x + 6 * y - 13) * (-2 * y + 6)
    term3 = -y / 3
    return term1 + term2 + term3

def h_grad(x: float, y: float) -> float:
    '''
        Compute the gradient of h at (x, y).
    '''
    return dhx(x, y), dhy(x, y)