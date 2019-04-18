#!/usr/bin/env python3

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import math
# import ipywidgets as widgets
# from IPython.display import display, Markdown, Latex
import sympy
from sympy import *
from scipy.optimize import minimize
from sympy.utilities.lambdify import lambdify

class Solver(object):
    def __init__(self, sensor_poses):
        x, y, dx, dy = symbols("x y dx dy")
        xdot, ydot = symbols("xdot ydot", cls=Function)
        drs = [symbols("dr{}".format(i)) for i in range(len(sensor_poses))]
        
        self.zz0 = np.concatenate([sum(sensor_poses)/len(sensor_poses), np.array([0, 0])])
        
        # Per Sesnor Error Function
        Es = [(xdot(x)*(x-p[0])+ydot(y)*(y-p[1]))*((x-p[0])**2 + (y-p[1])**2)**(-1/2) - dri 
                   for p, dri in zip(sensor_poses, drs)]
        
        # Objective Function (total error)
        F = sum((E**2 for E in Es))
        
        dxdtx = diff(xdot(x), x)
        dydtx = diff(ydot(y), y)
        dfdx = diff(F, x).subs(dxdtx, 0).subs(xdot(x), dx).subs(ydot(y), dy)
        dfdy = diff(F, y).subs(dydtx, 0).subs(xdot(x), dx).subs(ydot(y), dy)
        F = F.subs(xdot(x), dx).subs(ydot(y), dy)
        self.F_fn = lambdify((x, y, dx, dy, *drs), F, "numpy")
        dfdxdt = diff(F, dx)
        dfdydt = diff(F, dy)
        G = [dfdx, dfdy, dfdxdt, dfdydt] # gradient of F
        self.G_fn = [lambdify((x, y, dx, dy, *drs), g, "numpy") for g in G]
        
        self.F = F
        self.G = G
    
    def get_objective_expr(self):
        return self.F
    
    def get_gradient_expr(self):
        return self.G
    
    # Returns (position estimate, velocity estimate)
    def find_min(self, drs, method='SLSQP', **args):
        F_call = lambda zz: self.F_fn(*zz, *drs)
        G_call = lambda zz: np.array([g_fn(*zz, *drs) for g_fn in self.G_fn])
        
        rslts = minimize(F_call, self.zz0, method=method, jac=G_call)
        
        return rslts.x[:2], rslts.x[2:]
