import numpy as np
from scipy.integrate import solve_ivp,odeint
from scipy.optimize import fsolve, root
from math import nan
from github.emat30008.Solvers import ODE_Solver as od
import matplotlib.pyplot as plt


def standard_phase_condition(f,u,t,dict):
    """
    Function that calculates the derivative at time = 0
    :param f: The system of ODEs
    :param dict: Dictionary of all the constants in the ODE
    :param u: Initial conditions that is being inputted into the system of ODEs
    :return: Numpy array of the derivative at dxdt(0) giving the phase condition
    """
    phase_condition = f(u[:-1],t,dict)[0]

    return np.array([phase_condition])

def initial_guess(guess,func,dict):
    """
    Function that takes the initial guess for the ODE and solves the ODE with the
    initial conditions. Returns a set of equations.

    :param guess: Initial guess of the conditions, set up as an array [x0,y0,z0,T]
    :param func: The system of ODE or ODE that you are applying numerical shooting to.
    :param dict: Dictionary of all the constants of the system of ODE.
    :return: Numpy array of x condition, y condition and phase condition
    """
    # Setting the initial guess as a numpy arrayy
    initial = np.array(guess)
    # Using my solve_ode (numerical integrator) to solve initial value problem for a system of ODEs
    solution = od.solve_ode(od.rk4_step,func,initial[:-1],np.linspace(0,initial[-1],num=10),h=0.01,dict = dict)
    """ Limit cycle and Phase condition of the ODE"""
    cond = initial[:-1] - solution[-1]
    pc = standard_phase_condition(func,guess,nan,dict)
    return np.append(cond,pc)

def root_finder(func,guess,dict):
    """
    Function that uses a root finder to find the initial conditions and time period
    for the ODE

    :param func: The system of ODES that numerical shooting is being applied to
    :param guess: Initial guess of the conditions, set up as an array [x0,y0,z0,T]
    :param dict: Dictionary of all the constants of the system of ODE
    :return: Numpy array of the correct initial conditions of the ODES and the time period of the ODEs
    """

    u0 = np.array(guess)
    # Using fsolve (numerical root finder) to try equate the equations from the initial_guess to zero
    solution = fsolve(initial_guess, u0, args=(func,dict))
    u0 = solution[:-1] # Equates the variable u0 to the corrected initial values
    T = solution[-1] # Equates the variable T to the corrected time period
    return u0,T
