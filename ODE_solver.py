import numpy as np
import math
import matplotlib.pyplot as plt
from math import nan


def euler_step(f,x,t,h,dict):
    """
    Function that updates a single euler step

    :param f: The ODE function that you are applying euler method
    :param x: Initial x value (at time = 0)
    :param t: t value
    :param h: Step size
    :param dict: Dictionary of the constants in the ODE
    :return: Updated value of x
    """
    x_next = x + f(x,t,dict)*h
    return x_next


def rk4_step(f,x,t,h,dict):
    """
    Function that updates a single rk4 step

    :param f: The ODE function you are appyling rk4 to
    :param x: Initial x value (at time = 0)
    :param t: Initial t value
    :param h: Step size
    :param dict: Dictionary of the constants in the ODE
    :return: Updated value of x
    """
    k1 = f(x, t, dict)
    k2 = f(x + k1 * (h / 2), t + h / 2, dict)
    k3 = f(x + k2 * (h / 2), t + h / 2, dict)
    k4 = f(x + k3 * h, t + h, dict)
    x_next = (x + ((k1 + 2 * k2 + 2 * k3 + k4) / 6) * h)
    return x_next

def solve_to(solver,f,x0,t0,tf,h,dict):
    """
    Function that solves from one x and t value to the next x and t value

    :param solver: Chooses the type of solver, either euler_step or rk4
    :param x0: Initial x value (at t = 0)
    :param t0: Initial t value
    :param tf: Final t value
    :param h: Step size
    :param dict: Dictionary of the constants in the ODE
    :return: Numpy array of the next value of the variables
    """

    x = x0
    t = t0
    # This is so the ODE solver does not overshoot
    while t < tf:
        if tf < (t+h):
            h = tf - t
        x = solver(f,x,t,h,dict)
        t += h
    return x


def solve_ode(solver,f,x,time,h,dict):
    """
    Function to solve the ode using either euler_step or rk4

    :param solver: Chooses the type of solver, either euler_step or rk4
    :param x: Initial x value (at t = 0)
    :param t: Initial t value
    :param tf: Final t value
    :param h: Step size
    :return: Numpy array of the solution, the dimension of the array will match
             the dimension of the ODE system
    """
    # Determines the number of iterations the for loop needs to go through
    # print(time)
    A = [x] # An array to store the values of the euler or rk4 method values
    for i in range(len(time)-1):
        ans = solve_to(solver,f,A[i],time[i],time[i+1],h,dict)
        A.append(ans)
    return A