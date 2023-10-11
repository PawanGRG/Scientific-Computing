import unittest
import numpy as np
from github.emat30008.Solvers import Numerical_Shooting as ns
from github.emat30008.Solvers import ODE_Solver as od
import matplotlib.pyplot as plt

def hopf_explicit(theta, t, dict):

    u1 = np.sqrt(dict["B"])*np.cos(t + theta)
    u2 = np.sqrt(dict["B"])*np.sin(t + theta)

    return np.array([u1,u2])

def hopf(u, t, dict):

    u1 = u[0]
    u2 = u[1]
    du1dt = dict["B"]*u1 - u2 + dict["sigma"]*u1*(u1**2 + u2**2)
    du2dt = u1 + dict["B"]*u2 + dict["sigma"]*u2*(u1**2 + u2**2)

    return np.array([du1dt, du2dt])

class testcode(unittest.TestCase):
    def test(self):
        const = {"sigma": -1, "B": 0.5}
        u0 = np.array([3, 6, 9])
        results = ns.root_finder(hopf,guess=u0,dict=const);
        x0 = results[0]
        T = results[-1]
        t_span = np.linspace(0, 50, 51)
        sol = od.solve_ode(od.rk4_step,hopf,x=x0,time=t_span,h=0.01,dict=const)
        actual =  hopf_explicit(T,t_span[-1],dict=const)
        sol_x = [i[0] for i in sol]
        sol_y = [i[1] for i in sol]
        plt.plot(t_span,sol_x); plt.xlabel('x')
        plt.plot(t_span,sol_y); plt.ylabel('y')
        plt.legend()
        plt.grid()
        plt.xlabel('Time')
        plt.title('Hopf bifurcation ODE ')
        plt.show()
        print("yo",sol[-1])
        print(actual)

        check = abs(abs(sol[-1]) - abs(actual))

        self.assertTrue(np.all(check < 1e-5))
