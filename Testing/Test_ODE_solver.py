import unittest
from github.emat30008.Solvers import ODE_Solver as od
import numpy as np
from math import nan

def predator_prey(u,t,dict):
    x = u[0]
    y = u[1]
    xdot = x*(1-x) - (dict["a"]*x*y)/(dict["d"]+x)
    ydot = dict["b"]*y*(1-(y/x))
    return np.array([xdot,ydot])


constant = {"a":1,"b":0.1,"d":0.1}


class TestODE(unittest.TestCase):
    def test_euler_step(self):
        self.assertEqual(round(od.euler_step(predator_prey,np.array([2,1]),t=nan,h=0.1,dict=constant)[0],3),1.705)
        self.assertEqual(round(od.euler_step(predator_prey, np.array([2, 1]), t=nan, h=0.1, dict=constant)[1],3),1.005)


    def test_rk4_step(self):
        self.assertEqual(round(od.rk4_step(predator_prey,np.array([2,1]),t=nan,h=0.1,dict=constant)[0],3),1.743)
        self.assertEqual(round(od.rk4_step(predator_prey, np.array([2, 1]), t=nan, h=0.1, dict=constant)[1],3),1.005)

    def test_solve_to(self):
        self.assertEqual(round(od.solve_to(od.euler_step,predator_prey,np.array([2,1]),0,2,h=0.1,dict=constant)[0],3),0.125)
        self.assertEqual(round(od.solve_to(od.euler_step, predator_prey, np.array([2, 1]),0,2,h=0.1,dict=constant)[1],3),0.836)
        self.assertEqual(round(od.solve_to(od.rk4_step,predator_prey,np.array([2,1]),0,2,h=0.1,dict=constant)[0],3),0.173)
        self.assertEqual(round(od.solve_to(od.rk4_step, predator_prey, np.array([2, 1]),0,2,h=0.1,dict=constant)[1],3),0.858)


    def test_solve_ode(self):
        self.assertEqual(round(od.solve_ode(od.euler_step,predator_prey,np.array([2,1]),np.linspace(0,2,50),h = 0.1,dict=constant)[-1][0],3),0.154)
        self.assertEqual(round(od.solve_ode(od.euler_step, predator_prey, np.array([2, 1]),np.linspace(0,2,50),h = 0.1, dict=constant)[-1][1],3),0.850)
        self.assertEqual(round(od.solve_ode(od.rk4_step,predator_prey,np.array([2,1]),np.linspace(0,2,50),h = 0.1,dict=constant)[-1][0],3),0.173)
        self.assertEqual(round(od.solve_ode(od.rk4_step, predator_prey, np.array([2, 1]),np.linspace(0,2,50),h = 0.1, dict=constant)[-1][1],3),0.858)

if __name__ == '__main__':
    if __name__ == '__main__':
        unittest()
