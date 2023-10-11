import numpy as np
from github.emat30008.Solvers import Numerical_Shooting as ns
from scipy.optimize import fsolve
from math import nan
from github.emat30008.Solvers import PDE_Solver as pde
import matplotlib.pyplot as plt


def natural_parameter_continuation(f,u0,par_start,par_end,vary_par,increment,dict,T = None,L = None,discretisation = 'polynomial',mx = 10,
                                   mt = 100,p = pde.defualt_boundary_condition,q = pde.defualt_boundary_condition,
                                   pde_method = 'crank nicholson',boundary_cond = 'neumann' ):
    """
    A function that performs natural parameter continuation on polynomials and ODEs. Natural parameter continuation
    investigates the affect the constant has on the limit cycles of the ODEs.

    :param f: The ODE, takes in the state vector (initial guess) and returns the RHS of the equation

    :param u0: Initial guess (numpy array), if discretisation is polynomial then the initial guess = [x0,y0].
               If discretisation is shooting then the initial guess must include a guess of time period for
               the limit cycle of the ODE = [x0,y0,T]

    :param par_start: The start value of the varying parameter

    :param par_end: The end value of the varying parameter

    :param vary_param: The parameter that is varied throughout the continuation process, the input must be a string

    :param increment : The size increments you want to increase vary_par by

    :param dict: Dictonairy of all the constants in the ODE

    :param T: The time to solve the PDE for

    :param L: The length of the space domain

    :param discretisation: The discretisation to use, either 'polynomial' or 'shooting' or 'steady

    :param mx: The number of meshpoints in space

    :param mt: The number of meshpoints in time

    :param p: The value of the initial boundary

    :param q: The value of the finial boundary

    :param pde_method:The PDE method the user wants to use to solve the PDE - 'foward euler','backward euler' or 'steady
                      states;
                       crank nicholson'

    :param boundary_cond: The boundary type/condition the user wants to use to solve the PDE - 'dirichlet', 'neumann'
                          or 'periodic'


    :return: Returns 2 numpy arrays. The 'sol' array represents the corrected initial guesses of the polynomial or the
             ODE depending on the discretisation chosen. The 'par_arrray' array represents the range of the varied
             parameter.
    """
    sol = [] # Empty array to store the solution values of
    par_array = [] # Array to store the parameter values
    iteration = 0
    # The number of iterations the loop must perform
    n = ((par_end-par_start)/increment)
    # Solving regular polynomials using natural parameter continuation
    if discretisation == 'polynomial':
        while iteration <= (round(n)+1): # Looping for every value of the parameter array
            # Solving the polynomial for a specific parameter value
            ans = fsolve(f,u0,args = (u0,dict))
            # Uses the solution as the initial conditions in the next iteration
            sol.append(ans) # Appending the solution to an empty array
            u0 = ans
            # Increasing the parameter by the increment (step size)
            dict[vary_par] += increment
            par_array.append(dict[vary_par]) # Appending the parameter values to an empty array
            iteration += 1

    # Solving ODEs or system of ODEs using natural parameter continuation
    elif discretisation == 'shooting': # Looping for every value of the parameter array
        while iteration <= (round(n)+1):
            # Solving the polynomial for a specific parameter value
            ans = ns.root_finder(f,u0,dict) # Since its numerical shooting, an initial guess of time period is needed
            sol.append(ans) # Appending the solution to an empty array
            # Uses the solution as the initial conditions in the next iteration
            u0 = np.append(ans[0],ans[1])
            # Increasing the parameter by the increment (step size)
            dict[vary_par] += increment
            par_array.append(dict[vary_par]) # Appending the parameter values to an empty array
            iteration += 1
    elif discretisation == 'steady states':
        while iteration <= (round(n)+1):
            ans = pde.pde_solver(f,L,T,mx,mt,vary_par,dict,pde_method,boundary_cond,p,q,steady_states=True)
            sol.append(ans[-1]) # Append the last value
            # increasing the varying param by the increment
            dict[vary_par] += increment
            # Appending the vary param
            par_array.append(dict[vary_par])
            iteration += 1

    return sol,par_array

def pseudo_euqation(u0,f,dict,vary_par,secant,predict,discretisation):
    """
    Function that calculates the phase condition at dx/dt(0) and the corrected
    initial condition and the corrected time period depending on the discretisation user
    specfies

    :param f: The ODE, takes in the state vector (initial guess) and returns the RHS of the equation

    :param u0: Initial guess (numpy array), if discretisation is polynomial then the initial guess = [x0,y0].
               If discretisation is shooting then the initial guess must include a guess of time period for
               the limit cycle of the ODE = [x0,y0,T]

    :param vary_par: The parameter that is varied throughout the continuation process, the input must be a string

    :param secant: An approximation to the tangent using the two initial conditions

    :param predict: The predicition of the solution

    :param dict: Dictonairy of all the constants in the ODE

    :param discretisation: The discretisation to use, either 'polynomial' or 'shooting'

    :return: Set of equations for fsolve to solve
    """
    dict[vary_par] = u0[0]
    if discretisation == 'polynomial':
        # Finding the equilibirum
        ans = f(u0[1:],0,dict)
    if discretisation == 'shooting':
        sol = ns.root_finder(f,u0[1:],dict)
        ans = np.append(sol[0],sol[1])
    # Psuedo arc length equation
    y = secant.dot(u0 - predict) # Dot product
    return (np.append(y,ans))




def pseudo_arc_length(f,u0,par_end,vary_par,increment,dict,T,L,discretisation = 'polynomial',mx = 10,
                                   mt = 100,p = pde.defualt_boundary_condition,q = pde.defualt_boundary_condition,
                                   pde_method = 'crank nicholson',boundary_cond = 'neumann'):
    """
    Function that performs pseudo arc length continuation on polynomials and ODEs. The pseudo arc length
    continuation solves the problem of natural arc length continuation where it fails to converge to
    find a limit cycle.

    :param f: The ODE, takes in the state vector (initial guess) and returns the RHS of the equation

    :param u0: Initial guess (numpy array), if discretisation is polynomial then the initial guess = [x0,y0].
               If discretisation is shooting then the initial guess must include a guess of time period for
               the limit cycle of the ODE = [x0,y0,T]

    :param par_start: The start value of the varying parameter

    :param par_end: The end value of the varying parameter

    :param vary_param: The parameter that is varied throughout the continuation process, the input must be a string

    :param increment : The size increments you want to increase vary_par by

    :param dict: Dictonairy of all the constants in the ODE

    :param T: The time to solve the PDE for

    :param L: The length of the space domain

    :param discretisation: The discretisation to use, either 'polynomial' or 'shooting' or 'steady

    :param mx: The number of meshpoints in space

    :param mt: The number of meshpoints in time

    :param p: The value of the initial boundary

    :param q: The value of the finial boundary

    :param pde_method:The PDE method the user wants to use to solve the PDE - 'foward euler','backward euler' or 'steady
                      states;
                       crank nicholson'

    :param boundary_cond: The boundary type/condition the user wants to use to solve the PDE - 'dirichlet', 'neumann'
                          or 'periodic'


    :return: Returns 2 numpy arrays. The 'sol' array represents the corrected initial guesses of the polynomial or the
             ODE depending on the discretisation chosen. The 'par_arrray' array represents the range of the varied
             parameter.

    :return: Numpy array containing the varying paramater and the solution
    """

    iteration = 0
    # Parameter value and the state vector appended
    solution = [np.append(dict[vary_par],u0)]
    yo = [u0]
    if discretisation == 'polynomial':
        while dict[vary_par] < par_end:
            if iteration < 2: # Psuedo arc length only operates after its got 2 initial conditions
                ans = fsolve(f,u0,args=(f,dict))
                u0 = ans
                dict[vary_par] += increment
                v = np.append(dict[vary_par],u0)
                solution.append(v)
                iteration += 1
            else:
                # Secant value that is dependant on the initial values
                secant = solution[-1] - solution[-2] # Secant is an approximation to tangent
                # Predicition of the true solution
                predict = solution[-1] + secant
                ans = fsolve(pseudo_euqation,predict,args=(f,dict,vary_par,secant,predict,discretisation))
                dict[vary_par] = ans[0]
                predict = ans[1:]
                solution.append(ans)
                iteration += 1
    elif discretisation == 'shooting':
        while dict[vary_par] < par_end:
            if iteration < 2:
                # print(dict[vary_par])
                ans = ns.root_finder(f,u0,dict)
                yo.append(ans[0])
                # Uses the solution as the initial conditions in the next iteration
                u0 = np.append(ans[0], ans[1])
                # Increasing the parameter by the increment (step size)
                dict[vary_par] += increment
                v = np.append(dict[vary_par],u0)
                solution.append(v)
                iteration += 1
            else:
                # Secant value that is dependant on the initial values
                secant = solution[-1] - solution[-2]  # Secant is an approximation to tangent
                # Predicition of the true solution
                x0 = solution[-1] + secant
                predict = solution[-1] + secant
                ans = fsolve(pseudo_euqation,x0,args=(f,dict,vary_par,secant,predict,discretisation))
                dict[vary_par] = ans[0] # First value of the fsolve output
                x0 = ans[1:-1] # State vector
                solution.append(ans)  # Append the ans to the solution array
                iteration += 1

    elif discretisation == 'steady states':
        while dict[vary_par] < par_end: # iterating till the varying param > par_end
            ans = pde.pde_solver(f,L,T,mx,mt,vary_par,dict,pde_method,boundary_cond,p,q,steady_states=True)
            v = np.append(dict[vary_par],ans[-1])
            # appends the solution to the array
            solution.append(v)
            # increasing the varying param by the increment
            dict[vary_par] += increment
            iteration += 1

    return solution
