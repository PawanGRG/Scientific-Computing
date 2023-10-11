import numpy as np
import pylab as pl
from math import pi
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def foward_euler(u_j,mx,lmbda,p_j,q_j,delta_x,boundary_cond):
    """
    A function that calculates the foward euler method on PDEs with the boundary
    conditions: 'dirichlet', 'neumann' or 'periodic'

    :param u_j: An array containing values of u at current time step

    :param mx: The number of meshpoints in space

    :param lmbda: Mesh fourier number - dependent on delta t, delta x and the constant

    :param p_j: The value of the initial boundary

    :param q_j: The value of the finial boundary

    :param delta_x: The increment of of the space (meshpoints)

    :param boundary_cond: The boundary condition/type the user wants to use

    :return: u_jp1 matrix - the value of u at the next time step
    """
    if boundary_cond == 'dirichlet':
        # Creating a tridiagonal matrix
        A_fe = diags([lmbda, 1 - 2 * lmbda, lmbda], [-1, 0, 1], shape=(mx - 2, mx - 2)).toarray()
        b_cond = np.zeros(mx-2)
        # Set the first boundary value as p
        b_cond[0] = p_j
        b_cond[-1] = q_j # Set the last boundary value as q
        return A_fe @ u_j[1:-1] + lmbda * b_cond
    elif boundary_cond == 'neumann':
        # Creating a tridiagonal matrix
        A_fe = diags([lmbda, 1 - 2 * lmbda, lmbda], [-1, 0, 1], shape=(mx, mx)).toarray()
        # Neumann matrix change
        A_fe[0, 1] = 2 * lmbda
        A_fe[-2, -1] = 2 * lmbda
        b_cond = np.zeros(mx)
        b_cond[0] = -p_j # Set the first boundary value as -p
        b_cond[-1] = q_j # Set the last boundary value as q
        return A_fe @ u_j + 2 * lmbda * delta_x * b_cond
    elif boundary_cond == 'periodic':
        A_periodic = diags([lmbda, 1 -2 * lmbda, lmbda], [-1, 0, 1], shape=(mx-1, mx-1)).toarray()
        # Periodic matrix change
        A_periodic[0, -1] = lmbda
        A_periodic[-1, 0] = lmbda
        return A_periodic @ u_j[:-1]

    else:
        print("print choose a boundary type: 'dirichlet' or 'neumann or 'periodic'")

def backward_euler(u_j,mx,lmbda,p_j,q_j,delta_x,boundary_cond):
    """
    A function that calculates the backward euler method on PDEs with the boundary
    conditions: 'dirichlet', 'neumann' or 'periodic'

    :param u_j: An array containing values of u at current time step

    :param mx: The number of meshpoints in space

    :param lmbda: Mesh fourier number - dependent on delta t, delta x and the constant

    :param p_j: The value of the initial boundary

    :param q_j: The value of the finial boundary

    :param delta_x: The increment of of the space (meshpoints)

    :param boundary_cond: The boundary condition/type the user wants to use

    :return: u_jp1 matrix - the value of u at the next time step
    """

    if boundary_cond == 'dirichlet':
        # Creating a tridiagonal matrix
        A_be = diags([-lmbda, 1 + 2 * lmbda, -lmbda], [-1, 0, 1], shape=(mx - 2, mx - 2)).toarray()
        b_cond = np.zeros(mx-2)
        b_cond[0] = p_j
        b_cond[-1] = q_j
        return spsolve(A_be, (u_j[1:-1]) + (lmbda * b_cond))
    elif boundary_cond == 'neumann':
        # Creating a tridiagonal matrix
        A_be = diags([-lmbda, 1 + 2 * lmbda, -lmbda], [-1, 0, 1], shape=(mx, mx)).toarray()
        # Neumann matrix change
        A_be[0,1] = 2 * lmbda
        A_be[-1,-2] = 2 * lmbda
        b_cond = np.zeros(mx)
        b_cond[0] = -p_j
        b_cond[-1] = q_j
        return spsolve(A_be, u_j + 2 * lmbda * delta_x * b_cond)
    elif boundary_cond == 'periodic':
        # Creating a tridiagonal matrix
        A_periodic = diags([-lmbda, 1 + 2 * lmbda, -lmbda], [-1, 0, 1], shape=(mx-1, mx-1)).toarray()
        # Periodic matrix change
        A_periodic[0,-1] = lmbda
        A_periodic[-1,0] = lmbda
        return spsolve(A_periodic, u_j[:-1])
    else:
        print("Please choose a boundary type: 'dirichlet' or 'neumann' or 'periodic")

def crank_nicholson(u_j,mx,lmbda,p_j,q_j,delta_x,boundary_cond):
    """
    A function that calculates the crank nicholson method on PDEs with the boundary
    conditions: 'dirichlet', 'neumann' or 'periodic'

    :param u_j: An array containing values of u at current time step

    :param mx: The number of meshpoints in space

    :param lmbda: Mesh fourier number - dependent on delta t, delta x and the constant

    :param p_j: The value of the initial boundary

    :param q_j: The value of the finial boundary

    :param delta_x: The increment of of the space (meshpoints)

    :param boundary_cond: The boundary condition/type the user wants to use

    :return: u_jp1 matrix - the value of u at the next time step
    """

    if boundary_cond == 'dirichlet':
        # Creating two tridiagonal matrix since Crank Nicholson requires two
        A_cn = diags([-lmbda/2, 1 + lmbda, -lmbda/2], [-1, 0, 1], shape=(mx - 2, mx - 2)).toarray()
        B_cn = diags([lmbda/2, 1 - lmbda, lmbda/2], [-1, 0, 1], shape=(mx - 2, mx - 2)).toarray()
        b_cond = np.zeros(mx-2)
        b_cond[0] = p_j
        b_cond[-1] = q_j
        return spsolve(A_cn, B_cn @ u_j[1:-1] + lmbda * b_cond)
    elif boundary_cond == 'neumann':
        # Creating two tridiagonal matrix since Crank Nicholson requires two
        A_cn = diags([-lmbda/2, 1 + lmbda, -lmbda/2], [-1, 0, 1], shape=(mx, mx)).toarray()
        B_cn = diags([lmbda/2, 1 - lmbda, lmbda/2], [-1, 0, 1], shape=(mx, mx)).toarray()
        # Neumann matrix change
        A_cn[0, 1] = -lmbda
        A_cn[-1, -2] = -lmbda
        B_cn[0, 1] = lmbda
        B_cn[-1, -2] = lmbda
        b_cond = np.zeros(mx)
        b_cond[0] = -p_j
        b_cond[-1] = q_j
        return spsolve(A_cn,B_cn @ u_j + 2 * lmbda * delta_x * b_cond)
    elif boundary_cond == 'periodic':
        # Creating two tridiagonal matrix since Crank Nicholson requires two
        A_cn = diags([-lmbda/2, 1 + lmbda, -lmbda/2], [-1, 0, 1], shape=(mx-1, mx-1)).toarray()
        B_cn = diags([lmbda/2, 1 - lmbda, lmbda/2], [-1, 0, 1], shape=(mx-1, mx-1)).toarray()
        # Periodic matrix change
        A_cn[0, -1] = -lmbda/2
        A_cn[-1, 0] = -lmbda/2
        B_cn[0, -1] = lmbda/2
        B_cn[-1, 0] = lmbda/2
        return spsolve(A_cn,B_cn @ u_j[0:-1])
    else:
        print("Please choose a boundary type: 'dirichlet' or 'neumann' or 'periodic")

def defualt_boundary_condition(t):
    """
    A function that returns the boundary conditions as 0 for PDEs if the boundary conditions
    are not specified by the user

    :param t: The time the function solves the PDE

    :return: default boundary condiiton
    """
    return 0


def pde_solver(func,L,T,mx,mt,vary_par,dict,pde_method,boundary_cond,p=defualt_boundary_condition,
               q=defualt_boundary_condition,steady_states= False):
    """
    Function that solves the pde, the user can specify which method they want to solve the pde - 'foward euler'
    'backward euler' or 'crank nicholson'. From that the user then can specify the boundary type they want to use -
    'dirichlet', 'neumann' or 'periodic'

    :param func: The PDE to be solved

    :param L: The length of the space domain

    :param T: The time to solve the PDE for

    :param mx: The number of meshpoints in space

    :param mt: The number of meshpoints in time

    :param pde_method: The PDE method the user wants to use to solve the PDE - 'foward euler','backward euler' or '
                       crank nicholson'

    :param boundary_cond: The boundary type/condition the user wants to use to solve the PDE - 'dirichlet', 'neumann'
                          or 'periodic'

    :param p: The value of the initial boundary

    :param q: The value of the finial boundary

    :param vary_par: The parameter you want to vary

    :param dict: Dictionary of all the constant/parameters in the PDE

    :return: An array of u_j values
    """

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx)  # mesh points in space
    t = np.linspace(0, T, mt)  # mesh points in time
    delta_x = x[1] - x[0]  # gridspacing in x
    delta_t = t[1] - t[0]  # gridspacing in t
    lmbda = dict[vary_par] * delta_t / (delta_x ** 2)  # mesh fourier number


    # Set up the solution variables
    u_j = np.zeros(x.size)  # u at current time step
    u_jp1 = np.zeros(x.size)  # u at next time step
    steady_states =[] # steady states

    for i in range(0, mx):
        u_j[i] = func(x[i])

    if pde_method == 'forward euler':
        for i in range(0,mt):
            p_j = p(i)
            q_j = q(i)
            if boundary_cond == 'dirichlet':
                u_jp1[1:-1] = foward_euler(u_j,mx,lmbda,p_j,q_j,delta_x,boundary_cond='dirichlet')
                u_jp1[0] = p_j
                u_jp1[-1] = q_j
            elif boundary_cond == 'neumann':
                    u_jp1 = foward_euler(u_j,mx,lmbda,p_j,q_j,delta_x,boundary_cond='neumann')
            elif boundary_cond == 'periodic':
                u_jp1[0:-1] = foward_euler(u_j,mx,lmbda,p_j,q_j,delta_x,boundary_cond='periodic')
                u_jp1[-1] = u_jp1[0]
            else:
                print("Please choose a boundary type: 'dirichlet' or 'neumann' or 'periodic ")

            if steady_states == True:
                if np.isclose(u_j,u_jp1,rtol=1e-5) == True:
                    steady_states.append(t[i])

            # Save u_j at time t[j+1]
            u_j[:] = u_jp1[:]
    elif pde_method == 'backward euler':
        for i in range(0,mt):
            p_j = p(i)
            q_j = q(i)
            if boundary_cond == 'dirichlet':
                u_jp1[1:-1] = backward_euler(u_j,mx,lmbda,p_j,q_j,delta_x,boundary_cond='dirichlet')
                u_jp1[0] = p_j
                u_jp1[-1] = q_j
            elif boundary_cond == 'neumann':
                u_jp1 = backward_euler(u_j,mx,lmbda,p_j,q_j,delta_x,boundary_cond='neumann')
            elif boundary_cond == 'periodic':
                u_jp1[0:-1] = backward_euler(u_j,mx,lmbda,p_j,q_j,delta_x,boundary_cond='periodic')
                u_jp1[-1] = u_jp1[0]
            else:
                print("Please choose a boundary type: 'dirichlet' or 'neumann' or 'periodic ")

            # Save u_j at time t[j+1]
            if steady_states == True:
                if np.isclose(u_j,u_jp1,rtol=1e-5) == True:
                    steady_states.append(t[i])

            u_j[:] = u_jp1[:]
    elif pde_method == 'crank nicholson':
        for i in range(0,mt):
            p_j = p(i)
            q_j = q(i)
            if boundary_cond == 'dirichlet':
                u_jp1[1:-1] = crank_nicholson(u_j,mx,lmbda,p_j,q_j,delta_x,boundary_cond='dirichlet')
                u_jp1[0] = p_j
                u_jp1[-1] = q_j
            elif boundary_cond == 'neumann':
                u_jp1 = crank_nicholson(u_j,mx,lmbda,p_j,q_j,delta_x,boundary_cond='neumann')
            elif boundary_cond == 'periodic':
                u_jp1[0:-1] = crank_nicholson(u_j,mx,lmbda,p_j,q_j,delta_x,boundary_cond='periodic')
                u_jp1[-1] = u_jp1[0]
            else:
                print("Please choose a boundary type: 'dirichlet' or 'neumann' or 'periodic ")
            if steady_states == True:
                if np.isclose(u_j,u_jp1,rtol=1e-5) == True:
                    steady_states.append(t[i])

            # Save u_j at time t[j+1]
            u_j[:] = u_jp1[:]
    else:
        print("Please choose a PDE method: 'forward euler' or 'backward euler' or 'crank nicholson")

    if steady_states == True:
        if len(steady_states) > 0:
            print("steady states founded:",steady_states[0])
            print("u_j:",u_j)

        else:
            print("no steady states found")
            print("u_j:", u_j)

    return u_j
