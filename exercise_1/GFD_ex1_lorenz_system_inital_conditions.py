import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import gridspec

# ==============================================================================
# GFD exercise 1, Problem 1.3: The Lorenz System
# Beth Saunders 03/10/2025
# ==============================================================================

# Specify path for plots
current_filepath = os.getcwd().replace('\\', '/') + '/'
assert os.path.exists(current_filepath)

# Assuming typical parameters for the atmosphere
p = 10  # Prandtl number, which describes the ratio of momentum diffusivity (kinematic viscosity) and thermal diffusivity
beta = 8 / 3  # a geometrical factor (all the ‘proportionalities’ in the variables arise from the fact that Lorenz solved the system in a non-dimensional form).

# timestep
dt = 0.01

# user-set (reduced) Rayleigh Number that determines whether the heat transfer is primarily in the form of conduction or convection

r_list = [28, 0.5, 22]

# Time vector
t = np.arange(0, 100 + dt, dt)
nt = len(t)


# ==============================================================================
# Runge-Kutta integration
# ==============================================================================
def rk4_lorenz(X0, Y0, Z0, p, beta, r, dt, nt):
    """
    Using the fourth-order Runge-Kutta method (given in Appendix 1) to solve an example from Lorenz (1993).

    :param X0: Initial X condition
    :param Y0: Initial Y condition
    :param Z0: Initial Z condition
    :param p: Prandtl number, which describes the ratio of momentum diffusivity (kinematic viscosity) and thermal diffusivity
    :param beta: a geometrical factor (all the ‘proportionalities’ in the variables arise from the fact that Lorenz solved the system in a non-dimensional form).
    :param r: (reduced) Rayleigh Number that determines whether the heat transfer is primarily in the form of conduction or convection
    :param dt: timestep
    :param nt: time vector
    :return: X, Y, Z
    """

    X = np.zeros(nt)
    Y = np.zeros(nt)
    Z = np.zeros(nt)
    X[0] = X0
    Y[0] = Y0
    Z[0] = Z0

    for it in range(1, nt):
        # TASK 1: add the equations for dYdt, dZdt, Y1, and Z1
        dXdt = p * (Y[it - 1] - X[it - 1])
        dYdt = (X[it - 1] * (r - Z[it - 1])) - Y[it - 1]
        dZdt = (X[it - 1] * Y[it - 1]) - (beta * Z[it - 1])

        X1 = X[it - 1] + (dXdt * (dt / 2))
        Y1 = Y[it - 1] + (dYdt * (dt / 2))
        Z1 = Z[it - 1] + (dZdt * (dt / 2))

        # TASK 2: add the equations for dYdt, dZdt, Y2, and Z2
        dXdt = p * (Y1 - X1)
        dYdt = (X1 * (r - Z1)) - Y1
        dZdt = (X1 * Y1) - (beta * Z1)

        X2 = X[it - 1] + (dXdt * (dt / 2))
        Y2 = Y[it - 1] + (dYdt * (dt / 2))
        Z2 = Z[it - 1] + (dZdt * (dt / 2))

        dXdt = p * (Y2 - X2)
        dYdt = (X2 * (r - Z2)) - Y2
        dZdt = (X2 * Y2) - (beta * Z2)
        X3 = X[it - 1] + (dXdt * dt)
        Y3 = Y[it - 1] + (dYdt * dt)
        Z3 = Z[it - 1] + (dZdt * dt)

        dXdt = p * (Y3 - X3)
        dYdt = (X3 * (r - Z3)) - Y3
        dZdt = (X3 * Y3) - (beta * Z3)
        X4 = X[it - 1] - (dXdt * (dt / 2))
        Y4 = Y[it - 1] - (dYdt * (dt / 2))
        Z4 = Z[it - 1] - (dZdt * (dt / 2))

        X[it] = (X1 + (2 * X2) + X3 - X4) / 3
        Y[it] = (Y1 + (2 * Y2) + Y3 - Y4) / 3
        Z[it] = (Z1 + (2 * Z2) + Z3 - Z4) / 3

    return X, Y, Z


for r in r_list:
    # ====== Time integration ====================================================

    X0_a, Y0_a, Z0_a = (0.0, 1.0, 0.0)
    X0_b, Y0_b, Z0_b = (0.0 + 0.001, 1.0 + 0.001, 0.0 + 0.001)
    X0_c, Y0_c, Z0_c = (0.0 - 0.001, 1.0 - 0.001, 0.0 - 0.001)

    # Varying initial conditions (as given in the exercise as (X, Y, Z) = (0, 1, 0)
    X_a, Y_a, Z_a = rk4_lorenz(X0_a, Y0_a, Z0_a, p, beta, r, dt, nt)
    X_b, Y_b, Z_b = rk4_lorenz(X0_b, Y0_b, Z0_b, p, beta, r, dt, nt)
    X_c, Y_c, Z_c = rk4_lorenz(X0_c, Y0_c, Z0_c, p, beta, r, dt, nt)

    # ====== Plotting results ====================================================
    plt.figure(figsize=(7, 7))

    spec = gridspec.GridSpec(ncols=1, nrows=3)

    ax1 = plt.subplot(spec[0])
    ax2 = plt.subplot(spec[1])
    ax3 = plt.subplot(spec[2])

    ax1.set_title(f'r = {r}')
    ax1.plot(t, X_a, 'k', label=f'X0, Y0, Z0 = {X0_a, Y0_a, Z0_a}', linewidth=0.7)
    ax1.plot(t, X_b, 'r', label=f'X0, Y0, Z0 = {X0_b, Y0_b, Z0_b}', linewidth=0.7)
    ax1.plot(t, X_c, 'g', label=f'X0, Y0, Z0 = {X0_c, Y0_c, Z0_c}', linewidth=0.7)
    ax1.set_ylabel('X')
    ax1.set_xlim(0, 50)

    ax2.plot(t, Y_a, 'k', label=f'X0, Y0, Z0 = {X0_a, Y0_a, Z0_a}', linewidth=0.7)
    ax2.plot(t, Y_b, 'r', label=f'X0, Y0, Z0 = {X0_b, Y0_b, Z0_b}', linewidth=0.7)
    ax2.plot(t, Y_c, 'g', label=f'X0, Y0, Z0 = {X0_c, Y0_c, Z0_c}', linewidth=0.7)
    ax2.set_ylabel('Y')
    ax2.set_xlim(0, 50)

    ax3.plot(t, Z_a, 'k', label=f'X0, Y0, Z0 = {X0_a, Y0_a, Z0_a}', linewidth=0.7)
    ax3.plot(t, Z_b, 'r', label=f'X0, Y0, Z0 = {X0_b, Y0_b, Z0_b}', linewidth=0.7)
    ax3.plot(t, Z_c, 'g', label=f'X0, Y0, Z0 = {X0_c, Y0_c, Z0_c}', linewidth=0.7)
    ax3.set_ylabel('Z')
    ax3.set_xlabel('Time')
    ax3.set_xlim(0, 50)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.23))

    plt.savefig(current_filepath + 'init_conditions_r_' + str(r).replace('.', '_') + '.png', bbox_inches='tight',
                dpi=300)
