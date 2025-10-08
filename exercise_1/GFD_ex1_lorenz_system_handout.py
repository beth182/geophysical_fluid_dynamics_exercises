import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# GFD exercise 1: Lorenz System
# Beth Saunders 03/10/2025
# ==============================================================================

# Specify path for plots
current_filepath = os.getcwd().replace('\\', '/') + '/'
assert os.path.exists(current_filepath)

# Initial conditions
X0 = 0.0
Y0 = 1.0
Z0 = 0.0

# Define parameters
p = 10
beta = 8 / 3
# r = 0.5
# r = 22
r = 28
dt = 0.01

# Time vector
t = np.arange(0, 100 + dt, dt)
nt = len(t)


# ==============================================================================
# Runge-Kutta integration
# ==============================================================================
def rk4_lorenz(X0, Y0, Z0, p, beta, r, dt, nt):
    X = np.zeros(nt)
    Y = np.zeros(nt)
    Z = np.zeros(nt)
    X[0] = X0
    Y[0] = Y0
    Z[0] = Z0

    for it in range(1, nt):
        # TASK 1: add the equations for dYdt, dZdt, Y1, and Z1
        dXdt = p * (Y[it - 1] - X[it - 1])

        # dYdt =
        # dZdt =

        X1 = X[it - 1] + (dXdt * (dt / 2))

        # Y1 =
        # Z1 =

        # TASK 2: add the equations for dYdt, dZdt, Y2, and Z2

        dXdt = p * (Y1 - X1)

        # dYdt =
        # dZdt =

        X2 = X[it - 1] + (dXdt * (dt / 2))

        # Y2 =
        # Z2 =

        # TASK 3: Calculate steps 3 & 4

        # dXdt =
        # dYdt =
        # dZdt =
        # X3 =
        # Y3 =
        # Z3 =

        # dXdt =
        # dYdt =
        # dZdt =
        # X4 =
        # Y4 =
        # Z4 =

        # TASK 4: Add the equations for X(it), Y(it), and Z(it)

        # X[it] =
        # Y[it] =
        # Z[it] =

    return X, Y, Z


# ====== Time integration ====================================================
X, Y, Z = rk4_lorenz(X0, Y0, Z0, p, beta, r, dt, nt)

# ====== Plotting results ====================================================
plt.figure(figsize=(7, 3))
plt.title(f'r = {r}')
plt.plot(t, X, 'k', label='X', linewidth=1.2)
plt.plot(t, Y, 'g--', label='Y', linewidth=1.2)
plt.plot(t, Z, 'r', label='Z', linewidth=1.2)
plt.xlabel('Time')
plt.legend(loc='upper right', fontsize=12)
plt.grid()
plt.savefig(current_filepath + 'r_' + str(r).replace('.', '_') + '.png', dpi=300)
# plt.show()

if abs(r - 28.0) < 1e-5:
    # Butterfly plot
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X, Y, Z, 'k', linewidth=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(current_filepath + 'butterfly.png', dpi=300)
