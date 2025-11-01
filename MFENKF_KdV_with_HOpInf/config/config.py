import numpy as np
from itertools import product
from time import time

from scipy.integrate import solve_ivp
from scipy.linalg import svd, sqrtm
from scipy.linalg import sqrtm


# might want to fix rng for numpy for reproducibility
np.random.seed(16559)

# KdV equation setup
nu      = -1.0
alp     = -3.0
rho     = 0.0

KdV_params = { 
    "a" : 2*alp,
    "p" : rho,
    "v" : nu
}

nsol =  2

n       = 1000  # this is the number of discrete points
Xg      = np.linspace(-10, 10, n)
dx      = Xg[1] - Xg[0]
x_ends  = [-10, 10] #Useful for building KdV matrices

N       = 25
N_ROM   = 100

dt          = 1e-3  # step size
t_train_end = 2.0
t_pred_end  = 5.0
t_train     = np.arange(0, t_train_end + dt, dt) #Time span
t_pred      = np.arange(0, t_pred_end + dt, dt) #Time span

nt      = t_train.shape[0]
nt_all  = t_pred.shape[0]
#########################


# ROM setup
target_energy = 0.9999

r           = 50
r_all       = [40, 45, 50, 55, 60, 65, 70]
#########################


# MFEnKF setup
# define the observation operators and observation error covariance
# we note that there is no model error here
observe_index   = list(range(0, n, 20))
num_observation = len(observe_index)

eta     = 1
R       = np.square(eta) * np.eye(num_observation, num_observation)  # this is for the primary and control variate
sqrtR   = sqrtm(R)

R3 = 3 * R  # noise for the ancillary/rom variate

H = np.eye(n)            # Full identity matrix of size n×n
H = H[observe_index, :]

infl        = 1.01  # inflation factor for primary and control variate
infl_ROM    = 1.05  # inflation factor for ancillary variate

# create ensembles for FOM and ROM
rand_vals       = 1.5 * np.random.randn(N) # shape (N,)
Xg_col          = Xg[:, np.newaxis]           # shape (n, 1)
rand_vals_row   = rand_vals[np.newaxis, :]  # shape (1, N)
xf              = 6 * (1 / np.cosh(Xg_col - rand_vals_row))**2  # shape (n, N)

rand_vals       = 1.5 * np.random.randn(N_ROM)                 # shape: (N_ROM,)
Xg_col          = Xg[:, np.newaxis]                               # shape: (n, 1)
rand_vals_row   = rand_vals[np.newaxis, :]                 # shape: (1, N_ROM)
profiles        = 6 * (1 / np.cosh(Xg_col - rand_vals_row))**2  # shape: (n, N_ROM)
##################


# Paths
X_full  = './data/X_full.npy'
Xt_full = './data/Xt_full.npy'
gH_full = './data/gH_full.npy'

HOpInf_SVD = './data/HOpInf_SVD.npy'

HOpInf_Ops = lambda r: './data/HOpInf_Ops_' + str(r) + '.npy'

Galerkin_ROM_sol_file 	= './data/Galerkin_ROM_sol.npy'
HOpInf_ROM_sol_file 	= './data/HOpInf_ROM_sol.npy'
##################