import numpy as np
from itertools import product
from time import time

# set r = - 1 if you want to the algorithm to use it based on the threshold for retained energy
r       		= 26
target_energy	= 0.999

# problem setup
n   	= 63*127  # this is the number of discrete points
nt 		= 400
nt_all 	= 500


# setup for regularization
B1 = np.logspace(-2., 5., num=20)
B2 = np.logspace(-2., 8., num=20)

max_growth = 1.2

DATA_NORMALIZATION 	= False
VERBOSE 			= False

data_file 			= './data/trajectory_for_ROM_500.mat'
OpInf_operator_file = './data/OpInf_ROM_operators.npz'
POD_basis_file 		= './data/OpInf_POD_basis.npy'