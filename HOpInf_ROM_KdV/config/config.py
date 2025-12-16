# Configuration and some global variables used for KdV eqs
# IMPORTS 
import os
import numpy as np

# VARIABLES
# OS (lots of paths)
# Data directory
dataBase = 'data'

# Data subdirs
snapshots = 'snapshots'
Basis = 'Basis'
H_OPS = 'H_OPS'
L_OP = 'L_OP'

# Filenames
X_f  = 'X.npy'
Xt_f = 'Xt.npy'
gH_f = 'gH.npy'
Xtest_f = 'Xtest.npy'

UU_f = 'UU.npy'
SS_f = 'SS.npy'

c_f  = 'c.npy'
CHat_f = 'CHat.npy'
THat_f = 'THat.npy'

LHat_f = 'LHat.npy'

X_path = os.path.join(dataBase, snapshots, X_f)
Xt_path = os.path.join(dataBase, snapshots, Xt_f)
gH_path = os.path.join(dataBase, snapshots, gH_f)
Xtest_path = os.path.join(dataBase, snapshots, Xtest_f)

UU_path = os.path.join(dataBase, Basis, UU_f)
SS_path = os.path.join(dataBase, Basis, SS_f)

c_path = os.path.join(dataBase, H_OPS, c_f)
CHat_path = os.path.join(dataBase, H_OPS, CHat_f)
THat_path = os.path.join(dataBase, H_OPS, THat_f)

LHat_path = os.path.join(dataBase, L_OP, LHat_f)

# specific paths
# KdV equation parameters
params = { 
    "a" : -6,
    "p" : 0,
    "v" : -1
}

# Training
# Spatial params
N       = 1000 #Snapshots and # of discrete points
xTrain  = np.linspace(-10, 10, N) #Area of interest x = [-10, 10]
xEnds   = [-10, 10] #Useful for building KdV matrices
dx      = xTrain[1] - xTrain[0]

# Time params
# Train
Nt      = 2000 #Number of time steps (dt = 10^-3)
T       = 2 #Total training time
tTrain  = np.linspace(0, T, Nt) #Time span
dt      = tTrain[1] - tTrain[0]

# Test
NtTest  = 10000
Ttest   = 10
tSpace  = np.linspace(0, Ttest, NtTest)

# ROM information
r = 92 #Dimension of reduced model (Test across [50, 70, 90])
MC = False #Mean centering (can implement later, leaving as note)