# IMPORTS
import os
import sys

from config.config import *
from utils.utils import *

# VARIABLES
ic = KdV_soliton(xTrain, nsol=2) #Two solition solutions

def main():
    if (sys.argv[1]):
        r = sys.argv[1]
    # Build KdV Matrices and Calculate FOM snapshots
    A, B = build_KdV_mats(N, xEnds) #KdV matrices
    print("KdV matrices calculated")

    # Load reduced operators and basis
    H_ROM_OPS ={
        "cHat" : np.load(c_path),
        "CHat" : np.load(CHat_path),
        "THat" : np.load(THat_path),
        "LHat" : np.load(LHat_path)
    }

    UU = np.load(UU_path)

    print("Operators and Basis Loaded")
    # Integrate FOM for test case
    Xtest = integrate_KdV_FOM(tSpace, ic, A, B, **params)[0]
    print("FOM integrated, T =", Ttest)

    # Integrate ROM for test case
    OpList = list(H_ROM_OPS.values())
    HOPINF_SOL = integrate_KdV_ROM(tSpace, OpList, ic, UU, r)
    print("ROM integrated")

    # Animation
    animate_multiple_trajectories(xTrain, Xtest, HOPINF_SOL, dt, "Hamiltonian Forecast", "H_OpInf_Forecast.gif")

if __name__=='__main__':
    main()