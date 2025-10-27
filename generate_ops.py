# Helper file to generate snapshots, basis, and operators
# IMPORTS
# Libraries
import os.path

# Modules
from config.config import *
from utils.utils import *

# VARIABLES
ic = KdV_soliton(xTrain, nsol=2) #Two solition solutions

# FUNCTIONS
def main():
    # Build KdV Matrices and Calculate FOM snapshots
    A, B = build_KdV_mats(N, xEnds) #KdV matrices
    print("KdV matrices calculated")
    
    # Data matrix, finite difference matrix, Hamiltonian gradient snapshots
    # Overwrite existing operators
    X, Xt, gH = integrate_KdV_FOM(tTrain, ic, A, B, **params)
    np.save(X_path, X)
    np.save(Xt_path, Xt)
    np.save(gH_path, gH)
    print("Snapshots generated")

    # Build POD basis with r columns and pre-compute needed ROM ops
    UU, SS = np.linalg.svd(X)[:2]
    np.save(UU_path, UU)
    np.save(SS_path, SS)
    print("SVD calculated")

    # Calculate pre computable operators in NC procedure
    H_ROM_OPS = build_H_KdV_ROM_Ops(UU, B, ic, r=r, **params, MC=False)
    np.save(c_path, H_ROM_OPS.get("cHat"))
    np.save(CHat_path, H_ROM_OPS.get("CHat"))
    np.save(THat_path, H_ROM_OPS.get("THat"))
    print("Hamiltonian ROM ops calculated")

    # Equation setup for inferring L hat
    OpInf_eq = build_OpInf_eq(UU, Xt, gH, r=r)

    # Infer L hat operator
    LHat = NC_H_OpInf(OpInf_eq, r=r, eps=0.0) #implement regularization later
    np.save(LHat_path, LHat)
    print("L Hat operator calculated")

if __name__ == '__main__':
    main()