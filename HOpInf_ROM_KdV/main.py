# Main handler for Generating Operators
# IMPORTS
# Libraries
import os.path
import sys

# Modules
from config.config import *
from utils.utils import *

# VARIABLES
ic = KdV_soliton(xTrain, nsol=2) #Two solition solutions

# FUNCTIONS
def main():
    if (sys.argv[1]):
        r = sys.argv[1]
    # Build KdV Matrices and Calculate FOM snapshots
    A, B = build_KdV_mats(N, xEnds) #KdV matrices
    print("KdV matrices calculated")
    
    # Pre-load if available
    Xexist = os.path.isfile(X_path)
    Xtexist = os.path.isfile(Xt_path)
    gHexist = os.path.isfile(gH_path)

    if Xexist and Xtexist and gHexist:
        X = np.load(X_path)
        Xt = np.load(Xt_path)
        gH = np.load(gH_path)
    
    # Data matrix, finite difference matrix, Hamiltonian gradient snapshots
    else:     
        X, Xt, gH = integrate_KdV_FOM(tTrain, ic, A, B, **params)
        np.save(X_path, X)
        np.save(Xt_path, Xt)
        np.save(gH_path, gH)
    print("Snapshots generated")

    # Build POD basis with r columns and pre-compute needed ROM ops
    # Pre-load if available
    UUexist = os.path.isfile(UU_path)
    SSexist = os.path.isfile(SS_path)

    if UUexist and SSexist:
        UU = np.load(UU_path)
        SS = np.load(SS_path)
    else:
        UU, SS = np.linalg.svd(X)[:2]
        np.save(UU_path, UU)
        np.save(SS_path, SS)
    print("SVD calculated")

    # Pre-load if available
    cExist = os.path.isfile(c_path)
    CMatExist = os.path.isfile(CHat_path)
    ThatExist = os.path.isfile(THat_path)

    if cExist and CMatExist and ThatExist:
        H_ROM_OPS = {
            "cHat" : np.load(c_path),
            "CHat" : np.load(CHat_path),
            "THat" : np.load(THat_path),
        }
    
    else:
        H_ROM_OPS = build_H_KdV_ROM_Ops(UU, B, ic, r=r, **params, MC=False)
        np.save(c_path, H_ROM_OPS.get("cHat"))
        np.save(CHat_path, H_ROM_OPS.get("CHat"))
        np.save(THat_path, H_ROM_OPS.get("THat"))
    print("Hamiltonian ROM ops calculated")

    OpInf_eq = build_OpInf_eq(UU, Xt, gH, r=r)
    # Infer L hat operator
    H_ROM_OPS["LHat"] = NC_H_OpInf(OpInf_eq, r=r, eps=0.0) # Generate Lhat each run
    np.save(LHat_path, H_ROM_OPS["LHat"])
    print("L Hat operator calculated")

    OpList = list(H_ROM_OPS.values())

    HOPINF_SOL = integrate_KdV_ROM(tTrain, OpList, ic, UU, r)
    print("ROM projected onto FOM space")

    animate_multiple_trajectories(xTrain, X, HOPINF_SOL, dt, "Hamiltonian Solution", "hamiltonian_fom_sol.gif")

if __name__ == '__main__':
    main()