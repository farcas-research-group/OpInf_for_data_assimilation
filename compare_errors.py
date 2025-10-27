# IMPORTS
import os

# Modules
from config.config import *
from utils.utils import *

# VARIABLES
ic = KdV_soliton(xTrain, nsol=2)

# FUNCTIONS
def main():
    rList = [i*4 for i in range(1, 24)] #Stop at r=92

    # Build KdV Matrices and Calculate FOM snapshots
    A, B = build_KdV_mats(N, xEnds) #KdV matrices
    print("KdV matrices calculated")

    # Integrate FOM for test case
    if os.path.isfile(Xtest_path):
        Xtest = np.load(Xtest_path)
    else:
        Xtest = integrate_KdV_FOM(tSpace, ic, A, B, **params)[0]
        np.save(Xtest_path, Xtest)
    print("FOM integrated, T =", Ttest)

    # Load training snapshots
    Xt = np.load(Xt_path)
    gH = np.load(gH_path)

    # Load basis
    UU = np.load(UU_path)

    error = np.zeros(len(rList))
    for i, rd in enumerate(rList):
        print("r =", rd)
        # Build rom ops
        H_ROM_OPS = build_H_KdV_ROM_Ops(UU, B, ic, r=rd, **params, MC=False)

        # Set up OpInf equation
        OpInf_eq = build_OpInf_eq(UU, Xt, gH, rd)

        # Infer LHat
        H_ROM_OPS["LHat"] = NC_H_OpInf(OpInf_eq, r=rd, eps=0.0)

        # Integrate ROM for test case
        OpList = list(H_ROM_OPS.values())
        HOPINF_SOL = integrate_KdV_ROM(tSpace, OpList, ic, UU, rd)

        error[i] = relError(Xtest, HOPINF_SOL)

    print("ROM integrated")

    print("Relative errors:", error)

    plt.semilogy(rList, error, label='NC-H-OpInf (no MC)',
             marker='v', linestyle='--', linewidth=0.5, markersize=5)
    plt.ylabel('relative $L^2$ error')
    plt.xlabel('basis size $r$')
    plt.title('KdV ROM Errors (Predictive)')
    plt.ylim([10**-5,10])
    plt.legend(loc=3)

    plt.tight_layout()
    plt.savefig(f'KdVPlotT{T}', transparent=True)
    plt.show()

if __name__ == '__main__':
    main()