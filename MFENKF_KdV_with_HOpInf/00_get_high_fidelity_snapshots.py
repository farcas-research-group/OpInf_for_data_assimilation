from utils.utils import *
from config.config import *

if __name__ == '__main__':
    
    # VARIABLES
    ic = KdV_soliton(Xg, nsol=nsol) #Two solition solutions
    print(ic.shape)

    # Build KdV Matrices and Calculate FOM snapshots
    A, B = build_KdV_mats(n, x_ends) #KdV matrices
    print("KdV matrices calculated")
        
    print(A.shape)
    print(B.shape)

    print(t_train.shape)
        
    X, Xt, gH = integrate_KdV_FOM(t_pred, ic, A, B, **KdV_params)
    print("Snapshots generated")

    np.save(X_full, X)
    np.save(Xt_full, Xt)
    np.save(gH_full, gH)