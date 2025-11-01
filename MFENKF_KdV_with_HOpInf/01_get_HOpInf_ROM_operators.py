from utils.utils import *
from config.config import *
        
if __name__ == '__main__':

    ic = KdV_soliton(Xg, nsol=nsol) #Two solition solutions
    print(ic.shape)

    # Build KdV Matrices and Calculate FOM snapshots
    A, B = build_KdV_mats(n, x_ends) #KdV matrices
    print("KdV matrices calculated")

    X_full  = np.load(X_full)
    Xt_full = np.load(Xt_full)
    gH_full = np.load(gH_full)

    X_train     = X_full[:, :nt]
    Xt_train    = Xt_full[:, :nt]
    gH_train    = gH_full[:, :nt]

    U, S, _ = np.linalg.svd(X_train)
    print("SVD calculated")

    np.save(HOpInf_SVD, U)

    H_ROM_OPS = build_H_KdV_ROM_Ops(U, B, ic, r=r, **KdV_params, MC=False)
       
    OpInf_eq = build_OpInf_eq(U, Xt_train, gH_train, r=r)

    H_ROM_OPS["LHat"] = NC_H_OpInf(OpInf_eq, r=r, eps=1e-15) # Generate Lhat each run
    
    OpList = list(H_ROM_OPS.values())

    np.save(HOpInf_Ops(r), H_ROM_OPS)

    xHat0 = U[:, :r].T @ ic.flatten()

    HOpInf_sol = integrate_KdV_ROM(t_pred, OpList, xHat0, r)

    HOpInf_sol_full = U[:, :r] @ HOpInf_sol 

    np.save(HOpInf_ROM_sol_file, HOpInf_sol_full) 

    for r in r_all:
    
        H_ROM_OPS = build_H_KdV_ROM_Ops(U, B, ic, r=r, **KdV_params, MC=False)
       
        OpInf_eq = build_OpInf_eq(U, Xt_train, gH_train, r=r)

        H_ROM_OPS["LHat"] = NC_H_OpInf(OpInf_eq, r=r, eps=1e-15) # Generate Lhat each run
        
        OpList = list(H_ROM_OPS.values())

        np.save(HOpInf_Ops(r), H_ROM_OPS)

        xHat0 = U[:, :r].T @ ic.flatten()

        HOpInf_sol = integrate_KdV_ROM(t_pred, OpList, xHat0, r)

        HOpInf_sol_full = U[:, :r] @ HOpInf_sol  
        
        print('results for r = ', r)
        print(np.linalg.norm(HOpInf_sol_full[:, :nt] - X_full[:, :nt])/np.linalg.norm(X_full[:, :nt]))
        print(np.linalg.norm(HOpInf_sol_full[:, nt:nt_all] - X_full[:, nt:nt_all])/np.linalg.norm(X_full[:, nt:nt_all]))

