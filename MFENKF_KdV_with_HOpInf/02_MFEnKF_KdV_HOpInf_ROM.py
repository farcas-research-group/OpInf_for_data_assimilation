from utils.MFEnKF import *
from utils.utils import *
from config.config import *

# Load full POD basis and H-OpInf reduced operators
Phi_full            = np.load(HOpInf_SVD)
HOpInf_operators    = np.load(HOpInf_Ops(r), allow_pickle=True).item()

Phi     = Phi_full[:, :r]
OpList  = list(HOpInf_operators.values())

# this is to get one time iteration with the H-OpInf ROM
HOpInf_red_model  = lambda x: integrate_KdV_ROM_one_step(OpList=OpList, r=r, dt=dt, xHat0=x)

Dxf, Dx, Lx, D3x = diff(n, dx)  # extracting the differential operators

# Initial Conditions
ufom0 = nsol*(nsol+1) * (1 / np.cosh(Xg))**2
urom0 = Phi.T @ ufom0

ufom = ufom0
urom = urom0

ref_traj            = np.zeros((n, nt_all + 1)) # for plotting
xa_HOpInf_traj      = np.zeros((n, nt_all + 1)) # for plotting

xf_ROM                          = Phi.T @ profiles                                # shape: (r, N_ROM)
ref_traj[:, 0]                  = ufom
xa_HOpInf_traj[:, 0]            = np.mean(xf, axis=1)

xa                  = xf  # not needed, just to avoid confusion
xa_OpInf_ROM        = xf_ROM  # not needed, just to avoid confusion
xt                  = ufom

rmse = 0
rmse_plot = np.zeros(nt_all)
for i in range(nt_all):
    # propagate the truth
    sol = solve_ivp(rhs, [0, dt], xt, method='RK23', args=(nu, alp, rho, Dx, D3x))
    xt  = sol.y[:, -1]
    
    ref_traj[:, i+1] = xt

    # Project the FOM analysis to ROM space (control variate)
    xa_FOM_in_ROM = Phi.T @ xa  # Shape: (r, N)
    # propagate it through the rom dynamics
    for j in range(N):
        xa_FOM_in_ROM[:, j] = HOpInf_red_model(xa_FOM_in_ROM[:, j])

    # propagate the FOM ensembles through the FOM dynamics (primary variate)
    for j in range(N):
        sol         = solve_ivp(rhs, [0, dt], xa[:, j], method='RK23', args=(nu, alp, rho, Dx, D3x))
        xa[:, j]    = sol.y[:, -1]

    # propagate the ROM through the ROM dynamics (ancillary variate)
    for j in range(N_ROM):
        xa_OpInf_ROM[:, j]  = HOpInf_red_model(xa_OpInf_ROM[:, j])

    # create observations based around the truth (linear H for now)
    y = H@xt + sqrtR @ np.random.randn(num_observation)

    # inflating the prior covariances is done inside the MFEnKF function
    # call the MFEnKF algorithm for the filtering part
    xa, xa_OpInf_ROM        = mfenkf(xa, xa_OpInf_ROM, xa_FOM_in_ROM, Phi, y, H, R, R3, infl, infl_ROM)
    xa_mean                 = np.mean(xa, axis=1)
    xa_HOpInf_traj[:, i+1]  = xa_mean

    # you can apply spinoffs, if needed. basically we ignore the first few rmses
    rmse = np.sqrt(((np.linalg.norm(xa_mean-xt, 2))**2 + (rmse**2) * i * n)/((i+1)*n))
    # rmse = np.linalg.norm(xa_mean-xt, 2)
    rmse_plot[i] = rmse  # use this to plot, if needed

    print(f"Step = {i}, rmse = {rmse:.5f}")

np.save('data/xa_HOpInf_traj.npy', xa_HOpInf_traj)
np.save('data/rmse_HOpInf_traj.npy', rmse_plot)