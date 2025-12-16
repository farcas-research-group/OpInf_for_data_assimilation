import numpy as np
import scipy.sparse as sp
import scipy.special as special

from numpy.linalg import norm, solve

from functools import partial

from scipy.optimize import root
from scipy.sparse import csc_matrix, identity, diags,spdiags, lil_matrix
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def compute_Qhat_sq(Qhat):
	"""
	compute_Qhat_sq returns the non-redundant terms in Qhat squared

	:Qhat: reduced data

	:return: Qhat_sq containing the non-redundant in Qhat squared
	"""

	if len(np.shape(Qhat)) == 1:

	    r 		= np.size(Qhat)
	    prods 	= []
	    for i in range(r):
	        temp = Qhat[i]*Qhat[i:]
	        prods.append(temp)

	    Qhat_sq = np.concatenate(tuple(prods))

	elif len(np.shape(Qhat)) == 2:
	    K, r 	= np.shape(Qhat)    
	    prods 	= []
	    
	    for i in range(r):
	        temp = np.transpose(np.broadcast_to(Qhat[:, i], (r - i, K)))*Qhat[:, i:]
	        prods.append(temp)
	    
	    Qhat_sq = np.concatenate(tuple(prods), axis=1)

	else:
	    print('invalid input!')

	return Qhat_sq

def compute_train_err(Qhat_train, Qtilde_train):
	"""
	compute_train_err computes the OpInf training error

	:Qhat_train: 	Qhat_trainerence data
	:Qtilde_train: 	Qtilde_train data

	:return: train_err containing the value of the training error
	"""
	train_err = np.max(np.sqrt(np.sum( (Qtilde_train - Qhat_train)**2, axis=1) / np.sum(Qhat_train**2, axis=1)))

	return train_err

def solve_opinf_difference_model(qhat0, n_steps_pred, dOpInf_red_model):
	"""
	solve_opinf_difference_model solves the discrete OpInf ROM for n_steps_pred over the target time horizon (training + prediction)

	:qhat0: 			reduced initial condition Qtilde0=np.matmul (Vr.T, q[:, 0]
	:n_steps_pred: 		number of steps over the target time horizon to solve the OpInf reduced model
	:dOpInf_red_model: 	dOpInf ROM

	:return: contains_nan flag indicating NaN presence in in the Qtilde_train reduced solution, Qtilde
	"""

	Qtilde    		= np.zeros((np.size(qhat0), n_steps_pred))
	contains_nans  	= False

	Qtilde[:, 0] = qhat0
	for i in range(n_steps_pred - 1):
	    Qtilde[:, i + 1] = dOpInf_red_model(Qtilde[:, i])

	if np.any(np.isnan(Qtilde)):
	    contains_nans = True

	return contains_nans, Qtilde.T

def compute_Qhat_cube(Qhat):

	if len(np.shape(Qhat)) == 1:

		state = Qhat

		state2 = compute_Qhat_sq(state)

		lens = special.binom(np.arange(2, len(state) + 2), 2).astype(int)
		
		Qhat_cube = np.concatenate(
			[state[i] * state2[: lens[i]] for i in range(state.shape[0])],
			axis=0,
			)

	elif len(np.shape(Qhat)) == 2:

		nt, r = Qhat.shape

		r_cube = r*(r + 1)*(r + 2) // 6

		Qhat_cube = np.zeros((nt, r_cube))

		for i in range(nt):
			state = Qhat[i, :]

			state2 = compute_Qhat_sq(state)

			lens = special.binom(np.arange(2, len(state) + 2), 2).astype(int)

			Qhat_cube[i, :] = np.concatenate(
				[state[i] * state2[: lens[i]] for i in range(state.shape[0])],
				axis=0,
				)

	return Qhat_cube

def diff(n, dx):
    """
    Computes the finite difference operators

    Parameters
    ----------
    n : int
        size of the matrix
    dx : float
        discretization in space

    Returns
    ----------
    Dxf : sparse matrix
        Forward difference operator.
    Dx : sparse matrix
        Centered first derivative
    Lx : sparse matrix
        Second derivative (Laplacian)
    D3x : sparse matrix
        Third derivative
    """
    # Forward difference operator Dxf
    val = 1 / dx
    e1 = val * np.ones(n)
    Dxf = spdiags([-e1, e1], [0, 1], n, n, format='lil')
    Dxf[-1, 0] = val

    # Centered first derivative Dx
    val = 0.5 / dx
    e1 = val * np.ones(n)
    Dx = spdiags([-e1, e1], [-1, 1], n, n, format='lil')
    Dx[0, -1] = -val
    Dx[-1, 0] = val

    # Second derivative (Laplacian) Lx
    val = 1 / dx**2
    e1 = val * np.ones(n)
    Lx = spdiags([e1, -2*e1, e1], [-1, 0, 1], n, n, format='lil')
    Lx[0, -1] = val
    Lx[-1, 0] = val

    # Third derivative D3x
    val2 = 1 / dx**3
    e1 = val2 * np.ones(n)
    D3x = spdiags(
        [-0.5 * e1, e1, -e1, 0.5 * e1],
        [-2, -1, 1, 2],
        n, n, format='lil'
    )
    # Periodic BCs
    D3x[0, -2] = -0.5 * val2
    D3x[0, -1] = val2
    D3x[-1, 0] = -val2
    D3x[-1, 1] = 0.5 * val2
    D3x[1, -1] = -0.5 * val2
    D3x[-2, 0] = 0.5 * val2

    return Dxf.tocsr(), Dx.tocsr(), Lx.tocsr(), D3x.tocsr()


def rhs(t, y, nu, alp, rho, Dx, D3x):
    """
    Computes the right-hand side of the KdV-type PDE.

    Parameters
    ----------
    y : ndarray
        State vector (1D array).
    nu : float
        Dispersion coefficient.
    alp : float
        Nonlinear coefficient.
    rho : float
        Advection coefficient.
    Dx : sparse matrix
        First derivative matrix.
    D3x : sparse matrix
        Third derivative matrix.

    Returns
    -------
    dy : ndarray
        Time derivative dy/dt (rhs)
    """
    y_x = Dx @ y
    dy = alp * (Dx @ (y**2)) + rho * y_x + nu * (D3x @ y)
    return dy


def get_rom_operators(Phi, D1, D3, r):
    """
    Computes reduced-order operators for ROM of KdV-type PDE with sparse differential operators.

    Parameters
    ----------
    Phi : ndarray
        Basis matrix (n x r), dense.
    D1 : scipy.sparse matrix
        First-derivative operator (n x n), sparse.
    D3 : scipy.sparse matrix
        Third-derivative operator (n x n), sparse.
    r : int
        Number of modes in the reduced basis.

    Returns
    -------
    rl3 : ndarray
        ROM linear operator for third derivative (r x r).
    rl1 : ndarray
        ROM linear operator for first derivative (r x r).
    rq : ndarray
        ROM nonlinear operator (r x r x r), permuted to match MATLAB output.
    """

    # Linear ROM operators
    rl3 = Phi.T @ (D3 @ Phi) if sp.issparse(D3) else Phi.T @ D3 @ Phi
    rl1 = Phi.T @ (D1 @ Phi) if sp.issparse(D1) else Phi.T @ D1 @ Phi

    # Nonlinear ROM operator
    rq = np.zeros((r, r, r))
    D1Phi = D1.dot(Phi) if sp.issparse(D1) else D1 @ Phi  # (n x r)
    # D1Phi = D1 @ Phi

    # TODO: try einsum
    for i in range(r):
        for j in range(r):
            phi_j = Phi[:, j]
            for k in range(r):
                phi_jDphi_k = phi_j * D1Phi[:, k]
                rq[i, j, k] = 2 * (Phi[:, i].T @ phi_jDphi_k)

    rq = np.transpose(rq, (1, 2, 0))

    return rl3, rl1, rq

def rhsrom(t, y, alp, nu, rho, L3, L1, Q):
    """
    Computes the reduced-order model RHS for the KdV equation.

    Parameters
    ----------
    y : ndarray
        Reduced state vector (r,)
    alp : float
        Coefficient for the quadratic term
    nu : float
        Coefficient for third derivative term
    rho : float
        Coefficient for first derivative term
    L3 : ndarray
        ROM linear operator for third derivative (r x r)
    L1 : ndarray
        ROM linear operator for first derivative (r x r)
    Q : ndarray
        ROM nonlinear operator (r x r x r), Q[i,j,k] = contribution to i-th mode

    Returns
    -------
    dyrom : ndarray
        Time derivative of reduced state (r,)
    """
    r       = y.shape[0]
    yyt     = np.outer(y, y)  # (r x r)
    quad    = np.array([np.sum(Q[:, :, i] * yyt) for i in range(r)])

    # Linear + nonlinear combination
    dyrom = (nu * L3 + rho * L1) @ y + alp * quad
    
    return dyrom

# H-OpInf FUNCTIONS
# Soliton solution with nsol number of solitons
def KdV_soliton(xTrain, nsol=2):
    return nsol * (nsol + 1) * (1 / np.cosh(xTrain))**2

# Build tridiagonal matrix
def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

# Builds required matrices for KdV formulation
def build_KdV_mats(N, xEnds):
    x  = np.linspace(xEnds[0], xEnds[1], N)
    dx = x[1] - x[0]

    # Build derivative matrix A (this is v1 L)
    A       = tridiag(-np.ones(N-1), 
                      np.zeros(N), np.ones(N-1))
    A[-1,0] =  1
    A[0,-1] = -1
    A      *= 1 / (2*dx)
    A       = csc_matrix(A)

    # Build Laplace mtx B
    B       = tridiag(np.ones(N-1),
                      -2*np.ones(N), np.ones(N-1))
    B[-1,0] = 1
    B[0,-1] = 1
    B      *= (1/dx)**2
    B       = csc_matrix(B)

    return A, B

# Finite differences: 4th order in middle, 1st order at ends
def FDapprox(y, step):
    diffs       = np.zeros_like(y)
    diffs[0]    = (y[1] - y[0]) / step
    diffs[1]    = (y[2] - y[1]) / step
    diffs[2:-2] = (-y[4:] + 8*y[3:-1] - 
                   8*y[1:-3] + y[:-4]) / (12 * step)
    diffs[-2]   = (y[-2] - y[-3]) / step
    diffs[-1]   = (y[-1] - y[-2]) / step
    return diffs

# Newton solve given res, jac functions
# Gets super slow if you crank up the iterations or reduce the tolerance.
def do_newton(res, jac, xOld, tol=1e-6, maxIt=10, verbose=False, sparse=True):
    if not sparse:
        solver = solve
    else:
        solver = spsolve
    xNew = xOld
    err  = norm(res(xNew))
    iter = 0
    while err > tol:
        xNew  = xNew - solver(jac(xNew), res(xNew))
        iter += 1
        if iter > maxIt:
            err  = norm(res(xNew))
            if verbose: print('err =',err)
            break
    return xNew

# Function to collect snapshots of KdV FOM
# TIme discretization is AVF
def integrate_KdV_FOM(tRange, ic, A, B, a=-6, p=0, v=-1):
    N  = ic.shape[0]
    Nt = tRange.shape[0]
    dt = tRange[1] - tRange[0]
    
    # Build gradH for KdV, depends on central diff mtx B
    def gradH(x, B, a=-6, p=0, v=-1):
        return 0.5*a*x**2 + p*x + v*B@x

    # For root-finding alg
    term2mat = p*identity(N) + v*B
    
    def Nfunc(xOld, xNew):
        xMid  = 0.5 * (xOld + xNew)
        term1 = a/6 * (xOld**2 + xOld*xNew + xNew**2)
        rhs   = A @ (term1 + term2mat @ xMid)
        return xNew - (xOld + dt * rhs)

    def Nderiv(xOld, xNew):
        N     = xNew.shape[0]
        term1 = a/6 * A @ (diags(xOld) + 2*diags(xNew))
        term2 = 1/2 * A @ term2mat
        return identity(N) - dt * (term1 + term2)     

    # Generating snapshots
    Xdata      = np.zeros((N, Nt))
    Xdata[:,0] = ic.flatten()
    
    for i,t in enumerate(tRange[:-1]):
        res  = partial(Nfunc, Xdata[:,i])
        jac  = partial(Nderiv, Xdata[:,i])    
        Xdata[:,i+1] = do_newton(res, jac, Xdata[:,i])

    # Snapshots of gradH and time derivative Xdot
    gradHdata = gradH(Xdata, B, a, p, v)
    XdataDot  = FDapprox(Xdata.T, dt).T

    return Xdata, XdataDot, gradHdata

# Precomputing the hamiltonian KdV intrusive ROM operators once and for all
def build_H_KdV_ROM_Ops(UU, B, ic, r=50, a=-6, p=0, v=-1, MC=False):
    ic      = ic.flatten()
    U      = UU[:,:r]
    N       = U.shape[0]
    
    # LHat    = U.T @ A @ U (A might be used for Galerkin so will keep here)
    cVec  = np.zeros(r)
    Cmat  = U.T @ (p*identity(N)+v*B) @ U
    temp1   = np.einsum('ia,ib->iab', U, U)
    Ttens = a/2 * np.einsum('ia,ibc', U, temp1)

    # Extra terms in case of mean-centering (implement later)
    if MC:
        cVec += U.T @ (a/2 * (ic**2) + (p*identity(N)+v*B) @ ic)
        Cmat += a * U.T @ (ic.reshape(-1,1) * U)

    return {
        "cHat":cVec,
        "CHat":Cmat,
        "THat":Ttens,
        # "LHat":LHat
    }

# Builds required matrices for NC-H-OpInf Calculation
def build_OpInf_eq(UU, xDotData, gradHdata, r=50):
    # Reduced dimension
    U = UU[:,:r]
    
    xDotHat     = U.T @ xDotData
    gradHatH    = U.T @ gradHdata
    sgHatH      = gradHatH @ gradHatH.T
    rhs         = xDotHat @ gradHatH.T - gradHatH @ xDotHat.T

    return [sgHatH, rhs]

# Vectorize column-wise
def vec(A):
    m, n = A.shape[0], A.shape[1]
    return A.reshape(m*n, order='F')

# Infers Lhat operator
def NC_H_OpInf(OpList, r, eps=1e-12):
    sgHatH  = OpList[0][:r, :r]
    rhs     = OpList[1][:r, :r]

    I = np.eye(r)
    P = csc_matrix( np.kron(I, sgHatH) + np.kron(sgHatH, I))
    reg = eps * identity(r*r)
    Lhat = spsolve(P+reg, vec(rhs)).reshape((r,r), order="F")

    return 0.5 * (Lhat - Lhat.T)

# Function to integrate the ROMs for KdV v1
# OpList assumes order of [cVec, Cmat, Ttens, L]
# This function is overloaded for BBM case also
# def integrate_KdV_ROM(tTest, OpList, ic, UU, r, MC=False, 
#                           Hamiltonian=True, Newton=True):
#     nt = tTest.shape[0]
#     dt = tTest[1] - tTest[0]
#     ic = ic.reshape(-1,1)

#     # Building operators for ROM problem
#     U     = UU[:,:r]
#     cVec  = OpList[0][:r]
#     Cmat  = OpList[1][:r, :r]
#     Ttens = OpList[2][:r, :r, :r]

#     # if Hamiltonian:
#     LHat = OpList[-1][:r, :r]
#     # else:
#     #     LHat = np.eye(n)

#     # Functions for root finding
#     def Nfunc(xHatOld, xHatNew):
#         xHatMid   = 0.5 * (xHatOld + xHatNew)
#         tensTerm  = ( 2*(Ttens @ xHatOld) @ xHatMid
#                   + (Ttens @ xHatNew) @ xHatNew ) / 3
#         rhs       = cVec + Cmat @ xHatMid + tensTerm
#         return xHatNew - (xHatOld + dt * LHat @ rhs)

#     Id = np.eye(r)
#     def Nderiv(xHatOld, xHatNew):
#         tensTerm  = Ttens @ xHatOld + 2*Ttens @ xHatNew
#         return Id - dt * LHat @ (Cmat/2 + tensTerm/3) 

#     # Initialize array and set initial conditions
#     xHat = np.zeros((r, nt))
    
#     # if MC:
#     #     xHat[:,0] = np.zeros(n)
#     # else:
#     xHat[:,0] = U.T @ ic.flatten()

#     # Integrate FOM/ROMs over test interval
#     for i,time in enumerate(tTest[:-1]):
#         res = partial(Nfunc, xHat[:,i])
#         if not Newton:
#             xHat[:,i+1] = root(res, xHat[:,i], method='krylov').x
#         else:
#             jac = partial(Nderiv, xHat[:,i])  
#             xHat[:,i+1] = do_newton(res, jac, xHat[:,i], 
#                                     maxIt=3, sparse=False)

#     # Reconstruct FO solutions
#     # if MC:
#     #     xRec = ic + U @ xHat
#     # else:
#     xRec = U @ xHat

#     return xRec

def integrate_KdV_ROM(tTest, OpList, xHat0, r, MC=False, 
                          Hamiltonian=True, Newton=True):
    nt = tTest.shape[0]
    dt = tTest[1] - tTest[0]

    # Building operators for ROM problem
    cVec  = OpList[0][:r]
    Cmat  = OpList[1][:r, :r]
    Ttens = OpList[2][:r, :r, :r]

    # if Hamiltonian:
    LHat = OpList[-1][:r, :r]
    # else:
    #     LHat = np.eye(n)

    # Functions for root finding
    def Nfunc(xHatOld, xHatNew):
        xHatMid   = 0.5 * (xHatOld + xHatNew)
        tensTerm  = ( 2*(Ttens @ xHatOld) @ xHatMid
                  + (Ttens @ xHatNew) @ xHatNew ) / 3
        rhs       = cVec + Cmat @ xHatMid + tensTerm
        return xHatNew - (xHatOld + dt * LHat @ rhs)

    Id = np.eye(r)
    def Nderiv(xHatOld, xHatNew):
        tensTerm  = Ttens @ xHatOld + 2*Ttens @ xHatNew
        return Id - dt * LHat @ (Cmat/2 + tensTerm/3) 

    # Initialize array and set initial conditions
    xHat = np.zeros((r, nt))
    
    # if MC:
    #     xHat[:,0] = np.zeros(n)
    # else:
    xHat[:,0] = xHat0

    # Integrate FOM/ROMs over test interval
    for i,time in enumerate(tTest[:-1]):
        res = partial(Nfunc, xHat[:,i])
        if not Newton:
            xHat[:,i+1] = root(res, xHat[:,i], method='krylov').x
        else:
            jac = partial(Nderiv, xHat[:,i])  
            xHat[:,i+1] = do_newton(res, jac, xHat[:,i], 
                                    maxIt=3, sparse=False)

    return xHat

def plot_trajectory(x_plot, x_value, dt, title, name_of_file):
    """
    Parameters
    -----------
    x_plot : ndarray
    x_value : ndarray
    """
    fig, ax = plt.subplots()
    line, = ax.plot(x_plot, x_value[:, 0], lw=2)

    ax.set_xlim([-10, 10])
    ax.set_ylim([-1, 9])
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title(title)

    def update(frame):
        line.set_ydata(x_value[:, frame])
        ax.set_title(f't = {frame * dt:.2f}')
        return line,

    ani = animation.FuncAnimation(fig, update, frames=range(x_value.shape[1]), blit=True, interval=20)
    plt.show()
    ani.save(name_of_file, writer='pillow')

def animate_multiple_trajectories(x_plot, true_val, mean_val, dt, title, name_of_file):
    """

    :param x_plot:
    :param true_val:
    :param mean_val:
    :param dt:
    :param title:
    :param name_of_file:
    :return:
    """
    fig, ax = plt.subplots()
    line1, = ax.plot(x_plot, true_val[:, 0], lw=2)
    line2, = ax.plot(x_plot, mean_val[:, 0], lw=2)

    ax.set_xlim([-10, 10])
    ax.set_ylim([-1, 9])
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title(title)

    def update(frame):
        line1.set_ydata(true_val[:, frame])
        line2.set_ydata(mean_val[:, frame])
        ax.set_title(f't = {frame * dt:.2f}')
        return line1, line2

    print("animating frames")
    ani = animation.FuncAnimation(fig, update, frames=range(true_val.shape[1]), blit=True, interval=2)
    plt.show()
    ani.save(name_of_file, writer='pillow')

# Relative L2 error
def relError(x, xHat):
    num = norm(x - xHat)
    den = norm(x)
    return num / den