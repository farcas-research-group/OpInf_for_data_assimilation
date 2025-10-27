from matplotlib.ticker import StrMethodFormatter

from utils.utils import *

nEigs = 100
idx = [i for i in range(nEigs)]

UU = np.load("data/Basis/UU.npy")
SS = np.load("data/Basis/SS.npy")
X = np.load("data/snapshots/X.npy")

# nList = [4*(i+1) for i in range(15)]
# errU2r = np.zeros(len(nList))
# for i,n in enumerate(nList):
#     U       = UU[:,:n]
#     reconU2  = U @ U.T @ X
#     errU2r[i] = relError(X, reconU2)

energy = np.cumsum(SS[:nEigs] / np.sum(SS))*100
print("Energy at r=50:", energy[50])
Threshold = 99.99
energy_rank = np.argwhere(energy >= Threshold)
print("Over 99.99% of energy preserved", energy_rank[0])

fig, ax = plt.subplots()
ax.scatter(idx, energy, s=10, label='Ordinary POD (no MC)')
ax.set_title('POD Snapshot Energy')
# ax[0].set_yscale('log')
# ax[1].semilogy(nList, errU2r, label='Ordinary POD (no MC)', marker='o', linestyle='-', markersize=5)
# ax[1].set_title('POD Projection Error')
# ax[1].set_ylabel('relative $L^2$ error')
# ax[0].get_shared_y_axes().join(ax[0], ax[1])
# ax[1].set_xticklabels([])
# for i in range(1):
ax.minorticks_on()
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
ax.set_ylabel('% POD energy')
# for i in range(2):
ax.set_xlabel('basis size $n$')
ax.legend(prop={'size': 8})
plt.tight_layout()
plt.savefig('KdVpodEnergy')
plt.show()