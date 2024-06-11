import igraph as ig
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
# from scipy.linalg import eigh
from scipy.sparse import coo_array
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

n_procs = 4

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Time",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
})

def inverse_participation_ratio(v):
    return np.sum(v**4.)/(np.sum(v**2.)**2.)

def physical_eigen_values(g, vols, which="leading"):
    row=[]
    col=[]
    val=[]
    for e in g.es:
        row.append(e.source)
        col.append(e.target)
        val.append(-1)

        row.append(e.target)
        col.append(e.source)
        val.append(-1)

    for vid, k in enumerate(g.degree()):
        row.append(vid)
        col.append(vid)
        val.append(k)
    
    Lap = coo_array((val, (row, col)), shape=(g.vcount(), g.vcount())).tocsc()
    
    row=[]
    col=[]
    val=[]
    for v in g.vs:
        row.append(v.index)
        col.append(v.index)
        val.append(vols[v.index])
    val = np.array(val)
    val = val**-.5

    vs = coo_array((val, (row, col)), shape=(g.vcount(), g.vcount())).tocsc()

    physLap = vs.dot(Lap.dot(vs))
    
    if which == "leading":
        eigs, vecs = eigsh(physLap, k=1, which="LM", return_eigenvectors=True)

        return eigs[0],vecs[:,0]
    elif which == "fiedler":
        eigs, vecs = eigsh(physLap, k=2, which="SM", return_eigenvectors=True)

        return eigs[1],vecs[:,1]

n_nodes = 10000
alpha = np.linspace(0., 2., 21)

# k_avg = 3.
# n_edges = int(k_avg * n_nodes / 2)
# g = ig.Graph.Erdos_Renyi(
#     n=n_nodes, m=n_edges, directed=False, loops=False
# )

# m = 2
# g = ig.Graph.Barabasi(
#     n=n_nodes, m=m, directed=False
# )

k_avg = 3.
n_edges = int(k_avg * n_nodes / 2)
# gamma = 3.

for gamma in tqdm([2., 2.5, 3., 3.5, 4.]):
    g = ig.Graph.Static_Power_Law(
        n=n_nodes, m=n_edges, exponent_out=gamma
    )
    degrees = np.array(g.degree())
    if np.min(degrees) == 0:
        degrees += 1

    leading_eigs = {}
    leading_iprs = {}
    fiedler_eigs = {}
    fiedler_iprs = {}

    for i, alp in tqdm(enumerate(alpha)):
        vols = degrees ** alp
        vols /= np.sum(vols)
        
        e, v = physical_eigen_values(g, vols, which="leading")
        leading_eigs[alp] = e
        leading_iprs[alp] = inverse_participation_ratio(v)
        
        e, v = physical_eigen_values(g,vols, which="fiedler")
        fiedler_eigs[alp] = e
        fiedler_iprs[alp] = inverse_participation_ratio(v)

    fig, axs = plt.subplots(2, 2, figsize=(8, 5))

    for ax in axs.flatten():
        ax.set_xlim(0., 2.)
        ax.set_xlabel(r"$\alpha$");


    axs[0, 0].set_title("(a) Leading")
    axs[0, 1].set_title("(b) Fielder")

    axs[0, 0].plot(alpha, leading_eigs.values(), "-ob")
    axs[0, 0].set_ylabel(r"$\lambda_{N}$")
    # axs[0, 0].set_xlabel(r"$\alpha$");
    # axs[0, 0].annotate("(a-1)", xy=(0.05, 0.9), xycoords="axes fraction")

    axs[1, 0].plot(alpha, leading_iprs.values(), "r-o")
    axs[1, 0].set_ylabel(r"IPR($\vec{v}_{N}$)")
    # axs[1, 0].set_xlabel(r"$\alpha$");
    # axs[1, 0].annotate("(a-2)", xy=(0.05, 0.9), xycoords="axes fraction")

    axs[0, 1].plot(alpha, fiedler_eigs.values(), "-og")
    axs[0, 1].set_ylabel(r"$\lambda_{2}$")
    # axs[0, 1].set_xlabel(r"$\alpha$");
    # axs[0, 1].annotate("(b-1)", xy=(0.05, 0.9), xycoords="axes fraction")

    axs[1, 1].plot(alpha, fiedler_iprs.values(), "y-o")
    axs[1, 1].set_ylabel(r"IPR($\vec{v}_{2}$)")
    # axs[1, 1].set_xlabel(r"$\alpha$");
    # axs[1, 1].annotate("(b-2)", xy=(0.05, 0.9), xycoords="axes fraction")

    fig.tight_layout()
    fig.savefig("./fig/config_N={:1.1e}_gamma={:1.1f}_localization.pdf".format(n_nodes, gamma))
    plt.close()
