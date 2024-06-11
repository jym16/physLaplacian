import igraph as ig
import numpy as np
from scipy.optimize import root_scalar
from scipy.sparse import coo_array
from scipy.sparse.linalg import eigsh

import matplotlib.pyplot as plt
import multiprocessing as mp
import pickle
from tqdm import tqdm

def inverse_participation_ratio(vec):
    """Calculate the inverse participation ratio of a vector.
    
    Parameters
    ----------
    vec : np.array
        The vector for which to calculate the inverse participation ratio.
    
    Returns
    -------
    ipr : float
        The inverse participation ratio of the vector.
    """
    return np.sum(vec**4.)/(np.sum(vec**2.)**2.)

def physical_eigen_values(g, vols, which='leading'):
    """Calculate the eigenvalue of the physical Laplacian.

    Parameters
    ----------
    g : igraph.Graph
        The graph for which to calculate the eigenvalue.
    
    vols : np.array
        The volume of each vertex in the graph.
    
    which : str
        Which eigenvalue to calculate. Options are 'leading' and 'fiedler'.
    
    Returns
    -------
    eigs[0] : float
        The eigenvalue of the physical Laplacian.
    """
    
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
    
    if which == 'leading':
        eigs, vecs = eigsh(physLap, k=1, which='LM', return_eigenvectors=True)
        return eigs[0], vecs[:,0]
    elif which == 'fiedler':
        eigs, vecs = eigsh(physLap, k=2, which='SM', return_eigenvectors=True)
        return eigs[1], vecs[:,1]

def run_simulation(params):
    # id, n_nodes, p, d_hub, alpha = params
    n_nodes, p, d_hub, alpha = params
    g = ig.Graph.Erdos_Renyi(n_nodes, p, directed=False, loops=False)
    g.add_vertex()
    for i in range(n_nodes):
        if np.random.rand() < (d_hub / n_nodes):
            g.add_edge(n_nodes, i)
    
    degs = np.array(g.degree())
    vols = degs**alpha
    eigs, vec = physical_eigen_values(g, vols, which='leading')
    ipr = inverse_participation_ratio(vec)

    data = {
        # 'id': id,
        'n_nodes': n_nodes,
        'p': p,
        'd_hub': d_hub,
        'alpha': alpha,
        'eigs': eigs,
        'ipr': ipr
    }

    with open(f'./data/ERRG/N={n_nodes:1.2e}_d={d_hub:1.2e}_alp={alpha:1.2f}.pkl', 'wb') as f:
    # with open(f'./data/ERRG_{id}_N={n_nodes:1.2e}_alp={alpha:1.2f}.pkl', 'wb') as f:
        pickle.dump(data, f)
    return eigs, ipr

def main():
    n_procs = 4

    nums_nodes = [int(1e3), int(1e4), int(1e5)]
    avg_deg = 4.
    ps = [avg_deg / n_nodes for n_nodes in nums_nodes]
    d_hubs = (avg_deg * np.array([5, 10, 20])).astype(int)
    alphas = np.linspace(0., 2., 21)
    
    # Parallelize the simulation
    with mp.Pool(n_procs) as pool:
        for n_nodes in nums_nodes:
            for p in ps:
                for d_hub in d_hubs:
                    params = [(n_nodes, p, d_hub, alpha) for alpha in alphas]
                    results = pool.map(run_simulation, params)
                    # for alpha in alphas:
                        # params = [(id, n_nodes, p, d_hub, alpha) for id in range(10)]
                        # pool.apply_async(run_simulation, args=(params,))
                    
    return 0

if __name__ == '__main__':
    main()

