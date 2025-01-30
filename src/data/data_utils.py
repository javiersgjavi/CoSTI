import torch
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import haversine_distances

def interpolate_sample(sample):
    return torch.tensor(pd.DataFrame(sample).interpolate(method='linear', axis=0).backfill(axis=0).ffill(axis=0).fillna(0).values)

def create_interpolation(batch):
    x = batch['x']
    x = torch.where(batch['mask']==0, torch.nan, x).cpu()

    B, T, N, C = x.shape
    x_interpolated = torch.stack(
        [interpolate_sample(x[i].view(T, N)).reshape(T, N, C) for i in range(B)]
    ).to(batch['mask'].device)

    batch['x_interpolated'] = x_interpolated
    return batch

def redefine_eval_mask(batch):
    og_mask = batch['og_mask']
    cond_mask = batch['mask']
    eval_mask = og_mask.int() - cond_mask
    batch['eval_mask'] = eval_mask
    return batch

# ------------------------ GNN utils ------------------------------------------------


def geographical_distance(x=None, to_rad=True):
    _AVG_EARTH_RADIUS_KM = 6371.0088

    # Extract values of X if it is a DataFrame, else assume it is 2-dim array of lat-lon pairs
    latlon_pairs = x.values if isinstance(x, pd.DataFrame) else x

    # If the input values are in degrees, convert them in radians
    if to_rad:
        latlon_pairs = np.vectorize(np.radians)(latlon_pairs)

    distances = haversine_distances(latlon_pairs) * _AVG_EARTH_RADIUS_KM

    # Cast response
    if isinstance(x, pd.DataFrame):
        res = pd.DataFrame(distances, x.index, x.index)
    else:
        res = distances

    return res

def thresholded_gaussian_kernel(x, theta=None, threshold=None, threshold_on_input=False):
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    if threshold is not None:
        mask = x > threshold if threshold_on_input else weights < threshold
        weights[mask] = 0.
    return weights

def get_similarity_AQI(dist, thr=0.1, include_self=False, force_symmetric=False, sparse=False):
    theta = np.std(dist[:36, :36])  # use same theta for both air and air36
    adj = thresholded_gaussian_kernel(dist, theta=theta, threshold=thr)
    if not include_self:
        adj[np.diag_indices_from(adj)] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj

def get_similarity_pemsbay(thr=0.1, force_symmetric=False, sparse=False):
    dist = np.load('../../data/pems_bay/pems_bay_dist.npy')
    finite_dist = dist.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj

def get_adj_AQI36():
    df = pd.read_csv("../../data/aqi-36/pm25_latlng.txt")
    df = df[['latitude', 'longitude']]
    res = geographical_distance(df, to_rad=False).values
    adj = get_similarity_AQI(res)
    return adj

def get_similarity_metrla(thr=0.1, force_symmetric=False, sparse=False):
    dist = np.load('../../data/metr_la/metr_la_dist.npy')
    finite_dist = dist.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj

def get_similarity_mimic(thr=0.1, force_symmetric=False, sparse=False):
    dist = np.load('../../data/mimic-iii/mimic_graph.npy')
    finite_dist = dist.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj

def get_similarity_mimic_challenge(thr=0.1, force_symmetric=False, sparse=False):
    dist = np.load('../../data/mimic-iii_challenge/mimic_graph.npy')
    finite_dist = dist.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj

