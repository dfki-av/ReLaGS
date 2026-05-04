import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d

def get_oriented_bounding_boxes(points, labels, obj_ids, remove_outlier=False):
    obj_id_to_obb = {}
    for obj_id in obj_ids:
        obj_points = points[labels.squeeze() == obj_id]
        # --- Convert numpy array to Open3D PointCloud ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        
        if remove_outlier:
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd = pcd.select_by_index(ind)
        mass_center = np.asarray(pcd.get_center())

        # --- Compute oriented bounding box ---
        obb = pcd.get_oriented_bounding_box()

        # --- Extract as numpy arrays ---
        center = np.asarray(obb.center)
        R = np.asarray(obb.R)
        extent = np.asarray(obb.extent)
        obj_id_to_obb[obj_id] = (center, R, extent, mass_center)
    return obj_id_to_obb

def get_oriented_bounding_box(obj_points, remove_outlier=False):
    # --- Convert numpy array to Open3D PointCloud ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_points)
    
    if remove_outlier:
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)
    # --- Compute oriented bounding box ---
    obb = pcd.get_oriented_bounding_box()
    mass_center = np.asarray(pcd.get_center())
    cetner = np.asarray(obb.center)
    R = np.asarray(obb.R)
    extent = np.asarray(obb.extent)
    return (cetner, R, extent, mass_center)

def obb_min_distance_matrix_numpy(centers, rotations, extents, dist_thresh=None, topk=None):
    """
    Compute approximate pairwise minimal distances between oriented bounding boxes (OBBs)
    and return both the distance matrix and adjacency mask.

    Args:
        centers:   (N, 3) ndarray - OBB centers
        rotations: (N, 3, 3) ndarray - rotation matrices (columns are axes)
        extents:   (N, 3) ndarray - half-lengths along local x/y/z axes
        dist_thresh: float, optional. If given, adjacency_mask[i,j] = True if distance < thresh
        topk: int, optional. If given, keep top-K nearest neighbors per node (including both directions)
                (overrides dist_thresh if both given)

    Returns:
        dist_mat:        (N, N) ndarray of minimal OBB distances (symmetric, zeros on diagonal)
        adjacency_mask:  (N, N) boolean ndarray (True where edge should exist)
    """
    N = len(centers)
    centers = np.asarray(centers, float)
    rotations = np.asarray(rotations, float)
    extents = np.asarray(extents, float)

    # --- pairwise center differences ---
    d = centers[:, None, :] - centers[None, :, :]          # (N, N, 3)

    # --- rotation differences ---
    R_i_T = rotations.transpose(0, 2, 1)                   # (N, 3, 3)
    R_diff = np.einsum('iab,jbc->ijac', R_i_T, rotations)  # (N, N, 3, 3)
    absR = np.abs(R_diff)                                  # (N, N, 3, 3)

    # --- transform center differences into i's local frame ---
    d_local = np.einsum('iab,ijb->ija', R_i_T, d)          # (N, N, 3)

    # --- expanded half-sizes ---
    # expand[i,j,:] = extents[i,:] + |R_i^T R_j| @ extents[j,:]
    expand = extents[:, None, :] + np.einsum('ijab,jb->ija', absR, extents)  # (N, N, 3)

    # --- axis-wise separation ---
    delta = np.abs(d_local) - expand
    delta = np.maximum(delta, 0.0)
    dist = np.linalg.norm(delta, axis=-1)                  # (N, N)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)

    # --- adjacency mask ---
    if topk is not None:
        # Keep top-K smallest distances (excluding self)
        adj = np.zeros_like(dist, dtype=bool)
        idx = np.argsort(dist, axis=1)
        for i in range(N):
            k = min(topk + 1, N)
            adj[i, idx[i, 1:k]] = True  # exclude self
        adjacency_mask = np.logical_or(adj, adj.T)  # make symmetric
    elif dist_thresh is not None:
        adjacency_mask = dist < dist_thresh
        np.fill_diagonal(adjacency_mask, False)
    else:
        adjacency_mask = np.ones((N, N), dtype=bool)
        np.fill_diagonal(adjacency_mask, False)

    return dist, adjacency_mask

def get_edge_geom_features(centers, R, S, edge_index, gravity=torch.tensor([0,0,1.0])):
    """
    Compute geometric edge features between OBB pairs.
    centers: (N,3)
    R: (N,3,3)
    S: (N,3)  half-lengths
    edge_index: (2,E) tensor
    gravity: (3,) vector
    Returns:
        x_geo: (E, D_geo) tensor (~19 dims)
    """
    
    if isinstance(centers, np.ndarray):
        centers = torch.tensor(centers, dtype=torch.float32)
    if isinstance(R, np.ndarray):
        R = torch.tensor(R, dtype=torch.float32)
    if isinstance(S, np.ndarray):
        S = torch.tensor(S, dtype=torch.float32)
    gravity = gravity.to(centers.device)
    i, j = edge_index[0,:], edge_index[1,:]  # (E,)
    ci, cj = centers[i], centers[j]
    Ri, Rj = R[i], R[j]
    Si, Sj = S[i], S[j]

    # 1. center difference
    d = cj - ci
    dist = torch.norm(d, dim=-1, keepdim=True)
    d_hat = d / (dist + 1e-6)

    # 2. axis alignment (similarity of principal axes)
    axis_align = torch.stack([
        (Ri[:,:,0]*Rj[:,:,0]).sum(-1),
        (Ri[:,:,1]*Rj[:,:,1]).sum(-1),
        (Ri[:,:,2]*Rj[:,:,2]).sum(-1),
    ], dim=-1)

    # 3. size ratio and volume diff
    size_ratio = Sj / (Si + 1e-6)
    vol_log_diff = (torch.log(Sj.prod(-1)+1e-6) - torch.log(Si.prod(-1)+1e-6)).unsqueeze(-1)

    # 4. OBB gap (same formula as distance matrix)
    RiT = Ri.transpose(1,2)
    dj_local = (RiT @ d.unsqueeze(-1)).squeeze(-1)           # (E,3)
    A = torch.abs(RiT @ Rj)                                  # (E,3,3)
    expand = Si + (A @ Sj.unsqueeze(-1)).squeeze(-1)         # (E,3)
    delta = torch.clamp(torch.abs(dj_local) - expand, min=0)
    obb_gap = delta.norm(dim=-1, keepdim=True)

    # 5. vertical offset
    g = gravity#.to(centers.device)
    vert_offset = (d @ g).unsqueeze(-1)

    # 6. overlap ratio (axis-wise intersection proportion)
    overlap_axis = torch.clamp(1 - delta / (Si + Sj + 1e-6), 0, 1)

    # concatenate all
    edge_geo_feat = torch.cat([
        d, dist, d_hat, axis_align,
        size_ratio, vol_log_diff,
        obb_gap, vert_offset, overlap_axis
    ], dim=-1)
    return edge_geo_feat


def build_geom_features(centers, R, extents):
    """
    centers: (N, 3)
    R: (N, 3, 3) rotation matrix
    extents: (N, 3) half-lengths along local x/y/z axes
    return: (N, 19) geometry feature tensor
    """
    if isinstance(centers, np.ndarray):
        centers = torch.from_numpy(centers)
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)
    if isinstance(extents, np.ndarray):
        extents = torch.from_numpy(extents)
    # ensure float tensors
    centers = centers.float()
    R = R.float()
    extents = extents.float()

    # --- basic geometric quantities ---
    s = extents                                         # (N,3)
    log_s = torch.log(s.clamp_min(1e-6))                # (N,3)
    
    # aspect ratios: (a/b, b/c, a/c)
    ratios = torch.stack([
        s[:, 0] / (s[:, 1] + 1e-6),
        s[:, 1] / (s[:, 2] + 1e-6),
        s[:, 0] / (s[:, 2] + 1e-6),
    ], dim=-1)                                          # (N,3)

    # volume of full OBB = 8 * a*b*c  (since s are half lengths)
    vol = (8.0 * s.prod(dim=-1, keepdim=True))          # (N,1)

    # 6D orientation representation (Zhou et al.)
    orient6 = torch.cat([R[:, :, 0], R[:, :, 1]], dim=-1)  # (N,6)

    # --- concatenate all ---
    geom_feat = torch.cat([centers, s, log_s, ratios, vol, orient6], dim=-1)  # (N,19)
    return geom_feat

def normalize_scene_centers(C):
    # C: Nx3; center to mean and scale by 95th percentile radius
    mu = C.mean(0, keepdim=True)
    X = C - mu
    scale = torch.quantile(X.norm(dim=1), 0.95).clamp_min(1e-6)
    return X/scale, mu, scale

def aggregate_clip(clip_feats, pixel_nums):
    
    weights = pixel_nums / max(pixel_nums) + 1e-6
    weights = torch.tensor(weights, dtype=clip_feats.dtype, device=clip_feats.device)
    weighted_feat = (clip_feats * weights.unsqueeze(-1)).sum(dim=0)
    weighted_feat = F.normalize(weighted_feat, dim=-1)

    return weighted_feat

def build_edge_index_knn(C, Kc=16):
    # C: Nx3
    N = C.shape[0]
    # brute-force; replace by faiss if large
    d2 = torch.cdist(C, C, p=2)  # NxN
    idx = d2.topk(Kc+1, largest=False).indices[:,1:]  # drop self
    src = torch.arange(N).repeat_interleave(Kc)
    dst = idx.reshape(-1)
    ei = torch.stack([src, dst], dim=0)  # 2xE
    return ei

def edge_features(C, R, S, G, has_clip, ei, gvec=torch.tensor([0,0,1.0])):
    # C: Nx3, R: Nx3x3, S: Nx3, G: Nx512
    # ei: 2xE
    i, j = ei
    ci, cj = C[i], C[j]
    di = cj - ci
    dist = di.norm(dim=-1, keepdim=True)
    di_hat = di / (dist + 1e-6)
    # orientation/axis align
    xi, yi, zi = R[i,:,0], R[i,:,1], R[i,:,2]
    xj, yj, zj = R[j,:,0], R[j,:,1], R[j,:,2]
    axis_align = torch.stack([ (xi*xj).sum(-1), (yi*yj).sum(-1), (zi*zj).sum(-1) ], dim=-1)
    g = gvec.to(C.device).expand_as(ci)
    grav = torch.stack([(zi*g).sum(-1), (zj*g).sum(-1), (di_hat*g).sum(-1)], dim=-1)
    # size ratios and volume diff
    Si, Sj = S[i], S[j]
    size_ratio = (Sj / (Si + 1e-6))
    vol_log_diff = (torch.log(Sj.prod(-1)+1e-6) - torch.log(Si.prod(-1)+1e-6)).unsqueeze(-1)
    # SAT-like separation in i-frame
    RiT = R[i].transpose(1,2)
    dj_local = (RiT @ di.unsqueeze(-1)).squeeze(-1)
    A = torch.abs((RiT @ R[j]))  # Nx3x3
    expand = Si + (A @ Sj.unsqueeze(-1)).squeeze(-1)  # Nx3
    delta = torch.clamp(torch.abs(dj_local) - expand, min=0)
    d_sat = delta.norm(dim=-1, keepdim=True)
    # overlap flags (axiswise)
    overlap_flags = (delta < 1e-3).float()  # 3 dims
    # CLIP cosine
    cos = F.cosine_similarity(G[i], G[j]).unsqueeze(-1)
    # has_clip flags
    hc = torch.cat([has_clip[i], has_clip[j]], dim=-1)
    # pack
    x = torch.cat([di, dist, di_hat, axis_align, grav, size_ratio, vol_log_diff, d_sat, overlap_flags, cos, hc], dim=-1)
    return x  # ExD
