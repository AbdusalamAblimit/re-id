

import time

import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from tqdm import tqdm


from IPython import embed


def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat = None, only_local = False):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea,galFea])
        print('using GPU to compute original distance')
        distmat = torch.pow(feat,2).sum(dim=1, keepdim=True).expand(all_num,all_num) + \
                      torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distmat.addmm_(1,-2,feat,feat.t())
        original_dist = distmat.numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist



def normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, type='euclidean'):
    """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
    assert type in ['cosine', 'euclidean']
    if type == 'cosine':
        array1 = normalize(array1, axis=1)
        array2 = normalize(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)
        return dist


def shortest_dist(dist_mat):
    """Parallel version.
  Args:
    dist_mat: numpy array, available shape
      1) [m, n]
      2) [m, n, N], N is batch size
      3) [m, n, *], * can be arbitrary additional dimensions
  Returns:
    dist: three cases corresponding to `dist_mat`
      1) scalar
      2) numpy array, with shape [N]
      3) numpy array with shape [*]
  """
    m, n = dist_mat.shape[:2]
    dist = np.zeros_like(dist_mat)
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i, j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i, j] = dist[i, j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i, j] = dist[i - 1, j] + dist_mat[i, j]
            else:
                dist[i, j] = \
                    np.min(np.stack([dist[i - 1, j], dist[i, j - 1]], axis=0), axis=0) \
                    + dist_mat[i, j]
    # I ran into memory disaster when returning this reference! I still don't
    # know why.
    # dist = dist[-1, -1]
    dist = dist[-1, -1].copy()
    return dist

def unaligned_dist(dist_mat):
    """Parallel version.
    Args:
      dist_mat: numpy array, available shape
        1) [m, n]
        2) [m, n, N], N is batch size
        3) [m, n, *], * can be arbitrary additional dimensions
    Returns:
      dist: three cases corresponding to `dist_mat`
        1) scalar
        2) numpy array, with shape [N]
        3) numpy array with shape [*]
    """

    m = dist_mat.shape[0]
    dist = np.zeros_like(dist_mat[0])
    for i in range(m):
        dist[i] = dist_mat[i][i]
    dist = np.sum(dist, axis=0).copy()
    return dist


def meta_local_dist(x, y, aligned):
    """
  Args:
    x: numpy array, with shape [m, d]
    y: numpy array, with shape [n, d]
  Returns:
    dist: scalar
  """
    eu_dist = compute_dist(x, y, 'euclidean')
    dist_mat = (np.exp(eu_dist) - 1.) / (np.exp(eu_dist) + 1.)
    if aligned:
        dist = shortest_dist(dist_mat[np.newaxis])[0]
    else:
        dist = unaligned_dist(dist_mat[np.newaxis])[0]
    return dist


# Tooooooo slow!
def serial_local_dist(x, y):
    """
  Args:
    x: numpy array, with shape [M, m, d]
    y: numpy array, with shape [N, n, d]
  Returns:
    dist: numpy array, with shape [M, N]
  """
    M, N = x.shape[0], y.shape[0]
    dist_mat = np.zeros([M, N])
    for i in range(M):
        for j in range(N):
            dist_mat[i, j] = meta_local_dist(x[i], y[j])
    return dist_mat


def parallel_local_dist(x, y, aligned):
    """Parallel version.
  Args:
    x: numpy array, with shape [M, m, d]
    y: numpy array, with shape [N, n, d]
  Returns:
    dist: numpy array, with shape [M, N]
  """
    M, m, d = x.shape
    N, n, d = y.shape
    x = x.reshape([M * m, d])
    y = y.reshape([N * n, d])
    # shape [M * m, N * n]
    dist_mat = compute_dist(x, y, type='euclidean')
    dist_mat = (np.exp(dist_mat) - 1.) / (np.exp(dist_mat) + 1.)
    # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
    dist_mat = dist_mat.reshape([M, m, N, n]).transpose([1, 3, 0, 2])
    # shape [M, N]
    if aligned:
        dist_mat = shortest_dist(dist_mat)
    else:
        dist_mat = unaligned_dist(dist_mat)
    return dist_mat


def local_dist(x, y, aligned):
    if (x.ndim == 2) and (y.ndim == 2):
        return meta_local_dist(x, y, aligned)
    elif (x.ndim == 3) and (y.ndim == 3):
        return parallel_local_dist(x, y, aligned)
    else:
        raise NotImplementedError('Input shape not supported.')

def low_memory_matrix_op(
        func,
        x, y,
        x_split_axis, y_split_axis,
        x_num_splits, y_num_splits,
        verbose=False, aligned=True):
    """
  For matrix operation like multiplication, in order not to flood the memory
  with huge data, split matrices into smaller parts (Divide and Conquer).

  Note:
    If still out of memory, increase `*_num_splits`.

  Args:
    func: a matrix function func(x, y) -> z with shape [M, N]
    x: numpy array, the dimension to split has length M
    y: numpy array, the dimension to split has length N
    x_split_axis: The axis to split x into parts
    y_split_axis: The axis to split y into parts
    x_num_splits: number of splits. 1 <= x_num_splits <= M
    y_num_splits: number of splits. 1 <= y_num_splits <= N
    verbose: whether to print the progress

  Returns:
    mat: numpy array, shape [M, N]
  """

    if verbose:
        import sys
        import time
        printed = False
        st = time.time()
        last_time = time.time()

    mat = [[] for _ in range(x_num_splits)]
    for i, part_x in enumerate(
            np.array_split(x, x_num_splits, axis=x_split_axis)):
        for j, part_y in enumerate(
                np.array_split(y, y_num_splits, axis=y_split_axis)):
            part_mat = func(part_x, part_y, aligned)
            mat[i].append(part_mat)

            if verbose:
                if not printed:
                    printed = True
                else:
                    # Clean the current line
                    sys.stdout.write("\033[F\033[K")
                print('Matrix part ({}, {}) / ({}, {}), +{:.2f}s, total {:.2f}s'
                    .format(i + 1, j + 1, x_num_splits, y_num_splits,
                            time.time() - last_time, time.time() - st))
                last_time = time.time()
        mat[i] = np.concatenate(mat[i], axis=1)
    mat = np.concatenate(mat, axis=0)
    return mat

def low_memory_local_dist(x, y, aligned=True):
    print('Computing local distance...')
    x_num_splits = int(len(x) / 200) + 1
    y_num_splits = int(len(y) / 200) + 1
    z = low_memory_matrix_op(local_dist, x, y, 0, 0, x_num_splits, y_num_splits, verbose=True, aligned=aligned)
    return z


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def test(model, queryloader, galleryloader,cfg):


    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids, lqf = [], [], [], []
        for batch_idx, (imgs, pids, camids) in tqdm(enumerate(queryloader),total=len(queryloader),desc='Extracting features for query set'):
            if cfg.use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)


            features = features.data.cpu()
            local_features = local_features.data.cpu()
            qf.append(features)
            lqf.append(local_features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        lqf = torch.cat(lqf,0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        gf, g_pids, g_camids, lgf = [], [], [], []
        end = time.time()
        for batch_idx, (imgs, pids, camids) in tqdm(enumerate(galleryloader),total=len(galleryloader),desc='Extracting features for gallery set'):
            if cfg.use_gpu: imgs = imgs.cuda()

            end = time.time()
            features, local_features = model(imgs)


            features = features.data.cpu()
            local_features = local_features.data.cpu()
            gf.append(features)
            lgf.append(local_features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        lgf = torch.cat(lgf,0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)



    # feature normlization
    qf = 1. * qf / (torch.norm(qf, 2, dim = -1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim = -1, keepdim=True).expand_as(gf) + 1e-12)
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    if not cfg.test.distance== 'global':
        # print("Only using global branch")

        lqf = lqf.permute(0,2,1)
        lgf = lgf.permute(0,2,1)
        local_distmat = low_memory_local_dist(lqf.numpy(),lgf.numpy(), aligned= cfg.model.aligned)
        if cfg.test.distance== 'local':
            # print("Only using local branch")
            distmat = local_distmat
        if cfg.test.distance == 'global_local':
            # print("Using global and local branches")
            distmat = local_distmat+distmat
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in cfg.test.ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    if cfg.test.reranking:
        if cfg.test.distance == 'global':
            print("Only using global branch for reranking")
            distmat = re_ranking(qf,gf,k1=20, k2=6, lambda_value=0.3)
        else:
            local_qq_distmat = low_memory_local_dist(lqf.numpy(), lqf.numpy(),aligned= cfg.model.aligned)
            local_gg_distmat = low_memory_local_dist(lgf.numpy(), lgf.numpy(),aligned= cfg.model.aligned)
            local_dist = np.concatenate(
                [np.concatenate([local_qq_distmat, local_distmat], axis=1),
                 np.concatenate([local_distmat.T, local_gg_distmat], axis=1)],
                axis=0)
            if cfg.test.distance == 'local':
                print("Only using local branch for reranking")
                distmat = re_ranking(qf,gf,k1=20,k2=6,lambda_value=0.3,local_distmat=local_dist,only_local=True)
            elif cfg.test.distance == 'global_local':
                print("Using global and local branches for reranking")
                distmat = re_ranking(qf,gf,k1=20,k2=6,lambda_value=0.3,local_distmat=local_dist,only_local=False)
        print("Computing CMC and mAP for re_ranking")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

        print("Results ----------")
        print("mAP(RK): {:.1%}".format(mAP))
        print("CMC curve(RK)")
        for r in cfg.test.ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")
    return cmc[0]