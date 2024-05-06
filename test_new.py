# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric
from loguru import logger
from tqdm import tqdm
def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
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
    num_valid_q = 0.  # number of valid query
    for q_idx in tqdm(range(num_q), 'Computing CMC'):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
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
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP





import torch
import torch.nn.functional as F
import numpy as np

class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes', metric='cosine'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.metric = metric

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def euclidean_dist(self, qf, gf):
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        return distmat

    def cosine_dist(self, qf, gf):
        qf_norm = F.normalize(qf, p=2, dim=1)
        gf_norm = F.normalize(gf, p=2, dim=1)
        distmat = 1 - torch.mm(qf_norm, gf_norm.t())
        return distmat

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = F.normalize(feats, dim=1, p=2)
        
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        if self.metric == 'euclidean':
            distmat = self.euclidean_dist(qf, gf)
        elif self.metric == 'cosine':
            distmat = self.cosine_dist(qf, gf)
        else:
            raise ValueError("Unsupported metric: {}".format(self.metric))

        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP




def test(model, query_loader, gallery_loader,tested_feature,cfg):
    model.eval()  # 确保模型处于评估模式
    logger.info(f'Testing on {tested_feature} feature.')
    r1_map_evaluator = R1_mAP(num_query=len(query_loader.dataset), max_rank=50, feat_norm='yes',metric='cosine')
    with torch.no_grad():
        for data in tqdm(query_loader,'Extracting query images',total=len(query_loader)):
            imgs, pids, camids = data
            imgs = imgs.to('cuda:0')
            gf,lf = model(imgs)  # 确保模型输出与R1_mAP中的更新逻辑相匹配
            if tested_feature == 'global':
                r1_map_evaluator.update((gf, pids, camids))
            elif tested_feature =='local':
                r1_map_evaluator.update((lf, pids, camids))
            elif tested_feature =='concat':
                r1_map_evaluator.update((torch.cat([gf,lf],dim=1), pids, camids))
            else:
                raise f'No such tested_feature name:{tested_feature}'

        for data in tqdm(gallery_loader,'Extracing gallery images',total=len(gallery_loader)):
            imgs, pids, camids = data
            imgs = imgs.to('cuda:0')
            gf,lf = model(imgs)
            if tested_feature == 'global':
                r1_map_evaluator.update((gf, pids, camids))
            elif tested_feature =='local':
                r1_map_evaluator.update((lf, pids, camids))
            elif tested_feature =='concat':
                r1_map_evaluator.update((torch.cat([gf,lf],dim=1), pids, camids))
            else:
                raise f'No such tested_feature name:{tested_feature}'
    
    cmc, mAP = r1_map_evaluator.compute()
    logger.info(f"Test Results({tested_feature})")
    logger.info(f"mAP: {mAP:.1%}")
    for r in cfg.test.ranks:
        logger.info(f"CMC curve, Rank-{r}: {cmc[r - 1]:.1%}")
    return cmc, mAP

# class R1_mAP_reranking(Metric):
#     def __init__(self, num_query, max_rank=50, feat_norm='yes'):
#         super(R1_mAP_reranking, self).__init__()
#         self.num_query = num_query
#         self.max_rank = max_rank
#         self.feat_norm = feat_norm

#     def reset(self):
#         self.feats = []
#         self.pids = []
#         self.camids = []

#     def update(self, output):
#         feat, pid, camid = output
#         self.feats.append(feat)
#         self.pids.extend(np.asarray(pid))
#         self.camids.extend(np.asarray(camid))

#     def compute(self):
#         feats = torch.cat(self.feats, dim=0)
#         if self.feat_norm == 'yes':
#             print("The test feature is normalized")
#             feats = torch.nn.functional.normalize(feats, dim=1, p=2)

#         # query
#         qf = feats[:self.num_query]
#         q_pids = np.asarray(self.pids[:self.num_query])
#         q_camids = np.asarray(self.camids[:self.num_query])
#         # gallery
#         gf = feats[self.num_query:]
#         g_pids = np.asarray(self.pids[self.num_query:])
#         g_camids = np.asarray(self.camids[self.num_query:])
#         # m, n = qf.shape[0], gf.shape[0]
#         # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#         #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#         # distmat.addmm_(1, -2, qf, gf.t())
#         # distmat = distmat.cpu().numpy()
#         print("Enter reranking")
#         distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
#         cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

#         return cmc, mAP