import torch
from torch import nn


class TripletHardLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletHardLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, features, labels):
        n = features.size(0)
        # 计算所有样本间的距离矩阵
        dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist = dist - 2 * torch.mm(features, features.t())
        dist = torch.sqrt(dist.clamp(min=1e-12))

        # 对于每个锚点样本找到最难正样本和最难负样本
        mask = labels.expand(n, n).eq(labels.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)

        dist_an = torch.cat(dist_an)

        # 计算triplet loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss