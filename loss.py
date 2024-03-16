import torch
from torch import nn
from torch.nn import functional as F
from IPython import embed

# class TripletHardLoss(nn.Module):
#     def __init__(self, margin=0.3):
#         super(TripletHardLoss, self).__init__()
#         self.margin = margin
#         self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
#     def forward(self, features, labels):
#         n = features.size(0)
#         # 计算所有样本间的距离矩阵
#         dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
#         dist = dist + dist.t()
#         dist = dist - 2 * torch.mm(features, features.t())
#         dist = torch.sqrt(dist.clamp(min=1e-12))

#         # 对于每个锚点样本找到最难正样本和最难负样本
#         mask = labels.expand(n, n).eq(labels.expand(n, n).t())
#         dist_ap, dist_an = [], []
#         for i in range(n):
#             dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
#             dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
#         dist_ap = torch.cat(dist_ap)

#         dist_an = torch.cat(dist_an)

#         # 计算triplet loss
#         y = torch.ones_like(dist_an)
#         loss = self.ranking_loss(dist_an, dist_ap, y)
#         return loss
    


# class LabelSmoothingLoss(nn.Module):
#     def __init__(self, classes, smoothing=0.1, dim=-1):
#         super(LabelSmoothingLoss, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.cls = classes
#         self.dim = dim

#     def forward(self, pred, target):
#         pred = pred.log_softmax(dim=self.dim)
#         with torch.no_grad():
#             # Create the smoothed label tensor
#             true_dist = torch.zeros_like(pred)
#             true_dist.fill_(self.smoothing / (self.cls - 1))
#             true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# # # Example of how to replace the standard Cross Entropy Loss with Label Smoothing Loss
# # criterion = LabelSmoothingLoss(classes=NUM_CLASSES, smoothing=0.1)



# class CenterLoss(nn.Module):
#     """Center loss.
    
#     Args:
#         num_classes (int): number of classes.
#         feat_dim (int): feature dimension.
#         device (torch.device): the device where the parameters are stored.
#     """
#     def __init__(self, num_classes, feat_dim, device):
#         super(CenterLoss, self).__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.device = device

#         # Initialize the centers
#         self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

#     def forward(self, features, labels):
#         """
#         Args:
#             features(tensor): feature matrix with shape (batch_size, feat_dim).
#             labels(tensor): ground truth labels with shape (num_classes).
#         """
#         centers_batch = self.centers.index_select(0, labels)
#         loss = F.mse_loss(features, centers_batch)

#         return loss





class CrossEntropyLoss(nn.Module):
    """Cross entropy loss.

    """
    def __init__(self, use_gpu=True):
        super(CrossEntropyLoss, self).__init__()
        self.use_gpu = use_gpu
        self.crossentropy_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        if self.use_gpu: targets = targets.cuda()
        loss = self.crossentropy_loss(inputs, targets)
        return loss

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class TripletHardLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletHardLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        dist.addmm_(mat1=inputs, mat2=inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss

class TripletLossAlignedReID(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletLossAlignedReID, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.ranking_loss_local = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets, local_features):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        #inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        dist_ap,dist_an,p_inds,n_inds = hard_example_mining(dist,targets,return_inds=True)
        local_features = local_features.permute(0,2,1)
        p_local_features = local_features[p_inds]
        n_local_features = local_features[n_inds]
        local_dist_ap = batch_local_dist(local_features, p_local_features)
        local_dist_an = batch_local_dist(local_features, n_local_features)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        global_loss = self.ranking_loss(dist_an, dist_ap, y)
        local_loss = self.ranking_loss_local(local_dist_an,local_dist_ap, y)
        if self.mutual:
            return global_loss+local_loss,dist
        return global_loss,local_loss

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss
    

class ReIdTotalLoss(nn.Module):
    def __init__(self, cfg):
        super(ReIdTotalLoss, self).__init__()
        self.config = cfg
        self.loss_cfg = cfg.train.loss
        self.losses = nn.ModuleDict()
        self.weights = dict()


        if self.loss_cfg.metric.enabled:
            loss_func, weight, margin = self.loss_cfg.metric.loss_func, self.loss_cfg.metric.weight,  self.loss_cfg.metric.margin
            if loss_func == 'triplet-hard':
                self.losses['metric_loss'] = TripletHardLoss(margin=margin)
            self.weights['metric_loss'] = weight

        if self.loss_cfg.id.enabled:
            num_classes = self.config.train.num_classes
            loss_func, weight= self.loss_cfg.id.loss_func, self.loss_cfg.id.weight
            if loss_func == 'cross-entropy-label-smooth':
                epsilon = self.loss_cfg.id.epsilon
                self.losses['id_loss'] = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=epsilon)
            self.weights['id_loss'] = weight

        if self.loss_cfg.center.enabled:
            num_classes = self.config.train.num_classes
            feat_dim = self.config.model._global.feature_dim
            weight = self.loss_cfg.center.weight
            self.losses['center_loss'] = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)  # num_classes, feat_dim可配置
            self.weights['center_loss'] = weight
    def forward(self,predicted_pids, global_features, local_fuetures , pids, **kwargs):
        total_loss = 0.0
        loss_dict = {}

        for name, loss_func in self.losses.items():

            if isinstance(loss_func,CrossEntropyLoss) or isinstance(loss_func,CrossEntropyLabelSmooth):
                loss_value = loss_func(predicted_pids,pids)
            elif isinstance(loss_func, TripletHardLoss):
                loss_value = loss_func(global_features,pids)
            elif isinstance(loss_func,CenterLoss):
                loss_value = loss_func(global_features,pids)
            else:
                raise ValueError(f"Unknown loss function type for {name}")
            # embed()
            weight = self.weights[name]
            total_loss += weight * loss_value
            loss_dict[name] = loss_value.item() * weight
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
