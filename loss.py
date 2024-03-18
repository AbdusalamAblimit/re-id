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



def hard_example_mining(dist_mat, labels, return_inds=False):
  """For each anchor, find the hardest positive and negative sample.
  Args:
    dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
    labels: pytorch LongTensor, with shape [N]
    return_inds: whether to return the indices. Save time if `False`(?)
  Returns:
    dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
    dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    p_inds: pytorch LongTensor, with shape [N];
      indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
    n_inds: pytorch LongTensor, with shape [N];
      indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
  NOTE: Only consider the case in which all labels have same num of samples,
    thus we can cope with all anchors in parallel.
  """

  assert len(dist_mat.size()) == 2
  assert dist_mat.size(0) == dist_mat.size(1)
  N = dist_mat.size(0)

  # shape [N, N]
  is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
  is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

  # `dist_ap` means distance(anchor, positive)
  # both `dist_ap` and `relative_p_inds` with shape [N, 1]
  dist_ap, relative_p_inds = torch.max(
    dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
  # `dist_an` means distance(anchor, negative)
  # both `dist_an` and `relative_n_inds` with shape [N, 1]
  dist_an, relative_n_inds = torch.min(
    dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
  # shape [N]
  dist_ap = dist_ap.squeeze(1)
  dist_an = dist_an.squeeze(1)

  if return_inds:
    # shape [N, N]
    ind = (labels.new().resize_as_(labels)
           .copy_(torch.arange(0, N).long())
           .unsqueeze( 0).expand(N, N))
    # shape [N, 1]
    p_inds = torch.gather(
      ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
    n_inds = torch.gather(
      ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
    # shape [N]
    p_inds = p_inds.squeeze(1)
    n_inds = n_inds.squeeze(1)
    return dist_ap, dist_an, p_inds, n_inds

  return dist_ap, dist_an



def shortest_dist(dist_mat):
  """Parallel version.
  Args:
    dist_mat: pytorch Variable, available shape:
      1) [m, n]
      2) [m, n, N], N is batch size
      3) [m, n, *], * can be arbitrary additional dimensions
  Returns:
    dist: three cases corresponding to `dist_mat`:
      1) scalar
      2) pytorch Variable, with shape [N]
      3) pytorch Variable, with shape [*]
  """
  m, n = dist_mat.size()[:2]
  # Just offering some reference for accessing intermediate distance.
  dist = [[0 for _ in range(n)] for _ in range(m)]
  for i in range(m):
    for j in range(n):
      if (i == 0) and (j == 0):
        dist[i][j] = dist_mat[i, j]
      elif (i == 0) and (j > 0):
        dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
      elif (i > 0) and (j == 0):
        dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
      else:
        dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
  dist = dist[-1][-1]
  return dist

def batch_euclidean_dist(x, y):
  """
  Args:
    x: pytorch Variable, with shape [Batch size, Local part, Feature channel]
    y: pytorch Variable, with shape [Batch size, Local part, Feature channel]
  Returns:
    dist: pytorch Variable, with shape [Batch size, Local part, Local part]
  """
  assert len(x.size()) == 3
  assert len(y.size()) == 3
  assert x.size(0) == y.size(0)
  assert x.size(-1) == y.size(-1)

  N, m, d = x.size()
  N, n, d = y.size()

  # shape [N, m, n]
  xx = torch.pow(x, 2).sum(-1, keepdim=True).expand(N, m, n)
  yy = torch.pow(y, 2).sum(-1, keepdim=True).expand(N, n, m).permute(0, 2, 1)
  dist = xx + yy
  dist.baddbmm_(1, -2, x, y.permute(0, 2, 1))
  dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
  return dist

def batch_local_dist(x, y):
  """
  Args:
    x: pytorch Variable, with shape [N, m, d]
    y: pytorch Variable, with shape [N, n, d]
  Returns:
    dist: pytorch Variable, with shape [N]
  """
  assert len(x.size()) == 3
  assert len(y.size()) == 3
  assert x.size(0) == y.size(0)
  assert x.size(-1) == y.size(-1)

  # shape [N, m, n]
  dist_mat = batch_euclidean_dist(x, y)
  dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
  # shape [N]
  dist = shortest_dist(dist_mat.permute(1, 2, 0))
  return dist

class TripletHardLossAlignedReID(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletHardLossAlignedReID, self).__init__()
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
         # Apply the same hard examples found by global features to local features
        
        local_features_positive = local_features[p_inds]
        local_features_negative = local_features[n_inds]
        
        # Compute distances for local features
        local_dist_ap = torch.norm(local_features - local_features_positive, p=2, dim=1)
        local_dist_an = torch.norm(local_features - local_features_negative, p=2, dim=1)

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
        distmat.addmm_(x, self.centers.t(),beta=1, alpha= -2)

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
            self.losses['center_loss'] = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
            self.weights['center_loss'] = weight

        if self.config.model.aligned:
            loss_func, weight, margin = self.loss_cfg.metric.loss_func, self.loss_cfg.metric.weight,  self.loss_cfg.metric.margin
            if loss_func != 'triplet-hard-aligned-reid':
                raise 'Triplet-hard loss is not available when training aligned.Only triplet-hard-aligned-reid can be used as matric loss.'
            self.losses['triplet-hard-aligned-reid'] = TripletHardLossAlignedReID(margin,mutual_flag=False)
            self.weights['triplet-hard-aligned-reid'] = weight


        
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
            elif isinstance(loss_func, TripletHardLossAlignedReID):
                global_loss, local_loss = loss_func(global_features, pids, local_fuetures)
                loss_value = global_loss + local_loss
            else:
                raise ValueError(f"Unknown loss function type for {name}")
            # embed()
            weight = self.weights[name]
            total_loss += weight * loss_value
            if name != 'triplet-hard-aligned-reid':
                loss_dict[name] = loss_value.item() * weight
            else:
                loss_dict['global_loss'] = global_loss.item() * weight
                loss_dict['local_loss'] = local_loss.item() * weight

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
