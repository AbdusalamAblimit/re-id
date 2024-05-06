import torch
from torch import nn
from torch.nn import functional as F
from IPython import embed
from loguru import logger
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




class ArcFaceLoss(nn.Module):
    """Implementation of ArcFace loss which enhances the feature discrimination based on angular distance."""

    def __init__(self, in_features, out_features, s=64.0, m=0.1, use_gpu=True):
        """
        Args:
            in_features (int): Dimensionality of the input features (size of the last layer of the model).
            out_features (int): Number of classes.
            s (float): Norm of the input feature vector.
            m (float): Margin to enhance inter-class separability.
            use_gpu (bool): Whether to use GPU.
        """
        super(ArcFaceLoss, self).__init__()
        self.use_gpu = use_gpu
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        if self.use_gpu:
          self.weight = nn.Parameter(self.weight.cuda())  # Correct way to move a parameter to GPU


    def forward(self, inputs, labels):
        """
        Args:
            inputs: Input features, shape (batch_size, in_features).
            labels: Target labels, shape (batch_size).
        """
        # Normalize weight and inputs
        cos_theta = F.linear(F.normalize(inputs), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)  # Numerical stability
        theta = torch.acos(cos_theta)
        # Add margin
        one_hot = torch.zeros(cos_theta.size(), device='cuda' if self.use_gpu else 'cpu')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        theta_m = theta + one_hot * self.m
        # Convert back to cos after adding margin
        cos_theta_m = torch.cos(theta_m)
        # Scale the logits
        logits = self.s * cos_theta_m
        # Use cross entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss



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

# class TripletHardLoss(nn.Module):
#     """Tripl`et loss with hard positive/negative mining based on cosine distance.

#     Reference:
#     Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

#     This version uses cosine distance instead of Euclidean distance.
#     """
#     def __init__(self, margin=0.3, mutual_flag=False):
#         super(TripletHardLoss, self).__init__()
#         self.margin = margin
#         self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#         self.mutual = mutual_flag

#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs: feature matrix with shape (batch_size, feat_dim), assumed to be normalized.
#             targets: ground truth labels with shape (num_classes)
#         """
#         n = inputs.size(0)
#         # Normalize the input features to unit length
#         inputs = F.normalize(inputs, p=2, dim=1)
#         # Compute pairwise cosine similarity
#         cos_sim = torch.mm(inputs, inputs.t())
#         # Convert cosine similarity to cosine distance
#         dist = 1 - cos_sim

#         # For each anchor, find the hardest positive and negative based on cosine distance
#         mask = targets.expand(n, n).eq(targets.expand(n, n).t())
#         dist_ap, dist_an = [], []
#         for i in range(n):
#             dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # Hardest positive: max cosine distance
#             dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # Hardest negative: min cosine distance
#         dist_ap = torch.cat(dist_ap)
#         dist_an = torch.cat(dist_an)

#         # Compute ranking hinge loss
#         y = torch.ones_like(dist_an)
#         loss = self.ranking_loss(dist_an, dist_ap, y)
#         if self.mutual:
#             return loss, dist
#         return loss


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x




class TripletHardLoss(nn.Module):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None, metric='euclidean'):
        super(TripletHardLoss, self).__init__()
        self.margin = margin
        self.metric = metric
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def cosine_dist(self, x, y):
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        cosine_similarity = torch.mm(x_norm, y_norm.t())
        dist = 1 - cosine_similarity
        return dist

    def forward(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = F.normalize(global_feat, p=2, dim=1)
        if self.metric == 'euclidean':
            dist_mat = self.euclidean_dist(global_feat, global_feat)
        elif self.metric == 'cosine':
            dist_mat = self.cosine_dist(global_feat, global_feat)
        else:
            raise ValueError("Unsupported metric: {}".format(self.metric))

        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
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





class TripletHardLossAlignedReID(nn.Module):
    def __init__(self, margin=0.3, mutual_flag=False,metric='euclidean'):
        super(TripletHardLossAlignedReID, self).__init__()
        self.margin = margin
        self.metric = metric
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.ranking_loss_local = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def cosine_dist(self, x, y):
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        cos_sim = torch.mm(x_norm, y_norm.t())
        dist = 1 - cos_sim
        return dist

    def forward(self, inputs, targets, local_features):
        n = inputs.size(0)

        # Normalize global and local features
        # inputs = F.normalize(inputs, p=2, dim=1)
        # local_features = F.normalize(local_features, p=2, dim=1)

        # Select the distance metric
        if self.metric == 'euclidean':
            dist = self.euclidean_dist(inputs, inputs)
        elif self.metric == 'cosine':
            dist = self.cosine_dist(inputs, inputs)
        else:
            raise ValueError("Unsupported metric: {}".format(self.metric))

        # For each anchor, find the hardest positive and negative
        dist_ap, dist_an, p_inds, n_inds = hard_example_mining(dist, targets, return_inds=True)

        # Apply the same hard examples found by global features to local features

        local_features_positive = local_features[p_inds]
        local_features_negative = local_features[n_inds]

        # Compute distances for local features using the same metric
        if self.metric == 'euclidean':
            local_dist_ap = self.euclidean_dist(local_features, local_features_positive).diag()
            local_dist_an = self.euclidean_dist(local_features, local_features_negative).diag()
        elif self.metric == 'cosine':
            local_dist_ap = self.cosine_dist(local_features, local_features_positive).diag()
            local_dist_an = self.cosine_dist(local_features, local_features_negative).diag()
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        global_loss = self.ranking_loss(dist_an, dist_ap, y)
        local_loss = self.ranking_loss_local(local_dist_an, local_dist_ap, y)
        
        if self.mutual:
            return global_loss + local_loss, dist
        return global_loss, local_loss


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
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
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        #dist = []
        #for i in range(batch_size):
        #    value = distmat[i][mask[i]]
        #    value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
        #    dist.append(value)
        #dist = torch.cat(dist)
        #loss = dist.mean()
        return loss
    

class ReIdTotalLoss(nn.Module):
    def __init__(self, cfg):
        super(ReIdTotalLoss, self).__init__()
        self.config = cfg
        self.loss_cfg = cfg.train.loss
        self.losses = nn.ModuleDict()
        self.weights = dict()
        self.metric = cfg.model.metric

        if self.loss_cfg.metric.enabled:
            loss_func, weight, margin = self.loss_cfg.metric.loss_func, self.loss_cfg.metric.weight,  self.loss_cfg.metric.margin
            if loss_func == 'triplet-hard':
                if self.config.model.aligned:
                    self.losses['global_loss'] = TripletHardLoss(margin=margin,metric=self.metric)
                    self.losses['local_loss'] = TripletHardLoss(margin=margin,metric=self.metric)
                    self.weights['global_loss'] = weight
                    self.weights['local_loss'] = weight
                else:
                    self.losses['metric_loss'] = TripletHardLoss(margin=margin,metric=self.metric)
                    self.weights['metric_loss'] = weight
            elif loss_func == 'triplet-hard-aligned-reid':
                if not self.config.model.aligned:
                    raise 'Triplet-hard loss is not available when training aligned.Only triplet-hard-aligned-reid can be used as matric loss.'
                self.losses['triplet-hard-aligned-reid'] = TripletHardLossAlignedReID(margin,mutual_flag=False,metric=self.metric)
                self.weights['triplet-hard-aligned-reid'] = weight
        

        if self.loss_cfg.id.enabled:
            num_classes = self.config.train.num_classes
            loss_func, weight= self.loss_cfg.id.loss_func, self.loss_cfg.id.weight
            if loss_func == 'cross-entropy-label-smooth':
                epsilon = self.loss_cfg.id.epsilon
                self.losses['global_id_loss'] = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=epsilon)
                if self.config.model.aligned:
                    self.losses['local_id_loss'] = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=epsilon)
            elif loss_func == 'arcface-loss':
                self.losses['id_loss'] = ArcFaceLoss(self.config.model._global.feature_dim, self.config.train.num_classes)
            self.weights['global_id_loss'] = weight
            if self.config.model.aligned:
                self.weights['local_id_loss'] = weight

        if self.loss_cfg.center.enabled:
            num_classes = self.config.train.num_classes
            feat_dim = self.config.model._global.feature_dim
            weight = self.loss_cfg.center.weight
            self.losses['center_loss'] = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
            self.weights['center_loss'] = weight

            


        
    def forward(self,global_predicted_pids,local_predicted_pids, global_features, local_features , pids, **kwargs):
        total_loss = 0.0
        loss_dict = {}

        for name, loss_func in self.losses.items():

            if isinstance(loss_func,CrossEntropyLoss) or isinstance(loss_func,CrossEntropyLabelSmooth):
                if name=='global_id_loss':
                  loss_value = loss_func(global_predicted_pids,pids)
                elif name =='local_id_loss':
                  loss_value = loss_func(local_predicted_pids,pids)
            elif isinstance(loss_func,ArcFaceLoss):
                loss_value = loss_func(global_features,pids)
            elif isinstance(loss_func, TripletHardLoss):
                if self.config.model.aligned:
                    if name == 'global_loss':
                        loss_value = loss_func(global_features,pids)
                    elif name == "local_loss":
                        loss_value = loss_func(local_features,pids)
                    else:
                        raise "!!!!!!!!!!!!!!!!!!"
                else:
                    loss_value = loss_func(global_features,pids)
            elif isinstance(loss_func,CenterLoss):
                loss_value = loss_func(global_features,pids)
            elif isinstance(loss_func, TripletHardLossAlignedReID):
                global_loss, local_loss = loss_func(global_features, pids, local_features)
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
