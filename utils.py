
import os
import os
from PIL import Image
import glob
import torch
import torchvision.utils as vutils
from matplotlib import pyplot as plt
import time


def read_image(img_path):
    if not os.path.exists(img_path):
        raise IOError(f"{img_path} does not exist.")
    img = Image.open(img_path).convert('RGB')
    return img

def read_sample_images(dir_path,transform,device):
    img_paths = glob.glob(os.path.join(dir_path,"*.jpg")) # Return a list of paths matching the pattern '*jpg'
    imgs = []
    for img_path in img_paths:
        img = read_image(img_path)
        if transform:
            img = transform(img) 
        imgs.append(img)
    
    img_tensor = torch.stack(imgs).to(device)
    return img_tensor


class UnNormalize:
    #restore from T.Normalize
    #反归一化
    def __init__(self,mean=(0.485, 0.456, 0.406),std= (0.229, 0.224, 0.225)):
        self.mean=torch.tensor(mean,device='cuda:0').view((1,-1,1,1))
        self.std=torch.tensor(std,device='cuda:0').view((1,-1,1,1))
    def __call__(self,x):
        x=(x*self.std)+self.mean
        return torch.clip(x,0,None)



def save_combined_features_to_tensorboard(x, feature_map, writer, tag, unnorm=None):
    """
    将原始图像和所有特征图通道合并为一张大图，在垂直方向上拼接，并输出到TensorBoard。
    :param x: 原始图像，维度为(bs, 3, h, w)
    :param feature_map: 特征图，维度为(bs, c, h', w')
    :param writer: TensorBoard的writer对象
    :param tag: 在TensorBoard中显示的图像的标签
    :param unnorm: 反归一化函数，如果提供，将被用于原始图像
    """
    if unnorm:
        x = unnorm(x)
    x = torch.nn.functional.interpolate(x, size=feature_map.shape[2:], mode='bilinear', align_corners=False)

    # 将所有样本的原始图像和特征图排列成bs行(1+c)列的布局
    rows = []
    for i in range(x.size(0)):  # 遍历每个样本
        row_imgs = [x[i]]  # 开始每行的图片列表，首先加入原始图像
        for j in range(feature_map.size(1)):  # 遍历每个特征图
            fmap = feature_map[i, j].unsqueeze(0)  # 将特征图扩展到3D
            fmap3 = torch.cat([fmap, fmap, fmap], dim=0)  # 转换为三通道
            row_imgs.append(fmap3)
        rows.append(torch.cat(row_imgs, dim=2))  # 将一行中的所有图片横向拼接

    # 将所有行纵向拼接
    full_grid = torch.cat(rows, dim=1)
    # 使用make_grid调整布局，此时因为已经手动拼接，所以只需要一个大的grid即可
    grid = vutils.make_grid(full_grid.unsqueeze(0), scale_each=True, pad_value=1)
    
    # 输出到TensorBoard
    writer.add_image(tag, grid, global_step=0)



def save_combined_features(x, feature_map, base_dir='debug/feature_map'):
    """
    将原始图像和所有特征图通道合并为一张图并保存。
    :param x: 原始图像，维度为(bs, 3, h, w)
    :param feature_map: 特征图，维度为(bs, c, h', w')
    :param base_dir: 保存图像的基本目录
    """
    timestamp = str(int(time.time()))  # 获取当前时间戳
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)  # 创建保存目录
    unnorm = UnNormalize()
    x = unnorm(x)
    batch_size, _, h, w = x.shape
    _, feature_channels, hf, wf = feature_map.shape
    x = torch.nn.functional.interpolate(x, size=(hf, wf), mode='bilinear', align_corners=False)

    for i in range(batch_size):
        fig, axs = plt.subplots(1, feature_channels + 1, figsize=(2 * (feature_channels + 1), 2))  # 根据通道数调整图像大小

        # 显示原图
        axs[0].imshow(x[i].permute(1, 2, 0).cpu().numpy())
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        # 显示每个特征图
        for j in range(feature_channels):
            # embed()
            axs[j + 1].imshow(feature_map[i, j].cpu().detach().numpy(), cmap='viridis')
            axs[j + 1].set_title(f'Feature {j + 1}')
            axs[j + 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{i}.png'), bbox_inches='tight')
        plt.close()


def cosine_dist(x, y):
    """
    Compute the cosine distance between two sets of vectors.
    
    Args:
      x: PyTorch Tensor, shape [m, d]
      y: PyTorch Tensor, shape [n, d]
      
    Returns:
      dist: PyTorch Tensor, shape [m, n] where dist[i][j] is the cosine distance between x[i] and y[j]
    """
    # Normalize each vector to have unit norm
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=1)
    
    # Compute the cosine similarity
    cosine_similarity = torch.mm(x_norm, y_norm.t())
    
    # Convert cosine similarity to cosine distance
    dist = 1 - cosine_similarity
    return dist

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist