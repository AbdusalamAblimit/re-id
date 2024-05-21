
from torchvision.utils import save_image
import matplotlib

import torch
import torchvision.utils as vutils



import torch
import torchvision.utils as vutils





def save_feature_maps(x, feature_map,path, unnorm=None):
    """
    将原始图像和所有特征图通道合并为一张大图，在垂直方向上拼接，并保存到path
    :param x: 原始图像，维度为(bs, 3, h, w)
    :param feature_map: 特征图，维度为(bs, c, h', w')
    :param unnorm: 反归一化函数，如果提供，将被用于原始图像
    """
    if unnorm:
        x = unnorm(x)
    x = torch.nn.functional.interpolate(x, size=feature_map.shape[2:], mode='bilinear', align_corners=False)

    # 使用彩色映射
    cmap = matplotlib.colormaps['jet']  # 选择热力图颜色映射，例如 'jet', 'viridis', 'hot' 等
    rows = []
    for i in range(x.size(0)):
        row_imgs = [x[i].cpu()]  # 加入原始图像
        for j in range(feature_map.size(1)):
            fmap = feature_map[i, j].unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
            fmap = torch.nn.functional.interpolate(fmap, size=x.shape[2:], mode='bilinear', align_corners=False)
            fmap_normalized = (fmap - fmap.min()) / (fmap.max() - fmap.min())  # 归一化
            fmap_colored = cmap(fmap_normalized.squeeze().cpu().detach().numpy())  # 应用颜色映射
            fmap_colored = torch.from_numpy(fmap_colored[:, :, :3]).permute(2, 0, 1)  # 转换为torch tensor并取前三个通道
            row_imgs.append(fmap_colored)
        rows.append(torch.cat(row_imgs, dim=2))

    full_grid = torch.cat(rows, dim=1)
    grid = vutils.make_grid(full_grid.unsqueeze(0), scale_each=True, pad_value=1)
    save_image(grid, path)


if __name__ =='__main__':

    class UnNormalize:
        #restore from T.Normalize
        #反标准化
        def __init__(self,mean=(0.485, 0.456, 0.406),std= (0.229, 0.224, 0.225)):
            self.mean=torch.tensor(mean,device='cuda:0').view((1,-1,1,1))
            self.std=torch.tensor(std,device='cuda:0').view((1,-1,1,1))
        def __call__(self,x):
            x=(x*self.std)+self.mean
            return torch.clip(x,0,None)
    

    imgs = torch.rand(32,3,256,128).cuda()     # 假设这是经过了一个batch的输入，也就是整个网络的输入，经过了标准化了的图片，尺寸是[batch_size,c,h,w]
    feature_maps = torch.rand([32,10,64,32]).cuda()   # 假设这是中间某个层输出的特征图,，尺寸是[batch_size,c,h,w]
    unnorm = UnNormalize()  # 反标准化函数，用于将图片反标准化回来，恢复原来的色彩
    save_feature_maps(imgs,feature_maps,'test_saving.jpg')  