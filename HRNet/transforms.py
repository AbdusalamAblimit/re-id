import math
import random
from typing import Tuple
import os
import shutil
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from IPython import embed

def flip_images(img):
    assert len(img.shape) == 4, 'images has to be [batch_size, channels, height, width]'
    img = torch.flip(img, dims=[3])
    return img


def flip_back(output_flipped, matched_parts):
    assert len(output_flipped.shape) == 4, 'output_flipped has to be [batch_size, num_joints, height, width]'
    output_flipped = torch.flip(output_flipped, dims=[3])

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0]].clone()
        output_flipped[:, pair[0]] = output_flipped[:, pair[1]]
        output_flipped[:, pair[1]] = tmp

    return output_flipped


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be torch.Tensor'
    assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim'

    batch_size, num_joints, h, w = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

    maxvals = maxvals.unsqueeze(dim=-1)
    idx = idx.float()

    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps)

    preds[:, :, 0] = idx % w  # column 对应最大值的x坐标
    preds[:, :, 1] = torch.floor(idx / w)  # row 对应最大值的y坐标

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)

    preds *= pred_mask
    return preds, maxvals


def affine_points(pt, t):
    ones = np.ones((pt.shape[0], 1), dtype=float)
    pt = np.concatenate([pt, ones], axis=1).T
    new_pt = np.dot(t, pt)
    return new_pt.T


def get_final_preds(batch_heatmaps: torch.Tensor,
                    trans: list = None,
                    post_processing: bool = False):
    assert trans is not None
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if post_processing:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = torch.tensor(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    ).to(batch_heatmaps.device)
                    coords[n][p] += torch.sign(diff) * .25

    preds = coords.clone().cpu().numpy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = affine_points(preds[i], trans[i])

    return preds, maxvals.cpu().numpy()


def decode_keypoints(outputs, origin_hw, num_joints: int = 17):
    keypoints = []
    scores = []
    heatmap_h, heatmap_w = outputs.shape[-2:]
    for i in range(num_joints):
        pt = np.unravel_index(np.argmax(outputs[i]), (heatmap_h, heatmap_w))
        score = outputs[i, pt[0], pt[1]]
        keypoints.append(pt[::-1])  # hw -> wh(xy)
        scores.append(score)

    keypoints = np.array(keypoints, dtype=float)
    scores = np.array(scores, dtype=float)
    # convert to full image scale
    keypoints[:, 0] = np.clip(keypoints[:, 0] / heatmap_w * origin_hw[1],
                              a_min=0,
                              a_max=origin_hw[1])
    keypoints[:, 1] = np.clip(keypoints[:, 1] / heatmap_h * origin_hw[0],
                              a_min=0,
                              a_max=origin_hw[0])
    return keypoints, scores


def resize_pad(img: np.ndarray, size: tuple):
    h, w, c = img.shape
    src = np.array([[0, 0],       # 原坐标系中图像左上角点
                    [w - 1, 0],   # 原坐标系中图像右上角点
                    [0, h - 1]],  # 原坐标系中图像左下角点
                   dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    if h / w > size[0] / size[1]:
        # 需要在w方向padding
        wi = size[0] * (w / h)
        pad_w = (size[1] - wi) / 2
        dst[0, :] = [pad_w - 1, 0]            # 目标坐标系中图像左上角点
        dst[1, :] = [size[1] - pad_w - 1, 0]  # 目标坐标系中图像右上角点
        dst[2, :] = [pad_w - 1, size[0] - 1]  # 目标坐标系中图像左下角点
    else:
        # 需要在h方向padding
        hi = size[1] * (h / w)
        pad_h = (size[0] - hi) / 2
        dst[0, :] = [0, pad_h - 1]            # 目标坐标系中图像左上角点
        dst[1, :] = [size[1] - 1, pad_h - 1]  # 目标坐标系中图像右上角点
        dst[2, :] = [0, size[0] - pad_h - 1]  # 目标坐标系中图像左下角点

    trans = cv2.getAffineTransform(src, dst)  # 计算正向仿射变换矩阵
    # 对图像进行仿射变换
    resize_img = cv2.warpAffine(img,
                                trans,
                                size[::-1],  # w, h
                                flags=cv2.INTER_LINEAR)
    # import matplotlib.pyplot as plt
    # plt.imshow(resize_img)
    # plt.show()

    dst /= 4  # 网络预测的heatmap尺寸是输入图像的1/4
    reverse_trans = cv2.getAffineTransform(dst, src)  # 计算逆向仿射变换矩阵，方便后续还原

    return resize_img, reverse_trans


def adjust_box(xmin: float, ymin: float, w: float, h: float, fixed_size: Tuple[float, float]):
    """通过增加w或者h的方式保证输入图片的长宽比固定"""
    xmax = xmin + w
    ymax = ymin + h

    hw_ratio = fixed_size[0] / fixed_size[1]
    if h / w > hw_ratio:
        # 需要在w方向padding
        wi = h / hw_ratio
        pad_w = (wi - w) / 2
        xmin = xmin - pad_w
        xmax = xmax + pad_w
    else:
        # 需要在h方向padding
        hi = w * hw_ratio
        pad_h = (hi - h) / 2
        ymin = ymin - pad_h
        ymax = ymax + pad_h

    return xmin, ymin, xmax, ymax


def scale_box(xmin: float, ymin: float, w: float, h: float, scale_ratio: Tuple[float, float]):
    """根据传入的h、w缩放因子scale_ratio，重新计算xmin，ymin，w，h"""
    s_h = h * scale_ratio[0]
    s_w = w * scale_ratio[1]
    xmin = xmin - (s_w - w) / 2.
    ymin = ymin - (s_h - h) / 2.
    return xmin, ymin, s_w, s_h


def plot_heatmap(image, heatmap, kps, kps_weights):
    for kp_id in range(len(kps_weights)):
        if kps_weights[kp_id] > 0:
            plt.figure()  # 创建一个新的图形
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.plot(*kps[kp_id].tolist(), "ro")
            plt.title("image")
            
            plt.subplot(1, 2, 2)
            plt.imshow(heatmap[kp_id], cmap=plt.cm.Blues)
            plt.colorbar(ticks=[0, 1])
            plt.title(f"kp_id: {kp_id}")
            
            plt.savefig(f'./heatmaps/heatmap{kp_id}.png')

def clear_directory(directory_path):

    # 检查目录是否存在
    if os.path.exists(directory_path):
        # 删除目录下的所有文件
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def plot_heatmap_combined(image, heatmap, kps, kps_weights):
     # 清空目录
    clear_directory('./heatmaps')
    for kp_id in range(len(kps_weights)):
        if kps_weights[kp_id] > 0:
            plt.figure(figsize=(15, 5))  # 创建一个新的图形，设置更大的尺寸以便清晰显示

            # 第一张图显示原图
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 假设原图是BGR格式
            plt.title("Original Image")
            plt.axis('off')  # 不显示坐标轴

            # 第二张图显示热力图
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap[kp_id], cmap='jet', interpolation='nearest')
            plt.title(f"Heatmap: kp_id {kp_id}")
            plt.colorbar(ticks=[0, 1])
            plt.axis('off')

            # 第三张图显示原图与热力图混合
            plt.subplot(1, 3, 3)
            # 先将热力图转换为彩色图
            heatmap_resized = cv2.resize(heatmap[kp_id], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            # 混合原图和热力图
            overlayed_img = cv2.addWeighted(image, 1, heatmap_color, 0.5, 0)
            plt.imshow(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB))  # 假设原图是BGR格式
            plt.title("Overlayed Image")
            plt.axis('off')

            plt.savefig(f'./heatmaps/overlayed_heatmap{kp_id}.png')
            plt.close()  # 关闭图形，避免在notebook中显示

class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class HalfBody(object):
    def __init__(self, p: float = 0.3, upper_body_ids=None, lower_body_ids=None):
        assert upper_body_ids is not None
        assert lower_body_ids is not None
        self.p = p
        self.upper_body_ids = upper_body_ids
        self.lower_body_ids = lower_body_ids

    def __call__(self, image, target):
        if random.random() < self.p:
            kps = target["keypoints"]
            vis = target["visible"]
            upper_kps = []
            lower_kps = []

            # 对可见的keypoints进行归类
            for i, v in enumerate(vis):
                if v > 0.5:
                    if i in self.upper_body_ids:
                        upper_kps.append(kps[i])
                    else:
                        lower_kps.append(kps[i])

            # 50%的概率选择上或下半身
            if random.random() < 0.5:
                selected_kps = upper_kps
            else:
                selected_kps = lower_kps

            # 如果点数太少就不做任何处理
            if len(selected_kps) > 2:
                selected_kps = np.array(selected_kps, dtype=np.float32)
                xmin, ymin = np.min(selected_kps, axis=0).tolist()
                xmax, ymax = np.max(selected_kps, axis=0).tolist()
                w = xmax - xmin
                h = ymax - ymin
                if w > 1 and h > 1:
                    # 把w和h适当放大点，要不然关键点处于边缘位置
                    xmin, ymin, w, h = scale_box(xmin, ymin, w, h, (1.5, 1.5))
                    target["box"] = [xmin, ymin, w, h]

        return image, target


class AffineTransform(object):
    """scale+rotation"""
    def __init__(self,
                 scale: Tuple[float, float] = None,  # e.g. (0.65, 1.35)
                 rotation: Tuple[int, int] = None,   # e.g. (-45, 45)
                 fixed_size: Tuple[int, int] = (256, 192)):
        self.scale = scale
        self.rotation = rotation
        self.fixed_size = fixed_size

    def __call__(self, img, target):
        src_xmin, src_ymin, src_xmax, src_ymax = adjust_box(*target["box"], fixed_size=self.fixed_size)
        src_w = src_xmax - src_xmin
        src_h = src_ymax - src_ymin
        src_center = np.array([(src_xmin + src_xmax) / 2, (src_ymin + src_ymax) / 2])
        src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
        src_p3 = src_center + np.array([src_w / 2, 0])   # right middle

        dst_center = np.array([(self.fixed_size[1] - 1) / 2, (self.fixed_size[0] - 1) / 2])
        dst_p2 = np.array([(self.fixed_size[1] - 1) / 2, 0])  # top middle
        dst_p3 = np.array([self.fixed_size[1] - 1, (self.fixed_size[0] - 1) / 2])  # right middle

        if self.scale is not None:
            scale = random.uniform(*self.scale)
            src_w = src_w * scale
            src_h = src_h * scale
            src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
            src_p3 = src_center + np.array([src_w / 2, 0])   # right middle

        if self.rotation is not None:
            angle = random.randint(*self.rotation)  # 角度制
            angle = angle / 180 * math.pi  # 弧度制
            src_p2 = src_center + np.array([src_h / 2 * math.sin(angle), -src_h / 2 * math.cos(angle)])
            src_p3 = src_center + np.array([src_w / 2 * math.cos(angle), src_w / 2 * math.sin(angle)])

        src = np.stack([src_center, src_p2, src_p3]).astype(np.float32)
        dst = np.stack([dst_center, dst_p2, dst_p3]).astype(np.float32)

        trans = cv2.getAffineTransform(src, dst)  # 计算正向仿射变换矩阵
        dst /= 4  # 网络预测的heatmap尺寸是输入图像的1/4
        reverse_trans = cv2.getAffineTransform(dst, src)  # 计算逆向仿射变换矩阵，方便后续还原

        # 对图像进行仿射变换
        resize_img = cv2.warpAffine(img,
                                    trans,
                                    tuple(self.fixed_size[::-1]),  # [w, h]
                                    flags=cv2.INTER_LINEAR)

        if "keypoints" in target:
            kps = target["keypoints"]
            mask = np.logical_and(kps[:, 0] != 0, kps[:, 1] != 0)
            kps[mask] = affine_points(kps[mask], trans)
            target["keypoints"] = kps

        # import matplotlib.pyplot as plt
        # from draw_utils import draw_keypoints
        # resize_img = draw_keypoints(resize_img, target["keypoints"])
        # plt.imshow(resize_img)
        # plt.show()

        target["trans"] = trans
        target["reverse_trans"] = reverse_trans
        return resize_img, target


class RandomHorizontalFlip(object):
    """随机对输入图片进行水平翻转，注意该方法必须接在 AffineTransform 后"""
    def __init__(self, p: float = 0.5, matched_parts: list = None):
        assert matched_parts is not None
        self.p = p
        self.matched_parts = matched_parts

    def __call__(self, image, target):
        if random.random() < self.p:
            # [h, w, c]
            image = np.ascontiguousarray(np.flip(image, axis=[1]))
            keypoints = target["keypoints"]
            visible = target["visible"]
            width = image.shape[1]

            # Flip horizontal
            keypoints[:, 0] = width - keypoints[:, 0] - 1

            # Change left-right parts
            for pair in self.matched_parts:
                keypoints[pair[0], :], keypoints[pair[1], :] = \
                    keypoints[pair[1], :], keypoints[pair[0], :].copy()

                visible[pair[0]], visible[pair[1]] = \
                    visible[pair[1]], visible[pair[0]].copy()

            target["keypoints"] = keypoints
            target["visible"] = visible

        return image, target





def draw_line_heatmap(heatmap, p1, p2, weight1, weight2, kernel_radius, kernel, heatmap_hw):
    line_length = np.linalg.norm(p2 - p1)
    num_points = int(line_length)
    
    # 创建连线上的点
    for i in range(num_points + 1):
        alpha = i / max(num_points, 1)
        point = (1 - alpha) * p1 + alpha * p2
        weight = (1 - alpha) * weight1 + alpha * weight2
        x, y = point.astype(int)
        ul = [x - kernel_radius, y - kernel_radius]
        br = [x + kernel_radius, y + kernel_radius]
        
        # 如果超出热图范围，则跳过
        if ul[0] >= heatmap_hw[1] or ul[1] >= heatmap_hw[0] or br[0] < 0 or br[1] < 0:
            continue
        
        # 计算高斯核与热图的交叉区域
        g_x = (max(0, -ul[0]), min(br[0], heatmap_hw[1] - 1) - ul[0])
        g_y = (max(0, -ul[1]), min(br[1], heatmap_hw[0] - 1) - ul[1])
        img_x = (max(0, ul[0]), min(br[0], heatmap_hw[1] - 1))
        img_y = (max(0, ul[1]), min(br[1], heatmap_hw[0] - 1))
        # embed()
        # 加权并更新热图
        heatmap[ img_y[0]:img_y[1] + 1, img_x[0]:img_x[1] + 1] += \
            weight * kernel[g_y[0]:g_y[1] + 1, g_x[0]:g_x[1] + 1]
    return heatmap




def merge_heatmaps(original_heatmaps, kps_weights, combined_keypoint_indexes):
    # 假设original_heatmaps的形状为(N, 17, H, W)，N为样本数量，17为关键点通道数，H和W为热力图的高度和宽度
    # kps_weights形状为(17,)
    # combined_keypoint_indexes为要合并的关键点索引列表
    N, _, H, W = original_heatmaps.shape
    num_combined_kps = len(combined_keypoint_indexes)
    combined_heatmaps = np.zeros((N, num_combined_kps, H, W))
    
    for i, indexes in enumerate(combined_keypoint_indexes):
        valid_indexes = [index for index in indexes if index < original_heatmaps.shape[1]]
        # 使用有效的索引和对应的权重计算加权平均热力图
        for index in valid_indexes:
            weight = kps_weights[index]
            combined_heatmaps[:, i, :, :] += original_heatmaps[:, index, :, :] * weight
        # 归一化合并后的热力图
        if  np.sum(kps_weights[valid_indexes]) > 1e-7:
            combined_heatmaps[:, i, :, :] /= np.sum(kps_weights[valid_indexes])
    return combined_heatmaps


class KeypointToHeatMap(object):
    def __init__(self,
                 heatmap_hw: Tuple[int, int] = (256 // 4, 192 // 4),
                 gaussian_sigma: int = 2,
                 keypoints_weights=None,
                 skeleton = None,
                 combine_keypoints=False,
                 combined_keypoint_indexes = None
                 ):
        self.heatmap_hw = heatmap_hw
        self.sigma = gaussian_sigma
        self.kernel_radius = self.sigma * 3
        self.use_kps_weights = False if keypoints_weights is None else True
        self.kps_weights = keypoints_weights
        self.combine_keypoints = combine_keypoints
        self.combined_keypoint_indexes = combined_keypoint_indexes
        # embed()
        for i in range(len(combined_keypoint_indexes)):
            for j in range(len(combined_keypoint_indexes[i])):
                self.combined_keypoint_indexes[i][j]  = self.combined_keypoint_indexes[i][j] - 1

        self.skeleton = None
        if combine_keypoints:
            self.skeleton = []
            self.pair2groups = {}
            self.pair_count = [0]*len(combined_keypoint_indexes)
            for [p1,p2] in skeleton:
                p1-=1
                p2-=1
                is_combined = False
                group_nums = []
                for i,kps in enumerate(self.combined_keypoint_indexes):
                    if p1 in kps and p2 in kps:
                        is_combined = True
                        group_nums.append(i)
                if is_combined:
                    self.skeleton.append([p1,p2])
                    self.pair2groups[(p1,p2)] = group_nums
                    for group_num in group_nums:
                        self.pair_count[group_num] += 1
        # generate gaussian kernel(not normalized)
        kernel_size = 2 * self.kernel_radius + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        x_center = y_center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                kernel[y, x] = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * self.sigma ** 2))
        # print(kernel)

        self.kernel = kernel

    def __call__(self, image, target):
        
        kps = target["keypoints"]
        num_kps = kps.shape[0]
        kps_weights = np.ones((num_kps,), dtype=np.float32)
        if "visible" in target:
            visible = target["visible"]
            kps_weights = visible

        heatmap = np.zeros((num_kps, self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)
        heatmap_kps = (kps / 4 + 0.5).astype(int)  # round
        # if not self.combine_keypoints:
        for kp_id in range(num_kps):
            v = kps_weights[kp_id]
            if v < 0.5:
                # 如果该点的可见度很低，则直接忽略
                continue
            
            x, y = heatmap_kps[kp_id]
            ul = [x - self.kernel_radius, y - self.kernel_radius]  # up-left x,y
            br = [x + self.kernel_radius, y + self.kernel_radius]  # bottom-right x,y
            # 如果以xy为中心kernel_radius为半径的辐射范围内与heatmap没交集，则忽略该点(该规则并不严格)
            if ul[0] > self.heatmap_hw[1] - 1 or \
                    ul[1] > self.heatmap_hw[0] - 1 or \
                    br[0] < 0 or \
                    br[1] < 0:
                # If not, just return the image as is
                kps_weights[kp_id] = 0
                continue

            # Usable gaussian range
            # 计算高斯核有效区域（高斯核坐标系）
            g_x = (max(0, -ul[0]), min(br[0], self.heatmap_hw[1] - 1) - ul[0])
            g_y = (max(0, -ul[1]), min(br[1], self.heatmap_hw[0] - 1) - ul[1])
            # image range
            # 计算heatmap中的有效区域（heatmap坐标系）
            img_x = (max(0, ul[0]), min(br[0], self.heatmap_hw[1] - 1))
            img_y = (max(0, ul[1]), min(br[1], self.heatmap_hw[0] - 1))

            if kps_weights[kp_id] > 0.5:
                # 将高斯核有效区域复制到heatmap对应区域
                heatmap[kp_id][img_y[0]:img_y[1] + 1, img_x[0]:img_x[1] + 1] = \
                    self.kernel[g_y[0]:g_y[1] + 1, g_x[0]:g_x[1] + 1]

        if self.use_kps_weights:
            kps_weights = np.multiply(kps_weights, self.kps_weights)
        if self.combine_keypoints:
            heatmap = merge_heatmaps(heatmap[np.newaxis, :, :, :], kps_weights, self.combined_keypoint_indexes)
            heatmap = np.squeeze(heatmap, axis=0)
            new_kps_weights = np.zeros(len(self.combined_keypoint_indexes), dtype=np.float32)
            for heatmap_id, kps_idxs in enumerate(self.combined_keypoint_indexes):
                new_kps_weights[heatmap_id] = kps_weights[kps_idxs].sum()
            kps_weights = new_kps_weights

        
        # # 如果要求组合关键点，计算组合热力图和权重
        # if self.combine_keypoints and self.combined_keypoint_indexes is not None:
        #     combined_heatmap = np.zeros((len(self.combined_keypoint_indexes), self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)
        #     new_kps_weights = np.zeros(len(self.combined_keypoint_indexes), dtype=np.float32)
        #     for i, indexes in enumerate(self.combined_keypoint_indexes):
        #         # 使用切片而不是直接索引来避免索引超出范围的问题
        #         valid_indexes = np.array(indexes)[np.array(indexes) < len(kps_weights)]
        #         valid_weights = kps_weights[valid_indexes]
                
        #         # 确保有效权重之和大于零
        #         if len(valid_weights) > 0 and np.sum(valid_weights) > 0:
        #             valid_combined = np.take(heatmap, valid_indexes, axis=0)
        #             combined_heatmap[i] = np.average(valid_combined, axis=0, weights=valid_weights)
        #             new_kps_weights[i] = np.mean(valid_weights)
        #         else:
        #             # 如果没有有效的权重，或者有效权重之和为零，则跳过加权平均的计算
        #             combined_heatmap[i] = np.zeros_like(combined_heatmap[i])
        #             new_kps_weights[i] = 0  # 可能需要根据实际情况调整这里的默认权重值
        #         # 归一化(可以尝试注释掉)
        #         # combined_heatmap[i] /= np.max(combined_heatmap[i])  # Normalize to [0, 1]

        #     # 更新heatmap为组合后的heatmap
        #     # old_heatmap = heatmap
        #     # old_kps_weights = kps_weights
        # else:

            

            # combined_heatmap = np.zeros((len(self.combined_keypoint_indexes), self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)
            # new_kps_weights = np.zeros(len(self.combined_keypoint_indexes), dtype=np.float32)
            # for pair_id, (kp_id1, kp_id2) in enumerate(self.skeleton):
            #     # 确保关键点索引在范围内
            #     if kps_weights[kp_id1] == 0 or kps_weights[kp_id2]==0:
            #         continue
            #     pt1 = heatmap_kps[kp_id1]
            #     pt2 = heatmap_kps[kp_id2]
            #     # 在combined_heatmap中为每对关键点绘制直线
            #     for heatmap_id in self.pair2groups[(kp_id1,kp_id2)]:
            #         # embed()
            #         combined_heatmap[heatmap_id] = draw_line_heatmap(combined_heatmap[heatmap_id], pt1, pt2,
            #                                                         kps_weights[kp_id1],kps_weights[kp_id2],self.kernel_radius,self.kernel,
            #                                                         self.heatmap_hw)
            #         new_kps_weights[heatmap_id] += (kps_weights[kp_id1] + kps_weights[kp_id2]) / 2
            # for heatmap_id in range(len(new_kps_weights)):
            #     pair_count = self.pair_count[heatmap_id]  # self.pair_count记录了每组中包含的关键点对数量
            #     if pair_count > 0:
            #         new_kps_weights[heatmap_id] /= pair_count
            # # 更新heatmap为组合后的heatmap，此时heatmap中包含了关键点连线的热度信息
            # heatmap = combined_heatmap
            # kps_weights = new_kps_weights

                

        # embed()
        # plot_heatmap(image, heatmap, kps, kps_weights)
        target["heatmap"] = torch.as_tensor(heatmap, dtype=torch.float32)
        target["kps_weights"] = torch.as_tensor(kps_weights, dtype=torch.float32)

        return image, target
