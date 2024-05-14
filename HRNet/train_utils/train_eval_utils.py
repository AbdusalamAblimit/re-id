import math
import sys
import time

import torch

import transforms
import train_utils.distributed_utils as utils
from .coco_eval import EvalCOCOMetric
from .loss import KpLoss

from IPython import embed
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
from loguru import logger
def save_epoch_heatmaps(epoch, results, targets):
    # 确保目录存在
    epoch_dir = f"./heatmap_each_epochs/epoch_{epoch}"
    os.makedirs(epoch_dir, exist_ok=True)

    bs, num_joints, h, w = results.shape
    for i in range(bs):  # 对于每个样本
        plt.figure(figsize=(num_joints * 3, 6))  # 根据关键点数量调整图片大小
        for j in range(num_joints):  # 对于每个关键点
            # 获取预测的热力图和目标热力图
            heatmap_pred = results[i, j].detach().cpu().numpy()
            heatmap_true = targets[i]['heatmap'][j].detach().cpu().numpy()

            # 拼接目标热力图和预测热力图
            combined_heatmap = np.concatenate((heatmap_true, heatmap_pred), axis=1)

            # 绘制拼接后的热力图
            plt.subplot(1, num_joints, j+1)
            plt.imshow(combined_heatmap, cmap='jet')
            plt.title(f"Joint {j+1}")
            plt.axis('off')

        plt.savefig(f"{epoch_dir}/sample_{i}.png")
        plt.close()

def save_epoch_heatmaps2(epoch, results, targets):
    # 确保目录存在
    epoch_dir = f"./eval/"
    os.makedirs(epoch_dir, exist_ok=True)

    bs, num_joints, h, w = results.shape
    for i in range(bs):  # 对于每个样本
        plt.figure(figsize=(num_joints * 3, 6))  # 根据关键点数量调整图片大小
        for j in range(num_joints):  # 对于每个关键点
            # 获取预测的热力图和目标热力图
            heatmap_pred = results[i, j].detach().cpu().numpy()
            heatmap_true = targets[i,j].detach().cpu().numpy()

            # 拼接目标热力图和预测热力图
            combined_heatmap = np.concatenate((heatmap_true, heatmap_pred), axis=1)

            # 绘制拼接后的热力图
            plt.subplot(1, num_joints, j+1)
            plt.imshow(combined_heatmap, cmap='jet')
            plt.title(f"Joint {j+1}")
            plt.axis('off')

        plt.savefig(f"{epoch_dir}/sample_{i}.png")
        plt.close()

def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mse = KpLoss()
    mloss = torch.zeros(1).to(device)  # mean losses
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = torch.stack([image.to(device) for image in images])

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            results = model(images)
            # embed()
            losses = mse(results, targets)


        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict({"losses": losses})
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            logger.info("Loss is {}, stopping training".format(loss_value))
            logger.info(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
    save_epoch_heatmaps(epoch, results, targets)
    return mloss, now_lr

skeleton = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
split_idxs_dict = {}
for i in range(17):
    split_idxs_dict[i] = []
    for j in range(len(skeleton)):
        if i+1 in skeleton[j]:
            split_idxs_dict[i].append(j)

def split_heatmaps(outputs):
    # outputs 是bs,len(skeleton),h,wd的tensor
    bs = outputs.shape[0]
    c = outputs.shape[1]
    h = outputs.shape[2]
    w = outputs.shape[3]
    res = torch.zeros([bs,17,h,w], device='cuda')
    for key,values in split_idxs_dict.items():
        n_values = len(values)
        for i in range(n_values):
            for j in range(i+1,n_values):
                res[:,key,:,:] += outputs[:,values[i],:,:] * outputs[:,values[j],:,:]
    # Min-max normalization per image
    for i in range(bs):
        for j in range(17):
            img = res[i, j, :, :]
            min_val = img.min()
            max_val = img.max()
            # Avoid division by zero in case max_val equals min_val
            if max_val > min_val:
                res[i, j, :, :] = (img - min_val) / (max_val - min_val)
            else:
                res[i, j, :, :] = torch.zeros_like(img)

    return res

    
kps_weights= [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5]
def split_heatmaps2(merged_heatmaps):
    # 假设merged_heatmaps形状为(N, len(skeleton), H, W)，其中skeleton定义了组合关键点
    N, _, H, W = merged_heatmaps.shape
    num_original_kps = 17  # 原始关键点数目
    original_heatmaps = torch.zeros([N, num_original_kps, H, W], device='cuda')
    
    for i, combined_indexes in enumerate(skeleton):
        valid_indexes = [idx-1 for idx in combined_indexes if idx-1 < num_original_kps]
        # 使用相同的权重重新分配到原始关键点
        weight = kps_weights[i]
        for index in valid_indexes:
            original_heatmaps[:, index, :, :] += merged_heatmaps[:, i, :, :] * weight
    
    # 归一化每个关键点的热力图
    for i in range(num_original_kps):
        img = original_heatmaps[:, i, :, :]
        min_val = img.min()
        max_val = img.max()
        if max_val > min_val:
            original_heatmaps[:, i, :, :] = (img - min_val) / (max_val - min_val)
        else:
            original_heatmaps[:, i, :, :] = torch.zeros_like(img)

    return original_heatmaps

@torch.no_grad()
def evaluate(model, data_loader, device, flip=False, flip_pairs=None):
    if flip:
        assert flip_pairs is not None, "enable flip must provide flip_pairs."

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    key_metric = EvalCOCOMetric(data_loader.dataset.coco, "keypoints", "key_results.json")
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        images = torch.stack([img.to(device) for img in image])

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(images)
        # embed()
        outputs = split_heatmaps(outputs)

        if flip:
            flipped_images = transforms.flip_images(images)
            flipped_outputs = model(flipped_images)
            flipped_outputs = split_heatmaps(flipped_outputs)
            flipped_outputs = transforms.flip_back(flipped_outputs, flip_pairs)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5

        model_time = time.time() - model_time

        # decode keypoint
        reverse_trans = [t["reverse_trans"] for t in targets]
        outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)

        key_metric.update(targets, outputs)
        metric_logger.update(model_time=model_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 同步所有进程中的数据
    key_metric.synchronize_results()

    if utils.is_main_process():
        coco_info = key_metric.evaluate()
    else:
        coco_info = None

    return coco_info
