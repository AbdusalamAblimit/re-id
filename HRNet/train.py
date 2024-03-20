import json
import os
import datetime

import torch
from torch.utils import data
import numpy as np

import transforms
from model import HighResolutionNet
from my_dataset_coco import CocoKeypoint
from train_utils import train_eval_utils as utils
from IPython import embed
from loguru import logger
from matplotlib import pyplot as plt

def create_model(num_joints, load_pretrain_weights=True):
    model = HighResolutionNet(base_channel=32, num_joints=num_joints)
    
    if load_pretrain_weights:
        # 载入预训练模型权重
        # 链接:https://pan.baidu.com/s/1Lu6mMAWfm_8GGykttFMpVw 提取码:f43o
        weights_dict = torch.load("./pose_hrnet_w32_256x192.pth", map_location='cpu')

        for k in list(weights_dict.keys()):
            # 如果载入的是imagenet权重，就删除无用权重
            if ("head" in k) or ("fc" in k):
                del weights_dict[k]

            # 如果载入的是coco权重，对比下num_joints，如果不相等就删除
            if "final_layer" in k:
                if weights_dict[k].shape[0] != num_joints:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0:
            logger.info("missing_keys: ", missing_keys)

    return model

import torch.nn.functional as F
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
        combined_heatmaps[:, i, :, :] /= np.sum(kps_weights[valid_indexes])
    return combined_heatmaps

def calculate_mse(output, target):
    # 计算MSE
    # embed()
    mse_loss = F.mse_loss(output, target, reduction='mean')
    return mse_loss.item()

def eval(data_loader):
    kps_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5]
    combined_keypoint_indexes=[[0,1,2,3,4],
                 [5,6,11,12],
                 [5,7,9],[6,8,10],
                 [11,13,15],[12,14,16]]
    oroginal_model = create_model(17)
    new_model = create_model(6)
    original_weight_path = './model-17-0.pth'
    new_weight_path = './model-6-20.pth'
    original_checkpoint = torch.load(original_weight_path, map_location='cpu')
    oroginal_model.load_state_dict(original_checkpoint['model'])
    new_checkpoint = torch.load(new_weight_path, map_location='cpu')
    new_model.load_state_dict(new_checkpoint['model'])
    oroginal_model = oroginal_model.to('cuda:0')
    new_model = new_model.to('cuda:0')
    total_mse_new = 0.0
    total_mse_original = 0.0
    from tqdm import tqdm
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for i,(images,targets) in pbar:
        images = images.to('cuda:0')

        target_heatmaps = torch.stack([x['heatmap'] for x in targets])
        target_heatmaps = target_heatmaps.to('cuda:0')
        oroginal_model_output = oroginal_model(images)
        new_model_output=new_model(images)
        merged_original_model_output = merge_heatmaps(oroginal_model_output.detach().cpu().numpy(),np.array(kps_weights),combined_keypoint_indexes)
        merged_original_model_output = torch.tensor(merged_original_model_output,device='cuda:0')
        mse_original = calculate_mse(merged_original_model_output, target_heatmaps)
        mse_new = calculate_mse(new_model_output.detach(), target_heatmaps)
        total_mse_new += mse_new
        total_mse_original += mse_original
        pbar.set_postfix({'mse_original':total_mse_original/(i+1),
                          'mse_new': total_mse_new/(i+1)})
        


def main(args):


    logger.add(
        sink='training.log',
        level='INFO',     
        encoding='utf-8',  
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    with open(args.keypoints_path, "r") as f:
        person_kps_info = json.load(f)

    fixed_size = args.fixed_size
    heatmap_hw = (args.fixed_size[0] // 4, args.fixed_size[1] // 4)
    kps_weights = np.array(person_kps_info["kps_weights"],
                           dtype=np.float32).reshape((args.num_joints,))
    
    combined_keypoint_indexes = person_kps_info["combined_keypoint_indexes"]

    if(args.combine_keypoints):
        args.num_joints = len(combined_keypoint_indexes)

    data_transform = {
        "train": transforms.Compose([
            transforms.HalfBody(0.3, person_kps_info["upper_body_ids"], person_kps_info["lower_body_ids"]),
            transforms.AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),
            transforms.RandomHorizontalFlip(0.5, person_kps_info["flip_pairs"]),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights,
                                         combine_keypoints=args.combine_keypoints,
                                         combined_keypoint_indexes = combined_keypoint_indexes),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=fixed_size),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights,
                                         combine_keypoints=args.combine_keypoints,
                                         combined_keypoint_indexes = combined_keypoint_indexes),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    data_root = args.data_path

    # load train data set
    # coco2017 -> annotations -> person_keypoints_train2017.json
    train_dataset = CocoKeypoint(data_root, "train", transforms=data_transform["train"], fixed_size=args.fixed_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # embed()
    nw = 0
    logger.info('Using %g dataloader workers' % nw)

    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=nw,
                                        collate_fn=train_dataset.collate_fn)

    # load validation data set
    # coco2017 -> annotations -> person_keypoints_val2017.json
    val_dataset = CocoKeypoint(data_root, "val", transforms=data_transform["val"], fixed_size=args.fixed_size,
                               det_json_path=args.person_det)
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=nw,
                                      collate_fn=val_dataset.collate_fn)
    # eval(val_data_loader)

    # create model
    model = create_model(num_joints=args.num_joints)
    # logger.info(model)

    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        logger.info("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, logger.infoing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # # evaluate on the test dataset
        # coco_info = utils.evaluate(model, val_data_loader, device=device,
        #                            flip=True, flip_pairs=person_kps_info["flip_pairs"])

        # # write into txt
        # with open(results_file, "a") as f:
        #     # 写入的数据包括coco指标还有loss和learning rate
        #     result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
        #     txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
        #     f.write(txt + "\n")

        # val_map.append(coco_info[1])  # @0.5 mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        torch.save(save_files, "./save_weights/model-{}.pth".format(epoch))
    quit()

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录(coco2017)
    parser.add_argument('--data-path', default='data/coco2017', help='dataset')
    # COCO数据集人体关键点信息
    parser.add_argument('--keypoints-path', default="./person_keypoints.json", type=str,
                        help='person_keypoints.json path')
    # 原项目提供的验证集person检测信息，如果要使用GT信息，直接将该参数置为None，建议设置成None
    parser.add_argument('--person-det', type=str, default=None)
    parser.add_argument('--fixed-size', default=[256, 192], nargs='+', type=int, help='input size')
    # keypoints点数
    parser.add_argument('--num-joints', default=17, type=int, help='num_joints')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=210, type=int, metavar='N',
                        help='number of total epochs to run')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[170, 200], nargs='+', type=int, help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # 学习率
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # AdamW的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练的batch size
    parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                        help='batch size when training.')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # 是否组合人体关键点
    parser.add_argument("--combine-keypoints", action="store_true", help="Combine keypoints to reduce the number of heatmaps. If you turn on this option, the model can't predict the keypoints.")

    args = parser.parse_args()
    logger.info(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
