


import torch

from data_manager import Market1501
from dataset import ImageDataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
from model import ReID
from tqdm import tqdm
import numpy as np
from samplers import   RandomIdentitySampler
from torch.optim import lr_scheduler
from loss import ReIdTotalLoss
from trainsforms import RandomErasing
import random
from IPython import embed
import os
import time
from loguru import logger
import shutil

def main(cfg):

    os.makedirs(cfg.output_root_dir,exist_ok=True)
    time_now = time.strftime("%Y%m%d%H%M%S", time.localtime())
    cfg.output_dir = os.path.join(cfg.output_root_dir,cfg.name,time_now)
    os.makedirs(os.path.join(cfg.output_root_dir,cfg.name),exist_ok=True)
    os.makedirs(cfg.output_dir,exist_ok=True)
    shutil.copyfile(cfg.config_file, os.path.join(cfg.output_dir,'config.yaml'))

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""))
    logger.add(
        sink=os.path.join(cfg.output_dir, 'training.log'),
        level='INFO',     
        encoding='utf-8',  
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.add(
        sink=os.path.join(cfg.output_dir, 'warning-error.log'),
        level='WARNING',     
        encoding='utf-8',  
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )


    torch.manual_seed(cfg.random_seed)  # cpu种子
    torch.cuda.manual_seed_all(cfg.random_seed)  # 所有可用GPU的种子
    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)

    width = cfg.width
    height = cfg.height

    transform_train = T.Compose([
        T.Resize([height,width]),
        T.Pad(10),
        T.RandomCrop([height,width]),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        RandomErasing(probability=0.5,mean=[0.485,0.456,0.406])
    ])


    transform_test = T.Compose([
        T.Resize([height, width]),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])


    if cfg.dataset == 'Market1501':
        data = Market1501()

    train_dataset = ImageDataset(data.train, transform=transform_train)
    query_dataset = ImageDataset(data.query, transform=transform_test)
    gallery_dataset = ImageDataset(data.gallery, transform=transform_test)

    # PK sampling
    P = cfg.train.batch.p
    K = cfg.train.batch.k

    train_dataset = ImageDataset(data.train, transform = transform_train)
    sampler = RandomIdentitySampler(data.train, num_instances = K)
    train_loader = DataLoader(train_dataset, batch_size = P*K, sampler = sampler, 
                              num_workers = cfg.train.num_workers, pin_memory = cfg.train.pin_memory)
    
    query_loader = DataLoader(query_dataset,  shuffle=False, batch_size = cfg.test.batch_size,
                              num_workers = cfg.test.num_workers, pin_memory = cfg.test.pin_memory)
    
    gallery_loader = DataLoader(gallery_dataset, batch_size = cfg.test.batch_size, shuffle = False,
                                num_workers = cfg.test.num_workers, pin_memory = cfg.test.pin_memory)

    if cfg.model.aligned and cfg.train.loss.metric.loss_func != 'triplet-hard-aligned-reid':
        logger.warning(f'You can only use TripletHardAlignedReid as loss function when training is aligned. Metric loss function has been replaced with TripletHardAlignedReid.')
        cfg.train.loss.metric.loss_func = 'triplet-hard-aligned-reid'
    elif (not cfg.model.aligned) and  cfg.train.loss.metric.loss_func == 'triplet-hard-aligned-reid':
        logger.warning(f'You can not use TripletHardAlignedReid as metric loss function when training is not aligned.Metric loss function has been replaced with TripletHard.')
        cfg.train.loss.metric.loss_func = 'triplet-hard'

    loss_func = ReIdTotalLoss(cfg)
    model = ReID(cfg)
    lr_cfg = cfg.train.lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr = lr_cfg.init_lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = lr_cfg.milestones, gamma = lr_cfg.gamma)

    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cfg.start_epoch = checkpoint["epoch"]


    device = 'cuda:0' if cfg.use_gpu else 'cpu'

    if device == 'cuda:0' and cfg.use_cudnn:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark=True
        torch.backends.cudnn.deterministic = True

    model = model.to(device)

    # test(model,query_loader,gallery_loader,cfg)
    # embed()
    train(model, loss_func, optimizer, scheduler, device, train_loader, query_loader, gallery_loader, cfg)
    # embed()
    pass

from test import test




def train(model, loss_func, optimizer, scheduler, device, train_loader, query_loader, gallery_loader, cfg):
    start_epoch = cfg.train.start_epoch
    max_epochs = cfg.train.max_epochs
    evaluate_on_n_epochs = cfg.train.eval_on_n_epochs

    warmup = cfg.train.warmup

    if warmup.enabled:
        warmup.lr_growth = (cfg.train.lr_scheduler.init_lr - warmup.init_lr) / warmup.iters
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup.init_lr
    try:
        for epoch in range(start_epoch, max_epochs):
            model.train()
            epoch += 1
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
            loss_sums = {}  # 初始化损失累加字典
            total_running_correct_samples = 0
            total_samples = 0

            for i,data in pbar:
                imgs,pids,camids = data
                imgs = imgs.to(device)
                pids = pids.to(device)
                outputs,global_features,local_features = model(imgs)

                loss,loss_dict = loss_func(outputs, global_features, local_features, pids)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # embed()

                if i == 0:
                    for key in loss_dict.keys():
                        loss_sums[key] = 0.0

                # 累加每种损失和总损失
                for key, value in loss_dict.items():
                    loss_sums[key] += value
                
                # compute rank-1 by id
                _, predicted = torch.max(outputs,1) 
                total_running_correct_samples += (predicted == pids).sum().item()
                total_samples += pids.shape[0]
                current_lr = optimizer.param_groups[0]['lr']

                # create pbar information
                postfix_dict = {
                    'acc': total_running_correct_samples / total_samples,
                    'lr': current_lr,
                }

                # update pbar information
                postfix_dict.update({key: value / (i + 1) for key, value in loss_sums.items()})

                pbar.set_postfix(postfix_dict)
            logger.info(f'{pbar.desc}: {len(pbar)} iterations. {pbar.postfix}')
            if warmup.enabled and epoch <= warmup.iters:
                for param_group in optimizer.param_groups:
                    param_group['lr'] += warmup.lr_growth
            scheduler.step()

            if epoch % evaluate_on_n_epochs == 0:
                save_files = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch-1}
                test(model,query_loader,gallery_loader,cfg)
                saved_files_dir = os.path.join(cfg.output_dir, 'save')
                os.makedirs(saved_files_dir,exist_ok=True)
                saved_files_path = os.path.join(saved_files_dir,"checkpoint-interrupted-epoch{}.pth".format(epoch))
                torch.save(save_files, saved_files_path)
                logger.error('')
                
    except KeyboardInterrupt:
        logger.error('Catched KeyboardInterrupt. Saving trained model...')

        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch-1}
        
        saved_files_dir = os.path.join(cfg.output_dir, 'save')
        os.makedirs(saved_files_dir,exist_ok=True)
        saved_files_path = os.path.join(saved_files_dir,"checkpoint-interrupted-epoch{}.pth".format(epoch))
        torch.save(save_files, saved_files_path)
        logger.error('')
    




import argparse
import yaml
def to_namespace(d):
    """递归地将字典转换为Namespace对象"""
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = to_namespace(value)
        return argparse.Namespace(**d)
    return d

def load_config(args):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config = to_namespace(config)
    if args.use_gpu: config.use_gpu = True
    if args.use_cpu: config.use_gpu = False
    if args.output_root_dir: config.output_root_dir = args.output_root_dir
    if args.name: config.name = args.name
    if args.max_epochs: config.max_epochs = args.max_epochs
    config.resume = args.resume
    config.config_file = args.config
    # embed()
    return config


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--config', type=str, default='configs/baseline.yaml', help='Path to the configuration file.')
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument('--output-root-dir', help='Root directory where output files will be saved.')
    parser.add_argument('--name', help='Subdirectory name for saving output. The final output path will be {output-dir}/{name}/yyyy-MM-dd-HH-mm-ss.')
    parser.add_argument('--resume', type=str, help='resume from checkpoint')
    parser.add_argument('--max-epochs', type=int, help='number of total epochs to run')
    args = parser.parse_args()
    assert not (args.use_cpu and args.use_gpu), 'Cannot use both --use-gpu and --use-cpu options. Please specify only one.'
    embed()
    return args


if __name__=='__main__':
    args = parse_args()
    cfg = load_config(args)
    main(cfg)
    quit()
    from samplers import   RandomIdentitySampler
    from dataset import read_image
    import torchvision.transforms.functional as F
    def imshow(inp, title=None):
        """Imshow for Tensor."""

        plt.imshow(inp)
        if title is not None:
            plt.title(title)
    dataset = Market1501()

    # 使用RandomIdentitySampler
    train_dataset = ImageDataset(dataset.train,transform=transform_train)
    sampler = RandomIdentitySampler(dataset.train, num_instances=P)
    train_loader = DataLoader(train_dataset, batch_size=P*K, sampler=sampler, num_workers=0)

    import matplotlib.pyplot as plt

    # 获取一个批次的数据
    data_iter = iter(train_loader)
    images, labels, _ = next(data_iter)

    # embed()
    # 打印批次中的几张图片
    # 假设每行打印4张图片
    # 打印批次中的几张图片
    plt.figure(figsize=(10, 10))
    for i in range(P*K):
        ax = plt.subplot(P, K, i + 1)
        img = images[i]
        # 反归一化处理
        img = F.to_pil_image(img)
        plt.imshow(img)
        ax.set_title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.show()