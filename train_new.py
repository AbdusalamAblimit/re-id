


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
from utils import read_sample_images
from test import test_concat
from torch.utils.tensorboard import SummaryWriter

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
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop([height,width]),
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
    sampler = RandomIdentitySampler(data.train,batch_size= P*K,num_instances=K)
    train_loader = DataLoader(train_dataset, batch_size = P*K, sampler = sampler, 
                              num_workers = cfg.train.num_workers, pin_memory = cfg.train.pin_memory,
                              drop_last= True)
    
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

    device = 'cuda:0' if cfg.use_gpu else 'cpu'
    loss_func = ReIdTotalLoss(cfg)
    # model = ReID(cfg)
    from model import Baseline
    model = Baseline(751)
    model = model.to(device)

    lr_cfg = cfg.train.lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr = lr_cfg.init_lr,weight_decay=0.0005)
    from scheduler import WarmupMultiStepLR
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = lr_cfg.milestones, gamma = lr_cfg.gamma)
    scheduler =  WarmupMultiStepLR(optimizer,milestones=lr_cfg.milestones,gamma=lr_cfg.gamma,warmup_factor=0.01,warmup_iters=10,warmup_method='linear')


    
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cfg.train.start_epoch = checkpoint["epoch"]

    if device == 'cuda:0' and cfg.use_cudnn:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark=True
        torch.backends.cudnn.deterministic = True

    def val_collate_fn(batch):
        imgs, pids, camids = zip(*batch)
        return torch.stack(imgs, dim=0), pids, camids

    val_set = ImageDataset(data.query + data.gallery, transform_test)
    val_loader = DataLoader(
        val_set, batch_size=cfg.test.batch_size, shuffle=False, num_workers=cfg.test.num_workers,
        collate_fn=val_collate_fn
    )

    if cfg.tensorboard.enabled:
        tb_writer = SummaryWriter(os.path.join(cfg.output_dir,'tb'))
        cfg.tb_writer = tb_writer
        sample = torch.rand([1,3,height,width],device = device)
        tb_writer.add_graph(model, sample)
    sample_images = read_sample_images('./sample_images',transform_test,device)
    # embed()
    # test(model,query_loader,gallery_loader,cfg)
    # test_concat(model,query_loader,gallery_loader,cfg)
    # embed()
    from trainer import do_train
    do_train(cfg,model,train_loader,val_loader,optimizer,scheduler,loss_func,len(data.query),cfg.train.start_epoch)
    # embed()
    pass





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
    # embed()
    return args


if __name__=='__main__':
    args = parse_args()
    cfg = load_config(args)
    main(cfg)