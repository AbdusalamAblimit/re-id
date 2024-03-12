


import torch
import torchvision
from IPython import embed
from data_manager import Market1501
from dataset import ImageDataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
from model import ReID
from tqdm import tqdm
from torch import nn
import numpy as np
from loss import TripletHardLoss

from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark=True



from torch.utils.data.sampler import Sampler
import random
from collections import defaultdict



width = 128
height = 256
max_epochs = 100
transform_train = T.Compose([
    T.Resize(list(map(int,[height*9/8, width*9/8]))),
    T.RandomCrop([height,width]),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])


transform_test = T.Compose([
    T.Resize([height, width]),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

data = Market1501()

train_dataset = ImageDataset(data.train, transform=transform_train)
query_dataset = ImageDataset(data.query, transform=transform_test)
gallery_dataset = ImageDataset(data.gallery, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)
gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False)



def main():


    # Loss和Miner
    loss_func = losses.TripletMarginLoss(margin=0.1)
    miner_func = miners.TripletMarginMiner(margin=0.1, type_of_triplets="hard")

    # 优化器

    model = ReID()
    # criterion = TripletHardLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    device = 'cuda:0'
    model = model.to(device)
    train(model,loss_func,miner_func,optimizer,max_epochs,device,evaluate_on_n_epochs=1)
    # embed()
    pass



def train(model, loss_func, miner_func, optimizer, max_epochs,device = 'cuda:0', evaluate_on_n_epochs=5):

    for epoch in range(max_epochs):
        model.train()
        epoch += 1
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        total_loss = 0
        for i,data in pbar:
            imgs,pids,camids = data
            imgs = imgs.to(device)
            pids = pids.to(device)

            optimizer.zero_grad()
            embeddings = model(imgs)
            hard_pairs = miner_func(embeddings, pids)
            loss = loss_func(embeddings, pids, hard_pairs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix({
                              'train_loss' : total_loss/(i+1),
                              })
        if epoch % evaluate_on_n_epochs == 0:
            evaluate(model,device)




def extract_features(model, loader, device):
    model.eval()
    features = torch.FloatTensor()
    pids = []
    camids = []
    with torch.no_grad():
        for data in loader:
            images, pid, camid = data
            images = images.to(device)
            outputs = model(images)
            features = torch.cat((features, outputs.cpu()), 0)
            pids.extend(pid)
            camids.extend(camid)
    return features, np.array(pids), np.array(camids)

def compute_dist_matrix(a, b):
    m, n = a.shape[0], b.shape[0]
    dist_matrix = torch.pow(a, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(b, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_matrix.addmm_(1, -2, a, b.t())
    return dist_matrix.cpu().numpy()

def evaluate_rank_map(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # Exclude same id and same camera
    all_cams = g_camids[indices] == q_camids[:, np.newaxis]
    all_pids = g_pids[indices] == q_pids[:, np.newaxis]
    valid = ~(all_cams & all_pids)
    matches = matches * valid

    # Compute CMC
    cmc = matches.cumsum(axis=1) > 0
    cmc = cmc.astype(np.float32).sum(axis=0) / num_q
    cmc = cmc[:max_rank]

    # Compute mAP
    num_rel = matches.sum(axis=1)
    tmp_cmc = matches.cumsum(axis=1) * matches
    tmp_cmc = [x / (np.arange(len(x)) + 1) for x in tmp_cmc]
    tmp_cmc = np.asarray([x.sum() / (num_rel[i] + 1e-12) for i, x in enumerate(tmp_cmc)])
    mAP = tmp_cmc.sum() / num_q

    return cmc, mAP

def evaluate(model, device):
    qf, q_pids, q_camids = extract_features(model, query_loader, device)
    gf, g_pids, g_camids = extract_features(model, gallery_loader, device)
    distmat = compute_dist_matrix(qf, gf)
    cmc, mAP = evaluate_rank_map(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Results:")
    print("mAP: {:.1%}".format(mAP))
    for r in [1, 3, 5, 10, 20]:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    return cmc, mAP



if __name__=='__main__':
    main()