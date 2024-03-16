import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from IPython import embed
from HRNet.model.hrnet import HighResolutionNet
"""
    我准备做这样的行人重识别：
    在当前这个的基础上, 加上HRNET作为局部特征。HRNET的输入是一个[bs,num_joints,256,192]的张量（但是我的输入是256*128）
    
"""



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ReID(nn.Module):
    def __init__(self,cfg) -> None:
        super(ReID,self).__init__()
        self.cfg = cfg
        self.in_planes = cfg.model._global.feature_dim
        self.neck = cfg.model._global.neck
        self.num_classes = cfg.train.num_classes

        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        
        if self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes)


    def forward(self,x):
        x = self.base(x)
        # embed()
        x = F.max_pool2d(x,x.size()[2:])
        gf = x.view(x.size(0),-1)
        if self.neck == 'bnneck':
            gf = self.bottleneck(gf)
        if not self.training:
            return gf
        y = self.classifier(gf)
        return y,gf


if __name__ == '__main__':
    model = ReID(10)
    imgs = torch.Tensor(32,3,256,128)
    f = model(imgs)


