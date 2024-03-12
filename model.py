import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from IPython import embed

class ReID(nn.Module):
    def __init__(self) -> None:
        super(ReID,self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained = True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        # self.classifier = nn.Linear(2048, num_classes)
    def forward(self,x):
        x = self.base(x)
        # embed()
        x = F.avg_pool2d(x,x.size()[2:])
        f = x.view(x.size(0),-1)
        # embed()
        # if not self.training:
        #     return f  
        # y = self.classifier(f)
        # y = F.softmax(self.classifier(f),dim=1)
        return f


if __name__ == '__main__':
    model = ReID(10)
    imgs = torch.Tensor(32,3,256,128)
    f = model(imgs)


