import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from IPython import embed
from HRNet.model.hrnet import HighResolutionNet
from torchvision.models.resnet import resnet50, resnet34, resnet18
from loguru import logger
from utils import save_combined_features,save_combined_features_to_tensorboard,UnNormalize





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

"""
    我准备做这样的行人重识别：
    在当前这个的基础上, 加上HRNET作为局部特征。HRNET的输入是一个[bs,256,192]的张量我想做两个分支：
    1. 全局特征提取
    2. 基于HRNET的局部特征提取
        2.1 HRNET输入原图得到一个[bs,num_joints,64,48]的热力图
        2.2 一个CNN网络输入原图得到一个 [bs,num_joints,64,48]的特征图（代表图片的颜色信息等）
        将2.1和2.2部分合并得到[bs,2*num_joints,64,48]的混合特征, 继续输入一个小型cnn网络,然后展开得到2048维度的局部特征向量
        其中HRNET可以根据：cfg.model.local.hrnet.lock是否为True来决定是否锁定HRNET原来的参数
"""


class FeatureTransform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureTransform, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttentionModule(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channel = max(1, num_channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # embed()
        y = self.avg_pool(x)
        y = y.view(b,c)
        y = self.fc(y)
        y = y.view(b,c,1,1)
        return x * y.expand_as(x)


class LocalFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super(LocalFeatureExtractor, self).__init__()
        self.cfg = cfg
        self.hrnet_cfg = cfg.model.local.hrnet
        self.hrnet = self._init_hrnet(self.hrnet_cfg)
        

        # 初始化ResNet50作为颜色信息提取器，切割至layer1
        self.color_feature_extractor = self._init_color_feature_extractor()

        self.heatmap_channel_attention = ChannelAttentionModule(num_channels=256) 
        self.color_channel_attention = ChannelAttentionModule(num_channels=256) 
        self.combined_channel_attention = ChannelAttentionModule(num_channels=256) 

        # 定义处理HRNet heatmap的卷积层
        self.heatmap_transform = FeatureTransform(cfg.model.local.hrnet.num_joints, 256)

        # 定义处理color_features的卷积层
        self.color_features_transform = FeatureTransform(256, 128)
        
        # 定义合并特征后的处理层
        self.combined_features_transform = FeatureTransform(256, 256)

        # 定义融合后继续的ResNet50部分，从layer2开始
        self.resnet_continued = self._init_resnet_continued()
    
    def _init_hrnet(self,hrnet_cfg):

        model = HighResolutionNet(base_channel=32, num_joints=hrnet_cfg.num_joints)
        if hrnet_cfg.pretrained.enabled:
            weights_dict = torch.load(hrnet_cfg.pretrained.path, map_location='cpu')

            for k in list(weights_dict.keys()):
                # 如果载入的是imagenet权重，就删除无用权重
                if ("head" in k) or ("fc" in k):
                    del weights_dict[k]

                # 如果载入的是coco权重，对比下num_joints，如果不相等就删除
                if "final_layer" in k:
                    if weights_dict[k].shape[0] != hrnet_cfg.num_joints:
                        del weights_dict[k]
            missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
            if len(missing_keys) != 0:
                logger.info(f"missing_keys: {missing_keys}" )

        if hrnet_cfg.lock:
            for param in model.parameters():
                param.requires_grad = False
        return model

    def _init_color_feature_extractor(self):
        # 初始化ResNet50并保留到layer1的部分
        model = resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)
        # 提取ResNet50直到layer1的部分
        layers = list(model.children())[:5]  # 包括layer1
        color_feature_extractor = nn.Sequential(*layers)
        return color_feature_extractor

    def _init_resnet_continued(self):
        # 初始化从ResNet50的layer2开始的部分
        model = resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)
        # 提取从layer2到倒数第三层的部分
        layers = list(model.children())[5:-2]  # 从layer2开始，不包括最后的全局平均池化和fc层
        resnet_continued = nn.Sequential(*layers)
        return resnet_continued
    
    def forward(self, x):
        # HRNet提取特征
        heatmap = self.hrnet(x)
        heatmap = self.heatmap_transform(heatmap)  # 将通道数变为256

        attention_heatmap = self.heatmap_channel_attention(heatmap)

        # 使用ResNet50到layer1的部分提取颜色特征
        color_features = self.color_feature_extractor(x) # 输出的通道数为256

        attention_color = self.color_channel_attention(color_features)

        # 合并特征
        combined_features = attention_color * attention_heatmap

        combined_features = self.combined_features_transform(combined_features)

        # 继续通过ResNet50的剩余部分
        features = self.resnet_continued(combined_features)
        
        # 应用池化和展平
        f = F.adaptive_avg_pool2d(features, (1, 1))
        f = torch.flatten(f, 1)
        
        return f


    def draw_feature_to_tensorboard(self,x,writer,tag_prefix):

        unnnorm = UnNormalize()


        heatmap = self.hrnet(x)

        save_combined_features_to_tensorboard(x,heatmap,writer,f'{tag_prefix}_heatmap_hrnet',unnnorm)

        heatmap = self.heatmap_transform(heatmap)  # 将通道数变为256

        save_combined_features_to_tensorboard(x,heatmap,writer,f'{tag_prefix}_heatmap_transformed',unnnorm)

        attention_heatmap = self.heatmap_channel_attention(heatmap)

        save_combined_features_to_tensorboard(x,attention_heatmap,writer,f'{tag_prefix}_heatmap_attention',unnnorm)

        # 使用ResNet50到layer1的部分提取颜色特征
        color_features = self.color_feature_extractor(x) # 输出的通道数为256

        save_combined_features_to_tensorboard(x,color_features,writer,f'{tag_prefix}_color',unnnorm)

        attention_color = self.color_channel_attention(color_features)

        save_combined_features_to_tensorboard(x,attention_color,writer,f'{tag_prefix}_color_attention',unnnorm)

        # 合并特征
        combined_features = attention_color * attention_heatmap

        save_combined_features_to_tensorboard(x,combined_features,writer,f'{tag_prefix}_combined',unnnorm)

        combined_features = self.combined_features_transform(combined_features)

        save_combined_features_to_tensorboard(x,combined_features,writer,f'{tag_prefix}_combined_transformed',unnnorm)

        # 继续通过ResNet50的剩余部分
        features = self.resnet_continued(combined_features)

        save_combined_features_to_tensorboard(x,features,writer,f'{tag_prefix}_final_feature_maps',unnnorm)



class ReID(nn.Module):
    def __init__(self,cfg) -> None:
        super(ReID,self).__init__()
        self.cfg = cfg
        self.in_planes = cfg.model._global.feature_dim
        self.neck = cfg.model._global.neck
        self.num_classes = cfg.train.num_classes
        if cfg.model.aligned:
            self.local_net = LocalFeatureExtractor(cfg)

        resnet50 = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        # self.base = self.local_net

        if self.cfg.train.loss.id.enabled:
            if self.neck == 'bnneck':
                self.bottleneck = nn.BatchNorm1d(self.in_planes)
                self.bottleneck.bias.requires_grad_(False)  # no shift
                self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

                self.bottleneck.apply(weights_init_kaiming)
                self.classifier.apply(weights_init_classifier)
            else:
                self.classifier = nn.Linear(self.in_planes, self.num_classes)


    def forward(self,x):
        # embed()
        gf = self.base(x)
        # embed()
        gf = F.max_pool2d(gf,gf.size()[2:])
        gf = gf.view(gf.size(0),-1)
        if self.cfg.model.aligned:
            lf = self.local_net(x)
        else: 
            lf = gf
        if self.cfg.train.loss.id.enabled and self.neck == 'bnneck':
            gf = self.bottleneck(gf)
        if not self.training:
            return gf,lf
        y = None
        if self.cfg.train.loss.id.enabled:
            y = self.classifier(gf)
        return y,gf,lf


if __name__ == '__main__':
    from train import parse_args, load_config
    args = parse_args()
    cfg = load_config(args)


    model = LocalFeatureExtractor(cfg)
    a=torch.zeros([64,3,256,192])
    # res = ModifiedResNet34(3,17)
    # embed()
    model(a)
    logger.info('end')
    
# import torch
# import torchvision.models as models

# def logger.info_resnet50_layers_output_size():
#     net = models.resnet50(pretrained=True)  # 加载预训练的ResNet50模型
#     x = torch.randn(1, 3, 256, 192)  # 假设的输入尺寸，这里为[batch_size, channels, height, width]

#     logger.info("Input size:", x.size())

#     for name, layer in net.named_children():
#         x = layer(x)
#         logger.info(f"{name}: {x.size()}")

# # 调用函数查看每层输出尺寸
# logger.info_resnet50_layers_output_size()


