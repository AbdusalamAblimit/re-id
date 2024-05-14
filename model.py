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


# class LocalFeatureExtractor(nn.Module):
#     def __init__(self, cfg):
#         super(LocalFeatureExtractor, self).__init__()
#         self.cfg = cfg
#         self.hrnet_cfg = cfg.model.local.hrnet
#         self.hrnet = self._init_hrnet(self.hrnet_cfg)
        

#         # 初始化ResNet50作为颜色信息提取器，切割至layer1
#         self.color_feature_extractor = self._init_color_feature_extractor()

#         self.heatmap_channel_attention = ChannelAttentionModule(num_channels=256) 
#         self.color_channel_attention = ChannelAttentionModule(num_channels=256) 
#         self.combined_channel_attention = ChannelAttentionModule(num_channels=256) 

#         # 定义处理HRNet heatmap的卷积层
#         self.heatmap_transform = FeatureTransform(cfg.model.local.hrnet.num_joints, 256)

        
#         # 定义合并特征后的处理层
#         self.combined_features_transform = FeatureTransform(256, 256)

#         # 定义融合后继续的ResNet50部分，从layer2开始
#         self.resnet_continued = self._init_resnet_continued()
    
#     def _init_hrnet(self,hrnet_cfg):

#         model = HighResolutionNet(base_channel=32, num_joints=hrnet_cfg.num_joints)
#         if hrnet_cfg.pretrained.enabled:
#             weights_dict = torch.load(hrnet_cfg.pretrained.path, map_location='cpu')

#             for k in list(weights_dict.keys()):
#                 # 如果载入的是imagenet权重，就删除无用权重
#                 if ("head" in k) or ("fc" in k):
#                     del weights_dict[k]

#                 # 如果载入的是coco权重，对比下num_joints，如果不相等就删除
#                 if "final_layer" in k:
#                     if weights_dict[k].shape[0] != hrnet_cfg.num_joints:
#                         del weights_dict[k]
#             missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
#             if len(missing_keys) != 0:
#                 logger.info(f"missing_keys: {missing_keys}" )

#         if hrnet_cfg.lock:
#             for param in model.parameters():
#                 param.requires_grad = False
#         return model

#     def _init_color_feature_extractor(self):
#         # 初始化ResNet50并保留到layer1的部分
#         model = resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)
#         # 提取ResNet50直到layer1的部分
#         layers = list(model.children())[:5]  # 包括layer1
#         color_feature_extractor = nn.Sequential(*layers)
#         return color_feature_extractor

#     def _init_resnet_continued(self):
#         # 初始化从ResNet50的layer2开始的部分
#         model = resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)
#         # 提取从layer2到倒数第三层的部分
#         layers = list(model.children())[5:-2]  # 从layer2开始，不包括最后的全局平均池化和fc层
#         resnet_continued = nn.Sequential(*layers)
#         return resnet_continued
    
#     def forward(self, x):
#         # HRNet提取特征
#         heatmap = self.hrnet(x)
#         heatmap = self.heatmap_transform(heatmap)  # 将通道数变为256

#         attention_heatmap = self.heatmap_channel_attention(heatmap)

#         # 使用ResNet50到layer1的部分提取颜色特征
#         color_features = self.color_feature_extractor(x) # 输出的通道数为256

#         attention_color = self.color_channel_attention(color_features)

#         # 合并特征
#         combined_features = attention_color * attention_heatmap

#         combined_features = self.combined_features_transform(combined_features)

#         # 继续通过ResNet50的剩余部分
#         features = self.resnet_continued(combined_features)
        
#         # 应用池化和展平
#         f = F.adaptive_avg_pool2d(features, (1, 1))
#         f = torch.flatten(f, 1)
        
#         return f


#     def draw_feature_to_tensorboard(self,x,writer,tag_prefix):

#         unnnorm = UnNormalize()


#         heatmap = self.hrnet(x)

#         save_combined_features_to_tensorboard(x,heatmap,writer,f'{tag_prefix}_heatmap_hrnet',unnnorm)

#         heatmap = self.heatmap_transform(heatmap)  # 将通道数变为256

#         save_combined_features_to_tensorboard(x,heatmap,writer,f'{tag_prefix}_heatmap_transformed',unnnorm)

#         attention_heatmap = self.heatmap_channel_attention(heatmap)

#         save_combined_features_to_tensorboard(x,attention_heatmap,writer,f'{tag_prefix}_heatmap_attention',unnnorm)

#         # 使用ResNet50到layer1的部分提取颜色特征
#         color_features = self.color_feature_extractor(x) # 输出的通道数为256

#         save_combined_features_to_tensorboard(x,color_features,writer,f'{tag_prefix}_color',unnnorm)

#         attention_color = self.color_channel_attention(color_features)

#         save_combined_features_to_tensorboard(x,attention_color,writer,f'{tag_prefix}_color_attention',unnnorm)

#         # 合并特征
#         combined_features = attention_color * attention_heatmap

#         save_combined_features_to_tensorboard(x,combined_features,writer,f'{tag_prefix}_combined',unnnorm)

#         combined_features = self.combined_features_transform(combined_features)

#         save_combined_features_to_tensorboard(x,combined_features,writer,f'{tag_prefix}_combined_transformed',unnnorm)

#         # # 继续通过ResNet50的剩余部分
#         # features = self.resnet_continued(combined_features)

#         # save_combined_features_to_tensorboard(x,features,writer,f'{tag_prefix}_final_feature_maps',unnnorm)

class FeatureFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureFusion, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gf, lf):
        # 特征连接
        x = torch.cat((gf, lf), dim=1)  # 假设gf和lf的shape都是 [batch_size, 2048]
        # 通过全连接层
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x



from ibn.resnet import build_resnet_backbone

class BaseNet(nn.Module):
    def __init__(self, resnet):
        super(BaseNet, self).__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        # Assuming that NL_1 and its indices are necessary here
        self.NL_1 = resnet.NL_1 if hasattr(resnet, 'NL_1') else None
        self.NL_1_idx = resnet.NL_1_idx if hasattr(resnet, 'NL_1_idx') else []

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        NL1_counter = 0
        if len(self.NL_1_idx) == 0:
            self.NL_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        return x



class ContinuedNet(nn.Module):
    def __init__(self, resnet):
        super(ContinuedNet, self).__init__()
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Including Non-local modules and their indices for each layer if available
        self.NL_2 = resnet.NL_2
        self.NL_2_idx = resnet.NL_2_idx
        self.NL_3 = resnet.NL_3
        self.NL_3_idx = resnet.NL_3_idx
        self.NL_4 = resnet.NL_4
        self.NL_4_idx = resnet.NL_4_idx

    def forward(self, x):
        # layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0:
            self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1

        # layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0:
            self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1

        # layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0:
            self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1

        return x

class ReID(nn.Module):
    def __init__(self,cfg) -> None:
        super(ReID,self).__init__()
        self.cfg = cfg
        self.in_planes = cfg.model._global.feature_dim
        self.neck = cfg.model.neck
        self.num_classes = cfg.train.num_classes

        if cfg.model.aligned:
            self.hrnet = self._init_hrnet(cfg.model.local.hrnet)
            self.heatmap_channel_attention = ChannelAttentionModule(num_channels=256) 
            self.color_channel_attention = ChannelAttentionModule(num_channels=256) 
            self.combined_channel_attention = ChannelAttentionModule(num_channels=256) 
            # 定义处理HRNet heatmap的卷积层
            self.heatmap_transform = FeatureTransform(cfg.model.local.hrnet.num_joints, 256)
            # 定义合并特征后的处理层
            self.combined_features_transform = FeatureTransform(256, 256)
  
            self.local_net = self._init_resnet_continued()
            

        self.base = self._init_color_feature_extractor()
        self.global_net = self._init_resnet_continued()
        # self.base = self.local_net

        if self.neck == 'bnneck':
            self.global_bottleneck = nn.BatchNorm1d(self.in_planes)
            self.global_bottleneck.bias.requires_grad_(False)  # no shift
            self.global_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.global_bottleneck.apply(weights_init_kaiming)
            self.global_classifier.apply(weights_init_classifier)
            if cfg.model.aligned:
                self.local_bottleneck = nn.BatchNorm1d(self.in_planes)
                self.local_bottleneck.bias.requires_grad_(False)  # no shift
                self.local_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
                self.local_bottleneck.apply(weights_init_kaiming)
                self.local_classifier.apply(weights_init_classifier)

        else:
            self.global_classifier = nn.Linear(self.in_planes, self.num_classes)
            if cfg.model.aligned:
                self.local_classifier = nn.Linear(self.in_planes, self.num_classes)


    def forward(self,x):
        # embed()
        f = self.base(x)
        
        # embed()
        gf = self.global_net(f)
        # embed()
        # gf = F.max_pool2d(gf,gf.size()[2:])
        gf = F.adaptive_avg_pool2d(gf,1)
        gf = gf.view(gf.size(0),-1)

        gf_after_bnneck = self.global_bottleneck(gf)
        
        if self.cfg.model.aligned:
            heatmap = self.hrnet(x)
            heatmap = self.heatmap_transform(heatmap)  # 将通道数变为256
            attention_heatmap = self.heatmap_channel_attention(heatmap)
            attention_color = self.color_channel_attention(f)
            # 合并特征
            combined_features = attention_color * attention_heatmap
            combined_features = self.combined_features_transform(combined_features)
            # 继续通过ResNet50的剩余部分
            lf = self.local_net(combined_features)
            # lf = F.max_pool2d(lf,lf.size()[2:])
            lf = F.adaptive_avg_pool2d(lf,1)
            lf = lf.view(lf.size(0),-1)
            lf_after_bnneck = self.local_bottleneck(lf)
        else: 
            lf = gf
            lf_after_bnneck = gf_after_bnneck

        if not self.training:
            return gf_after_bnneck,lf_after_bnneck
        gy = None
        fy = None
        if self.cfg.train.loss.id.enabled:
            gy = self.global_classifier(gf_after_bnneck)
            if self.cfg.model.aligned:
                fy = self.local_classifier(lf_after_bnneck)
        return gy,fy,gf,lf


    def freeze_backbone(self):
        logger.info("Freezed the backbone.")
        for param in self.base.parameters():
            param.requires_grad = False
        for param in self.global_net.parameters():
            param.requires_grad = False
        if self.cfg.model.aligned:
            for param in self.local_net.parameters():
                param.requires_grad = False
    def unfreeze_backbone(self):
        logger.info("Unfreezed the backbone.")
        for param in self.base.parameters():
                param.requires_grad = True
        for param in self.global_net.parameters():
                param.requires_grad = True
        if self.cfg.model.aligned:
            for param in self.local_net.parameters():
                param.requires_grad = True

    def _init_color_feature_extractor(self):
        backbone = self.cfg.model.backbone
        if backbone == 'resnet50':
        # 初始化ResNet50并保留到layer1的部分
            model = resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)
            layers = list(model.children())[:5]  # 包括layer1
            model = nn.Sequential(*layers)
        elif backbone == 'ibn50':
            model = build_resnet_backbone(depth='50x')
            model = BaseNet(model)
        elif backbone == 'ibn101':
            model = build_resnet_backbone(depth='101x')
            model = BaseNet(model)
        else:
            raise f'Wrong backbone name: {backbone}'
        return model

    def _init_resnet_continued(self):
        backbone = self.cfg.model.backbone
        if backbone == 'resnet50':
            model = resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)
            layers = list(model.children())[5:-2]
            model = nn.Sequential(*layers)
        elif backbone == 'ibn50':
            model = build_resnet_backbone('50x')
            model = ContinuedNet(model)
        elif backbone == 'ibn101':
            model = build_resnet_backbone('101x')
            model = ContinuedNet(model)
        else:
            raise f'Wrong backbone name: {backbone}'
        return model
        
    
    
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
    def draw_feature_to_tensorboard(self,x,writer,tag_prefix):

        unnnorm = UnNormalize()


        heatmap = self.hrnet(x)

        save_combined_features_to_tensorboard(x,heatmap,writer,f'{tag_prefix}_heatmap_hrnet',unnnorm)

        heatmap = self.heatmap_transform(heatmap)  # 将通道数变为256

        save_combined_features_to_tensorboard(x,heatmap,writer,f'{tag_prefix}_heatmap_transformed',unnnorm)

        attention_heatmap = self.heatmap_channel_attention(heatmap)

        save_combined_features_to_tensorboard(x,attention_heatmap,writer,f'{tag_prefix}_heatmap_attention',unnnorm)

        # 使用ResNet50到layer1的部分提取颜色特征
        color_features = self.base(x) # 输出的通道数为256

        save_combined_features_to_tensorboard(x,color_features,writer,f'{tag_prefix}_color',unnnorm)

        attention_color = self.color_channel_attention(color_features)

        save_combined_features_to_tensorboard(x,attention_color,writer,f'{tag_prefix}_color_attention',unnnorm)

        # 合并特征
        combined_features = attention_color * attention_heatmap

        save_combined_features_to_tensorboard(x,combined_features,writer,f'{tag_prefix}_combined',unnnorm)

        combined_features = self.combined_features_transform(combined_features)

        save_combined_features_to_tensorboard(x,combined_features,writer,f'{tag_prefix}_combined_transformed',unnnorm)

# if __name__ == '__main__':
#     from train import parse_args, load_config
#     args = parse_args()
#     cfg = load_config(args)


#     model = LocalFeatureExtractor(cfg)
#     a=torch.zeros([64,3,256,192])
#     # res = ModifiedResNet34(3,17)
#     # embed()
#     model(a)
#     logger.info('end')
    
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




import math


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)    # add missed relu
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




class Baseline(nn.Module):
    in_planes = 2048

    def __init__ (self, num_classes, neck="bnneck", neck_feat="after"):
        super(Baseline, self).__init__()
        self.base = ResNet(last_stride=1,
                            block=Bottleneck,
                            layers=[3, 4, 6, 3])

        self.base.load_param('/home/server/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):

        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat,global_feat # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat,feat
            else:
                # print("Test with feature before BN")
                return global_feat,global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
