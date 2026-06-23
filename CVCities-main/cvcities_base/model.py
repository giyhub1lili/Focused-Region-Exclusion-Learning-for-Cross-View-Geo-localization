import cv2
import torch
import timm
import numpy as np
import torch.nn as nn
from cvcities_base import helper
from torchsummary import summary


class VPRModel(nn.Module):  # 继承pytorch-lightning.LightningModule模块
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                 # ---- Backbone 主干网络
                 model_name='dinov2_vitb14_MixVPR',
                 backbone_arch='dinov2_vitb14',
                 pretrained=True,
                 layers_to_freeze=1,
                 layers_to_crop=[],
                 layer1=20,
                 use_cls=False,
                 norm_descs=True,

                 # ---- Aggregator 聚合方法
                 agg_arch='MixVPR',  # CosPlace, NetVLAD, GeM
                 agg_config={},
                 model_type=None,
                 checkpoint_start_ready=None
                 ):
        super().__init__()
        self.pretrained = pretrained  # 是否预训练
        self.layers_to_freeze = layers_to_freeze  # 冻结网络层名称
        self.layers_to_crop = layers_to_crop  # layers_to_crop=[4],  # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
        self.layer1 = layer1
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.agg_config = agg_config  # 聚合方法参数
        # self.save_hyperparameters()  # write hyperparams into a file
        self.model_name = model_name
        self.model_type=model_type
        self.batch_acc = []  # we will keep track of the % of trivial pairs/triplets at the loss level
        self.checkpoint_start_ready=checkpoint_start_ready
        # ----------------------------------
        # get the backbone and the aggregator 获得主干网络和聚合器
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layer1=self.layer1, use_cls=self.use_cls,
                                            norm_descs=self.norm_descs,model_type=model_type,checkpoint_start_ready=checkpoint_start_ready)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config,model_type=model_type,checkpoint_start_ready=checkpoint_start_ready)
#         if model_type=='second':
#             model_state_dict_ready=torch.load(self.checkpoint_start_ready) 
#             # print(model_state_dict_ready.keys())
# #             missing_keys, unexpected_keys = self.backbone.load_state_dict(
# #     {k:v for k,v in model_state_dict_ready.items() if k.startswith('model.backbone')}, 
# #     strict=False
# # )
#             # print('111111111111111111111')
#             # print(model_state_dict_ready.keys())
#             # print('222222222222222')
#             # print(self.backbone.state_dict().keys())
#             # print('333333')
#             # print(self.aggregator.state_dict().keys())
#             adjusted_state_dict = {}
#             adjusted_agg_state_dict = {}
#             for k, v in model_state_dict_ready.items():
#                 if k.startswith("model.backbone."):
#                     new_key = k.replace("model.backbone.", "")  # 适配 self.backbone
#                     adjusted_state_dict[new_key] = v
#                 elif k.startswith("model.aggregator."):
#                     new_key = k.replace("model.aggregator.", "")  # 适配 self.agg
                
#                     adjusted_agg_state_dict[new_key] = v

#             # self.backbone.load_state_dict({k:v for k,v in adjusted_state_dict.items() if k.startswith('dino_model')},strict=False)
#             missing_keys, unexpected_keys =self.aggregator.load_state_dict({k:v for k,v in adjusted_agg_state_dict.items() },strict=False)
#             print("缺失的参数（未加载成功）:", missing_keys)
#             print("多余的参数（未使用）:", unexpected_keys) 



    # the forward pass of the lightning model
    def forward(self, x,type,mask=None):
        if self.model_type=='first':
            if type==False:
                x = self.backbone(x)
                x = self.aggregator(x)
                mask=None
                return x,mask
            else:
                x = self.backbone(x)
                features=x.detach().clone()
                x = self.aggregator(x)
                # 对每一张图像生成热力图
                heat_mask=[]
                for i in range(x.size(0)):
                    # 选择特征图的通道（例如选择第一个通道）
                    feature_map = features[i]  # shape: [dim, 32, 32]

                    # 对多个通道求均值，生成单通道的热力图（可以选择其他方法，如最大值等）
                    heatmap = torch.mean(feature_map, dim=0)  # shape: [32, 32]

                    # 归一化处理
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                
                    # 将热力图调整为与原图大小一致

                    # heatmap_resized = cv2.resize(heatmap.cpu().numpy(), (img.shape[1], img.shape[2]))  # 原图尺寸
                    
                    # heatmap_resized = cv2.resize(heatmap.cpu().numpy(), (32, 32))  # 中间特征图尺寸
                    heatmap_resized = heatmap.cpu().numpy() 

                    # 生成 mask
                    threshold = np.percentile(heatmap_resized, 80)  # 前 20% 的分位数
                    mask = np.ones_like(heatmap_resized, dtype=np.float32)  # 初始化为全 1
                    mask[heatmap_resized >= threshold] = 0  # 最热区域设置为 0，其余设置为 1
                    # 转换为 tensor 格式并存储
                    mask_tensor = torch.from_numpy(mask).to(x.device)  # 转为 float tensor，
                    heat_mask.append(mask_tensor)
                    
                heat_mask=torch.stack(heat_mask)
                return x,heat_mask
        elif self.model_type=='second':
            if type==False:
                x = self.backbone(x)
                x = self.aggregator(x)
                return x
            else:
                x = self.backbone(x,mask=mask)
                x = self.aggregator(x)
                return x
            
        else:
            raise('this is an error')
            
    # def forward(self, x,type,mask=None):
    #     if self.model_type=='first':
    #         if type==False:
    #             x = self.backbone(x)
    #             x = self.aggregator(x)
    #             mask=None
    #             return x
    #         else:
    #             x = self.backbone(x)
    #             x = self.aggregator(x)
    #             heat_mask=None
    #             return x
    #     elif self.model_type=='second':
    #         if type==False:
    #             x = self.backbone(x)
    #             x = self.aggregator(x)
    #             return x
    #         else:
    #             x = self.backbone(x,mask)
    #             x = self.aggregator(x)
    #             return x
            
    #     else:
    #         raise('this is an error')

class TimmModel(nn.Module):

    def __init__(self,
                 model_name='dinov2_vitb14_MixVPR',
                 pretrained_path=None,
                 backbone_arch='',
                 pretrained=True,
                 img_size=224,
                 layer1=8,
                 # Aggregator 聚合方法
                 agg_arch='MixVPR',
                 agg_config={},
                 model_type=None,
                 checkpoint_start_ready=None
                 ):

        super(TimmModel, self).__init__()

        self.img_size = img_size
        self.model_type=model_type
        if "dino" in backbone_arch:
            self.model = VPRModel(backbone_arch=backbone_arch, agg_arch=agg_arch, layer1=layer1, agg_config=agg_config,model_type=model_type,checkpoint_start_ready=checkpoint_start_ready)
        elif "vitt" in backbone_arch:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        #first
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        #second
        self.logit_scale1 = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale2 = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale3 = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale4 = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.bottleneck = nn.BatchNorm1d(4096)
        self.classifier = torch.nn.Linear(4096,701, bias=False)
        if pretrained_path:
            # 加载预训练模型的权重，但不包括输出层的权重
            state_dict = torch.load(pretrained_path)
            print("Start from:", pretrained_path)
            self.load_state_dict(state_dict)

    def get_config(self):
        # data_config = timm.data.resolve_model_data_config(self.model)
        # data_config = self.model.default_cfg
        data_config = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, img1, img2=None,mask=None,type=None,test=None):

        if img2 is not None:

            image_features1 = self.model(img1)
            image_features2 = self.model(img2)

            return image_features1, image_features2

        else:
            if self.model_type=='first':
                image_features,mask = self.model(img1,type,mask=mask)
                # return image_features
                if test == None:
                    return image_features,mask
                else:
                    return self.classifier(self.bottleneck(image_features)),mask
            elif self.model_type=='second':
                image_features = self.model(img1,type,mask=mask)
                if test == None:
                    return image_features
                return self.classifier(self.bottleneck(image_features))
            else:
                raise('this is an error')
                return None
            
            # image_features = self.model(img1,False,mask=mask)
            # return image_features

