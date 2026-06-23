# --<utf-8>--


import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal
from torchsummary import summary
import numpy as np
from einops import repeat


# Extract features from a Dino-v2 model
_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]

class DinoV2_myself(nn.Module):
    """
        Extract features from an intermediate layer in Dino-v2
        从 Dino-v2 中的中间层提取特征
    """

    def __init__(self, model_name: _DINO_V2_MODELS, layer1: int = 39,  facet1: _DINO_FACETS = "value", use_cls=False,
                 norm_descs=True,checkpoint_start_ready=None ,device: str = "cuda:0", pretrained=True) -> None:
        """
            Parameters:
            - dino_model:   The DINO-v2 model to use
            - layer:        The layer to extract features from
            - facet:    "query", "key", or "value" for the attention
                        facets. "token" for the output of the layer.
            - use_cls:  If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
            - norm_descs:   If True, the descriptors are normalized
            - device:   PyTorch device to use
        """
        super().__init__()
        self.model_name = model_name.lower()  # 将大写转化为小写
        self.layer1 = layer1

        self.pretrained = pretrained  # 是否采用与训练参数
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.device = torch.device(device)
        self.vit_type: str = model_name
        self.checkpoint_start_ready=checkpoint_start_ready

        print(f'loading DINOv2 model（{self.model_name}）...')
        if 'vitg14' in self.model_name:
            self.dino_model = torch.hub.load(r'D:\python_code\MixVPR(hgs)\models\backbones\facebookresearch_dinov2_main\dinov2', self.model_name, trust_repo=True, source='local')  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'D:\python_code\MixVPR(hgs)\models\backbones\facebookresearch_dinov2_main\dinov2_vitg14_pretrain.pth'))
            if self.layer1 > 39:
                print('请确认layer的正确性！vitg14最高block层为39层')
                exit()
        elif 'vitl14' in self.model_name:
            self.dino_model = torch.hub.load(r'D:\python_code\MixVPR(hgs)\models\backbones\facebookresearch_dinov2_main\dinov2', self.model_name, trust_repo=True, source='local')  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r'D:\python_code\MixVPR(hgs)\models\backbones\facebookresearch_dinov2_main\dinov2_vitl14_pretrain.pth'))
            if self.layer1 > 23:
                print('请确认layer的正确性！vitl14最高block层为23层')
                exit()
        elif 'vitb14' in self.model_name:
            self.dino_model = torch.hub.load(r"/home/ubuntu/ssd/lishuangjiang/dinov2-main", self.model_name, trust_repo=True, source='local')  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r"/home/ubuntu/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"))
            if self.layer1 > 11:
                print('请确认layer的正确性！vitb14最高block层为12层')
                exit()
        elif 'vits14' in self.model_name:
            # self.dino_model = torch.hub.load(r'D:\python_code\MixVPR(hgs)\models\backbones\facebookresearch_dinov2_main\dinov2', self.model_name, trust_repo=True, source='local')  # 加载DINOv2预训练模型
            # self.dino_model.load_state_dict(torch.load(r'D:\python_code\MixVPR(hgs)\models\backbones\facebookresearch_dinov2_main\dinov2_vits14_pretrain.pth'))
            self.dino_model = torch.hub.load(r"/home/ubuntu/dinov2-main", self.model_name, trust_repo=True, source='local')  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load(r"/home/ubuntu/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"))
            if self.layer1 > 11:
                print('请确认layer的正确性！vits14最高block层为12层')
                exit()
        else:
            print(f'模型名称定义错误，请检查model_name:{self.dino_model}是否正确')
            
        #todo
        #todo(对于没有用same model方法保存的需要按照下面的来)    
        model_state_dict_ready=torch.load(self.checkpoint_start_ready)     
        #todo(对于用same model方法保存的需要按照下面的来)
        # model_state_dict_ready1=torch.load(self.checkpoint_start_ready) 
        # model_state_dict_ready=model_state_dict_ready1
        # print(model_state_dict_ready.keys())
        #对state_dict进行处理，使其能被加载
        patch_embed_state_dict = {}
        blocks0_state_dict = {}
        blocks1_state_dict = {}
        blocks2_state_dict = {}
        blocks3_state_dict = {}
        blocks4_state_dict = {}
        blocks5_state_dict = {}
        blocks6_state_dict = {}
        blocks7_state_dict = {}
        blocks8_state_dict = {}
        blocks9_state_dict = {}
        blocks10_state_dict = {}
        blocks11_state_dict = {}
        for k, v in model_state_dict_ready.items():
            if k.startswith("model.backbone.dino_model.patch_embed."):
                new_key = k.replace("model.backbone.dino_model.patch_embed.", "")  # 适配 self.backbone
                patch_embed_state_dict[new_key] = v
            elif k.startswith("model.backbone.dino_model.blocks.0."):
                new_key = k.replace("model.backbone.dino_model.blocks.0.", "")  # 适配 self.agg
                blocks0_state_dict[new_key] = v
            elif k.startswith("model.backbone.dino_model.blocks.1."):
                new_key = k.replace("model.backbone.dino_model.blocks.1.", "")  # 适配 self.agg
                blocks1_state_dict[new_key] = v
            elif k.startswith("model.backbone.dino_model.blocks.2."):
                new_key = k.replace("model.backbone.dino_model.blocks.2.", "")  # 适配 self.agg
                blocks2_state_dict[new_key] = v
            elif k.startswith("model.backbone.dino_model.blocks.3."):
                new_key = k.replace("model.backbone.dino_model.blocks.3.", "")  # 适配 self.agg
                blocks3_state_dict[new_key] = v
            elif k.startswith("model.backbone.dino_model.blocks.4."):
                new_key = k.replace("model.backbone.dino_model.blocks.4.", "")  # 适配 self.agg
                blocks4_state_dict[new_key] = v
            elif k.startswith("model.backbone.dino_model.blocks.5."):
                new_key = k.replace("model.backbone.dino_model.blocks.5.", "")  # 适配 self.agg
                blocks5_state_dict[new_key] = v
            elif k.startswith("model.backbone.dino_model.blocks.6."):
                new_key = k.replace("model.backbone.dino_model.blocks.6.", "")  # 适配 self.agg
                blocks6_state_dict[new_key] = v
            elif k.startswith("model.backbone.dino_model.blocks.7."):
                new_key = k.replace("model.backbone.dino_model.blocks.7.", "")  # 适配 self.agg
                blocks7_state_dict[new_key] = v
            elif k.startswith("model.backbone.dino_model.blocks.8."):
                new_key = k.replace("model.backbone.dino_model.blocks.8.", "")  # 适配 self.agg
                blocks8_state_dict[new_key] = v
            elif k.startswith("model.backbone.dino_model.blocks.9."):
                new_key = k.replace("model.backbone.dino_model.blocks.9.", "")  # 适配 self.agg
                blocks9_state_dict[new_key] = v
            elif k.startswith("model.backbone.dino_model.blocks.10."):
                new_key = k.replace("model.backbone.dino_model.blocks.10.", "")  # 适配 self.agg
                blocks10_state_dict[new_key] = v
            elif k.startswith("model.backbone.dino_model.blocks.11."):
                new_key = k.replace("model.backbone.dino_model.blocks.11.", "")  # 适配 self.agg
                blocks11_state_dict[new_key] = v

            
        
        blocks_freeze=0
        # print(self.dino_model.blocks[1].state_dict().keys())
        self.dino_model.patch_embed.load_state_dict({k:v for k,v in patch_embed_state_dict.items() },strict=False)
        for i in range(0, self.layer1 + 1+blocks_freeze):
        # for i in range(0, 0):
            state_dict_name = f"blocks{i}_state_dict"  # 动态拼接变量名
            state_dict = locals().get(state_dict_name)  # 获取变量
            self.dino_model.blocks[i].load_state_dict({k:v for k,v in state_dict.items() },strict=False)
            # self.dino_model.blocks1[i].load_state_dict({k:v for k,v in state_dict.items() },strict=False)

        print('load sucessfully')
        self.dino_model = self.dino_model.to(self.device)
        if pretrained:
            self.dino_model.patch_embed.requires_grad_(False)
            # self.dino_model.cls_token.requires_grad_(False)
            # self.dino_model.pos_embed.requires_grad_(False)
            # self.dino_model.mask_token.requires_grad_(False)
            #设置共有的训练层
            
            for i in range(0, self.layer1 + 1+blocks_freeze):
                self.dino_model.blocks[i].requires_grad_(False)
                # self.dino_model.blocks1[i].requires_grad_(False)
            # try:
            #     for i in range(self.layer1 + 1, len(self.dino_model.blocks)):
            #         self.dino_model.blocks[i] = nn.Sequential()
            # except Exception as e:
            #     print(f'报错：{e}')

        #     self.dino_model.blocks[self.layer1].attn = nn.Sequential()
        #     self.dino_model.blocks[self.layer1].ls1 = nn.Sequential()
        #     self.dino_model.blocks[self.layer1].drop_path = nn.Sequential()

        #     self.dino_model.blocks[self.layer1].mlp = nn.Sequential()
        #     self.dino_model.blocks[self.layer1].ls2 = nn.Sequential()

        # self.dino_model.blocks[self.layer1].drop_path1 = None
        # self.dino_model.blocks[self.layer1].norm2 = None
        # self.dino_model.blocks[self.layer1].mlp = None
        # self.dino_model.blocks[self.layer1].ls2 = None
        # self.dino_model.blocks[self.layer1].drop_path2 = None
        #
        self.dino_model.norm = nn.Sequential()
        self.dino_model.head = nn.Sequential()


    def forward(self, x, masks=None,mask=None):

        x = self.dino_model.forward_features(x,mask=mask)

        # if isinstance(x, list):
        #     return self.forward_features_list(x, masks)
        #
        # x = self.dino_model.prepare_tokens_with_masks(x, masks)
        # for blk_num in range(len(self.dino_model.blocks)):
        #     x = self.dino_model.blocks[blk_num](x)
        #     if blk_num == 7:
        #         x1 = x
        #     if blk_num == 9:
        #         x2 = x
        #     if blk_num == 11:
        #         x3 = x
        # x = torch.cat((x1, x2, x3), dim=2)

        x = x['x_norm_patchtokens']  # 取无cls的输出
        bs, f, c = x.shape

        # x = (x + x.mean(dim=2, keepdim=True)) * 0.5

        x = x.view(bs, int(np.sqrt(f)), int(np.sqrt(f)), c)  # 拆分通道，转换成特征图形式
        return x.permute(0, 3, 1, 2)



def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


def main():
    x = torch.randn(1, 3, 224, 224).to('cuda')
    model = DinoV2_myself(model_name='dinov2_vitb14', layer1=11, facet1="value", use_cls=False, norm_descs=True, device="cuda", pretrained=True)
    # torch.onnx.export(model.dino_model, torch.randn(1, 3, 224, 224), 'dinov2_vitl14.onnx', do_constant_folding=True, verbose=False)

    print(model)
    # print(model.dino_model.cls_token)
    # print(model.dino_model.pos_embed)
    # print(model.dino_model.mask_token)
    for name, param in model.dino_model.named_parameters():
        if param.requires_grad:
            print(f'***{name}**')

    print('-' * 70)
    summary(model, (3, 224, 224), 1, 'cuda')
    print('-' * 70)

    r = model(x)

    print_nb_params(model)

    print(f'Input shape is {x.shape}')
    print(f'Output shape is {r.shape}')


if __name__ == '__main__':
    main()

