import os

import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import time
import math
import shutil
import sys
import torch
import pickle
from dataclasses import dataclass
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup
from PIL import Image
from cvcities_base.dataset.vigor_origin_data_form import VigorDatasetEval, VigorDatasetTrain
from cvcities_base.transforms import get_transforms_train, get_transforms_val
from cvcities_base.utils import setup_system, Logger
from cvcities_base.trainer import train
from cvcities_base.evaluate.vigor import evaluate, calc_sim
from cvcities_base.loss import InfoNCE
from cvcities_base.model import TimmModel
import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.transforms as transforms

@dataclass
class Configuration:
    # Model
    model = 'dinov2_vitb14_MixVPR'

    # backbone
    backbone_arch = 'dinov2_vitb14'
    pretrained = False
    layer1 = 2
    use_cls = True
    norm_descs = True

    # Aggregator 聚合方法
    agg_arch = 'MixVPR'
    agg_config = {'in_channels': 768,
                  'in_h': 32,  # 受输入图像尺寸的影响
                  'in_w': 32,
                  'out_channels': 1024,
                  'mix_depth': 2,
                  'mlp_ratio': 1,
                  'out_rows': 4}

    # Override model image size
    img_size: int = 448
    new_hight = 448
    new_width = 448

    # Training
    mixed_precision: bool = True
    seed = 1
    epochs: int = 50
    batch_size: int = 16  # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,1)  # GPU ids for training

    # Similarity Sampling
    custom_sampling: bool = True  # use custom sampling instead of random
    gps_sample: bool = True  # use gps sampling
    sim_sample: bool = True  # use similarity sampling
    neighbour_select: int = 64  # max selection size from pool
    neighbour_range: int = 128  # pool size for selection
    gps_dict_path: str = "/home/ubuntu/VIGOR/2gps_dict_same.pkl"  # gps_dict_cross.pkl | gps_dict_same.pkl

    # Eval
    batch_size_eval: int = 1
    eval_every_n_epoch: int = 4  # eval every n Epoch
    normalize_features: bool = True

    # Optimizer
    clip_grad = 100.  # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False  # Gradient Checkpointing
    use_sgd = True

    # Loss
    label_smoothing: float = 0.1

    # Learning Rate
    lr: float = 0.003  # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"  # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001  # only for "polynomial"

    # Dataset
    data_folder = "/home/ubuntu/VIGOR"
    same_area: bool = True  # True: same | False: cross
    ground_cutting = 0  # cut ground upper and lower

    # Augment Images
    prob_rotate: float = 0.75  # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5  # flipping the sat image and ground images simultaneously

    # Savepath for model checkpoints
    model_path: str ="/home/ubuntu/data/CV-cities_train/vigor_same/second"

    # Eval before training
    zero_shot: bool =True

    # Checkpoint to start from
   #todo(这个是预训练好的模型)
    checkpoint_start_ready="/home/ubuntu/data/CV-cities_train/vigor_same/dinov2_vitb14_MixVPR/2025-03-09_033414/weights_e50_76.9775.pth"
    #todo(这个是要训练的模型)
    checkpoint_start="/home/ubuntu/data/CV-cities_train/vigor_same/second/dinov2_vitb14_MixVPR/2025-04-02_031639/weights_end.pth"

    # set num_workers to 0 if on Windows
    num_workers: int = 4

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = True

    # make cudnn deterministic
    cudnn_deterministic: bool = False


# -----------------------------------------------------------------------------#
# Train Config                                                                #
# -----------------------------------------------------------------------------#

config = Configuration()


def reshape_transform(tensor, height=32, width=32):

    tensor[:,1,:]=tensor[:,1008,:]
    tensor[:,2,:]=tensor[:,1006,:]
    tensor[:,33,:]=tensor[:,1012,:]
    tensor[:,34,:]=tensor[:,1003,:]
    result = tensor[:, 1:, :].reshape(tensor.size(0),height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



if __name__ == '__main__':

    # model_path = "{}/{}/{}".format(config.model_path,
    #                                config.model,
    #                                time.strftime("%Y-%m-%d_%H%M%S"))

    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)
    # shutil.copyfile(os.path.abspath(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    # sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    # setup_system(seed=config.seed,
    #              cudnn_benchmark=config.cudnn_benchmark,
    #              cudnn_deterministic=config.cudnn_deterministic)

    # -----------------------------------------------------------------------------#
    # Model                                                                       #
    # -----------------------------------------------------------------------------#

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    # model = TimmModel(model_name=config.model,
    #                   pretrained=True,
    #                   img_size=config.img_size, backbone_arch=config.backbone_arch, agg_arch=config.agg_arch,
    #                   agg_config=config.agg_config, layer1=config.layer1)
    # print(model)

    # data_config = model.get_config()
    model_first = TimmModel(model_name=config.model,
                      pretrained=True,
                      img_size=config.img_size, backbone_arch=config.backbone_arch, agg_arch=config.agg_arch,
                      agg_config=config.agg_config, layer1=config.layer1,
                      model_type='first' )
    # print(model)
    
    model_state_dict_ready=torch.load(config.checkpoint_start_ready) 
    model_first.load_state_dict(model_state_dict_ready, strict=False) 
    
    # print(model_state_dict_ready.keys())
    del model_state_dict_ready
    
    model = TimmModel(model_name=config.model,
                      pretrained=True,
                      img_size=config.img_size, backbone_arch=config.backbone_arch, agg_arch=config.agg_arch,
                      agg_config=config.agg_config, layer1=config.layer1,
                      model_type='second' ,checkpoint_start_ready=config.checkpoint_start_ready)

    data_config = model_first.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size

    image_size_sat = (img_size, img_size)

    # new_width = img_size * 2
    new_width = img_size
    # new_hight = int(((1024 - 2 * config.ground_cutting) / 2048) * new_width)
    new_hight = img_size
    img_size_ground = (new_hight, new_width)

    # # Activate gradient checkpointing
    # if config.grad_checkpointing:
    #     model_first.set_grad_checkpointing(True)
        # model.set_grad_checkpointing(True)  

    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

        # Data parallel
        
        
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model_first = torch.nn.DataParallel(model_first, device_ids=config.gpu_ids)

    # Model to device   
    model = model.to(config.device)
    model_first = model_first.to(config.device)


    # satellite_transforms = A.Compose([
    #                                   A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
    #                                 # A.Resize(image_size_sat[0], image_size_sat[1], p=1.0),
    #                                   A.Normalize(mean, std),
    #                                   ToTensorV2(),
    #                                  ])

    # ground_transforms = A.Compose([
    #                                 # Cut(cutting=ground_cutting, p=1.0),
    #                                A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
    #                                A.Normalize(mean, std),
    #                                ToTensorV2(),
    #                               ])

    trans = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((image_size_sat[0], image_size_sat[1]))
                                     ])









#============test=============

    ground_imgname= 'Chicago/panorama/1j1tK4_s26qu7lK9ylNjcw,41.868389,-87.642169,.jpg'
    # sat_imgname= 'Chicago/satellite/satellite_41.87272263696881_-87.64185139873237.png'

    
    save_path1='/home/ubuntu/Self_CVCties/feature_ground_image/1_1feature.jpg'
    save_path2='/home/ubuntu/Self_CVCties/feature_ground_image/1_2feature.jpg'

    save_path3='/home/ubuntu/Self_CVCties/feature_ground_image/1_3feature.jpg'
    save_path4='/home/ubuntu/Self_CVCties/feature_ground_image/1_4feature.jpg'    

    
    img_type='ground'
    model_type='main'

    # img_type='sat'
    # model_type='sup'
    
    
    if img_type=='ground': 
        img_path= os.path.join(config.data_folder,ground_imgname)
        if model_type=='main':
    
            model_first.eval()
            print(model_first)
            target_layers=[model_first.model.backbone.dino_model.norm]
            img=cv2.imread(img_path,1)[:,:,::-1]
            # img = cv2.resize(img, (448, 448))
            # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = np.float32(img) / 255
            deal_img=trans(img)
            net_input=transforms.Normalize(mean, std)(deal_img).unsqueeze(0)
            print(net_input.shape)
            print(net_input.dtype)
            net_input.to(config.device)
            cam=pytorch_grad_cam.GradCAMPlusPlus(model=model_first,target_layers=target_layers,reshape_transform=reshape_transform,
                                                                        )

            targets=None
            grayscale_cam=cam(net_input,targets=targets)
            grayscale_cam=grayscale_cam[0,:]
            # print(img.shape)(1024, 2048, 3)
            grayscale_cam=cv2.resize(grayscale_cam, (2048, 1024))
            # print(grayscale_cam.shape) (448, 448)
            visualization_img=show_cam_on_image(img,grayscale_cam,use_rgb=False)
            cv2.imwrite(save_path1,visualization_img)  # 将图像保存到硬盘    
            
            
        elif model_type=='sup':
            model.eval()
            # print(model_first)
            target_layers=[model.model.backbone.dino_model.norm]
            img=cv2.imread(img_path,1)[:,:,::-1]
            # img = cv2.resize(img, (448, 448))
            # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = np.float32(img) / 255
            deal_img=trans(img)
            net_input=transforms.Normalize(mean, std)(deal_img).unsqueeze(0)
            print(net_input.shape)
            print(net_input.dtype)
            net_input.to(config.device)
            cam=pytorch_grad_cam.GradCAMPlusPlus(model=model,target_layers=target_layers,reshape_transform=reshape_transform,
                                                                        )

            targets=None
            grayscale_cam=cam(net_input,targets=targets)
            grayscale_cam=grayscale_cam[0,:]
            # print(img.shape)(1024, 2048, 3)
            grayscale_cam=cv2.resize(grayscale_cam, (2048, 1024))
            # print(grayscale_cam.shape) (448, 448)
            visualization_img=show_cam_on_image(img,grayscale_cam,use_rgb=False)
            cv2.imwrite(save_path2,visualization_img)  # 将图像保存到硬盘       
    
    
#SAT===============================


    elif img_type=='sat': 
        img_path= os.path.join(config.data_folder,sat_imgname)
        if model_type=='main':
            model_first.eval()
            # print(model_first)
            target_layers=[model_first.model.backbone.dino_model.norm]
            img=cv2.imread(img_path,1)[:,:,::-1]
            # img = cv2.resize(img, (448, 448))
            # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = np.float32(img) / 255
            deal_img=trans(img)
            net_input=transforms.Normalize(mean, std)(deal_img).unsqueeze(0)
            print(net_input.shape)
            print(net_input.dtype)
            net_input.to(config.device)
            cam=pytorch_grad_cam.GradCAMPlusPlus(model=model_first,target_layers=target_layers,reshape_transform=reshape_transform,
                                                                        )

            targets=None
            grayscale_cam=cam(net_input,targets=targets)
            grayscale_cam=grayscale_cam[0,:]
            # print(img.shape)(1024, 2048, 3)
            grayscale_cam=cv2.resize(grayscale_cam, (640, 640))
            # print(grayscale_cam.shape) (448, 448)
            visualization_img=show_cam_on_image(img,grayscale_cam,use_rgb=False)
            cv2.imwrite(save_path3,visualization_img)  # 将图像保存到硬盘    
        
        
        elif model_type=='sup':

            model.eval()
            # print(model_first)
            target_layers=[model.model.backbone.dino_model.norm]
            img=cv2.imread(img_path,1)[:,:,::-1]
            # img = cv2.resize(img, (448, 448))
            # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = np.float32(img) / 255
            deal_img=trans(img)
            net_input=transforms.Normalize(mean, std)(deal_img).unsqueeze(0)
            print(net_input.shape)
            print(net_input.dtype)
            net_input.to(config.device)
            cam=pytorch_grad_cam.GradCAMPlusPlus(model=model,target_layers=target_layers,reshape_transform=reshape_transform,
                                                                        )

            targets=None
            grayscale_cam=cam(net_input,targets=targets)
            grayscale_cam=grayscale_cam[0,:]
            # print(img.shape)(1024, 2048, 3)
            grayscale_cam=cv2.resize(grayscale_cam, (640, 640))
            # print(grayscale_cam.shape) (448, 448)
            visualization_img=show_cam_on_image(img,grayscale_cam,use_rgb=False)
            cv2.imwrite(save_path4,visualization_img)  # 将图像保存到硬盘          


     
#=============================
                             

    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # h=img.shape[1]
    # w=img.shape[0]
    # img=ground_transforms(image=img)['image']
    # img=img.unsqueeze(0)

    # model.eval()
    # model_first.eval()


    # with torch.no_grad():
    
    #     img = img.to(config.device)
        
    #     img_feature_first,_ = model_first(img,type=False)
    #     img_feature = model(img,type=False)
        
    #     print(img_feature_first.shape) #([1, 768, 32, 32])
    #     # img_feature_first=img_feature_first.reshape(1,768,32,32)
    #     # img_feature = img_feature.reshape(1,768,32,32)
    # # grad_output = torch.ones_like(img_feature_first)  # 与 img_feature_first 形状相同的张量
    # # img_feature_first.backward(grad_output)
        
    # ######特征一
    # img_feature_first = img_feature_first.requires_grad_()    
    # def extract(g):
    #     global features_grad
    #     features_grad = g
 
    # img_feature_first.register_hook(extract)
    # img_feature_first=img_feature_first.reshape(1,768,32,32)
    # # img_feature_first.backward(grad_out)
    # img_feature_first.backward() # 计算梯度
 
    # grads = features_grad   # 获取梯度
    # print(grads.shape)
    # pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
    # print(pooled_grads.shape)
    # # 此处batch size默认为1，所以去掉了第0维（batch size维）
    # pooled_grads = pooled_grads[0]
    # img_feature_first = img_feature_first[0]
    # # 512是最后一层feature的通道数
    # for i in range(768):
    #     feature1=img_feature_first[i, ...] * pooled_grads[i, ...]

    # feature_1=feature1.detach().cpu().numpy()
    # feature_map1=np.mean(feature_1, axis=0)
    
    # feature_map1 = cv2.resize(feature_map1, (h, w))

    # vmin, vmax = np.percentile(feature_map1, [0, 100])

    # # 限制热力值范围
    # feature_map1= np.clip(feature_map1, vmin, vmax)  
    # feature_map1 = (feature_map1 - feature_map1.min()) / (feature_map1.max() - feature_map1.min())
    # feature_map1 = np.uint8(255 * feature_map1)  # 将热力图转换为RGB格式
    # feature_map1 = cv2.applyColorMap(feature_map1, cv2.COLORMAP_JET)  # 将热力图应用于原始图像

    # # feature_map1 = np.clip(feature_map1, vmin, vmax)  
    # # feature_map1, = (feature_map1, - vmin) / (vmax - vmin)  # 归一化到 [0, 1]
    # img = cv2.imread(img_path)
    # # print(img.shape)
    # # print(feature_map1.shape)
    # # print(img.dtype)
    # # print(feature_map1.dtype)
    # # img = np.uint8(255 * ((img - img.min()) / (img.max() - img.min()))) 

    # superimposed_img1 = cv2.addWeighted(feature_map1, 0.6, img, 1.0, 0)   # 这里的0.4是热力图强度因子
    # superimposed_img1=cv2.cvtColor(superimposed_img1, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(save_path1, superimposed_img1)  # 将图像保存到硬盘       
        
        
        
        
        
        
        
        
        
        
    # feature1=img_feature_first[0]
    # feature_map1=torch.mean(feature1, dim=0)
    
    # feature_map1 = cv2.resize(feature_map1.cpu().numpy(), (h, w))

    # vmin, vmax = np.percentile(feature_map1, [0, 100])

    # # 限制热力值范围
    # feature_map1= np.clip(feature_map1, vmin, vmax)  
    # feature_map1 = (feature_map1 - feature_map1.min()) / (feature_map1.max() - feature_map1.min())
    # feature_map1 = np.uint8(255 * feature_map1)  # 将热力图转换为RGB格式
    # feature_map1 = cv2.applyColorMap(feature_map1, cv2.COLORMAP_JET)  # 将热力图应用于原始图像

    # # feature_map1 = np.clip(feature_map1, vmin, vmax)  
    # # feature_map1, = (feature_map1, - vmin) / (vmax - vmin)  # 归一化到 [0, 1]
    # img = cv2.imread(img_path)
    # # print(img.shape)
    # # print(feature_map1.shape)
    # # print(img.dtype)
    # # print(feature_map1.dtype)
    # # img = np.uint8(255 * ((img - img.min()) / (img.max() - img.min()))) 

    # superimposed_img1 = cv2.addWeighted(feature_map1, 0.6, img, 1.0, 0)   # 这里的0.4是热力图强度因子
    # superimposed_img1=cv2.cvtColor(superimposed_img1, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(save_path1, superimposed_img1)  # 将图像保存到硬盘

    # ######特征二
    # feature2=img_feature[0]
    # feature_map2=torch.mean(feature2, dim=0)
    # feature_map2 = (feature_map2 - feature_map2.min()) / (feature_map2.max() - feature_map2.min())
    # feature_map2 = cv2.resize(feature_map2.cpu().numpy(), (h, w))

    # feature_map2 = np.uint8(255 * feature_map2)  # 将热力图转换为RGB格式
    # feature_map2 = cv2.applyColorMap(feature_map2, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    # superimposed_img2 = cv2.addWeighted(feature_map2, 0.8, img, 1.0, 0)   # 这里的0.4是热力图强度因子
    # superimposed_img2=cv2.cvtColor(superimposed_img2, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(save_path2, superimposed_img2)  # 将图像保存到硬盘



 