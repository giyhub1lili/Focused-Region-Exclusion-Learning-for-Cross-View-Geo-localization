import os

import cv2
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

from cvcities_base.dataset.cvusa import CVUSADatasetEval, CVUSADatasetTrain
from cvcities_base.transforms import get_transforms_train, get_transforms_val
from cvcities_base.utils import setup_system, Logger
from cvcities_base.trainer import train
from cvcities_base.evaluate.cvusa_and_cvact import evaluate, calc_sim
from cvcities_base.loss import InfoNCE
from cvcities_base.model import TimmModel

# #反向关注机制，用来提取特征图
class MixedModel_first(torch.nn.Module):
    def __init__(self, children1):
        super(MixedModel_first, self).__init__()
      
        
        #TODO（在这里）
        
        # self.norm=torch.nn.LayerNorm(512)
        # self.pool=torch.nn.AdaptiveAvgPool2d((1,1))
        # self.norm=torch.nn.LayerNorm((1,1),eps=1e-6,elementwise_affine=True)
        # self.identity=torch.nn.Identity()
        # self.drop=torch.nn.Dropout(p=0.0,inplace=False)
        
        
        # self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        blocks=list(children1[0].dino_model.blocks.children()) 
        # print(blocks)
        self.deal1=torch.nn.Sequential(
            children1[0].dino_model.patch_embed,
            *blocks[:4]
        )    
        self.deal2=torch.nn.Sequential(
            *blocks[4:],
            children1[0].dino_model.norm,
            children1[0].dino_model.head
        )
        self.head=torch.nn.Sequential(
            children1[1]
        )
        
       

        # self.j=0
    def forward(self, x):

        # if x.size(2)==x.size(3):
        #     x=self.deal2(self.deal1(x))
        #     # x_mid=self.deal1(x)   
     
        #     ###从中间层得到的热力图太分散了，而且不起到辅导作用，怎么办  最后层的热力图上采样到这一层？
        #     # print(x_mid.shape)
        #     features=x.detach().clone()
        #     # 对每一张图像生成热力图
        #     heat_mask=[]
        #     for i in range(x.size(0)):
        #         # 选择特征图的通道（例如选择第一个通道）
        #         feature_map = features[i]  # shape: [dim, 24, 24]

        #         # 对多个通道求均值，生成单通道的热力图（可以选择其他方法，如最大值等）
        #         heatmap = torch.mean(feature_map, dim=0)  # shape: [24, 24]

        #         # 归一化处理
        #         heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
        #         # 将热力图调整为与原图大小一致

        #         # heatmap_resized = cv2.resize(heatmap.cpu().numpy(), (img.shape[1], img.shape[2]))  # 原图尺寸
                
        #         heatmap_resized = cv2.resize(heatmap.cpu().numpy(), (48, 48))  # 中间特征图尺寸


        #         # 生成 mask
        #         threshold = np.percentile(heatmap_resized, 80)  # 前 20% 的分位数
        #         mask = np.ones_like(heatmap_resized, dtype=np.float32)  # 初始化为全 1
        #         mask[heatmap_resized >= threshold] = 0  # 最热区域设置为 0，其余设置为 1
        #         # 转换为 tensor 格式并存储
        #         mask_tensor = torch.from_numpy(mask).to(x.device)  # 转为 float tensor，形状 [48, 48]
        #         heat_mask.append(mask_tensor)
                
        #     heat_mask=torch.stack(heat_mask)

            
        # else:
        #     x=self.deal2(self.deal1(x))    
        #     heat_mask=None
        x=self.deal2(self.deal1(x))    
        x=self.head(x)
        # return x,heat_mask
        return x


#二次检索模型，用于反向关注机制，在中间加入掩码
class MixedModel_second(torch.nn.Module):
    def __init__(self, children1,children2):
        super(MixedModel_second, self).__init__()
      
        
        #TODO（在这里）
        
        # self.norm=torch.nn.LayerNorm(512)
        # self.pool=torch.nn.AdaptiveAvgPool2d((1,1))
        # self.norm=torch.nn.LayerNorm((1,1),eps=1e-6,elementwise_affine=True)
        # self.identity=torch.nn.Identity()
        # self.drop=torch.nn.Dropout(p=0.0,inplace=False)
        
        #有必要再加一个logit_scale，用来给损失函数加上可学习的权重
        
        self.logit_scale1 = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # self.loss_weight = torch.nn.Parameter(torch.ones([]) * np.log(0.05))
        self.logit_scale2=torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale3 = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale4 = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.deal1=torch.nn.Sequential(
           torch.nn.Sequential(*(children1[0])),
            torch.nn.Sequential(*(children1[1])[:-2])
        )
        
        self.deal2=torch.nn.Sequential(
            torch.nn.Sequential(*(children2[1])[-2:]),
            torch.nn.Sequential(*(children2[-2:-1]) )
        )
        
        self.head=torch.nn.Sequential(*(children2[-1:]) )
        
        # self.spatial_attention = SpatialAttentionAdapter(kernel_size=7)
    def forward(self, x,type,mask=None):  #university中二者都是同样大小，再设一个标志
 
 
        # if x.size(2)==x.size(3):
        if type==True:
            # if mask==None:
            #     x=self.deal2(self.deal1(x))
            # else:
            x_mid=self.deal1(x)
            
            mask=mask.unsqueeze(1) 
            x_mid_change=x_mid*mask

            # x_mid_change=self.spatial_attention(x_mid_change)
            
            x=self.deal2(x_mid_change)
        else:
            x=self.deal2(self.deal1(x))

        x=self.head(x)
        return x


@dataclass
class Configuration:
    # Model
    model = 'dinov2_vitb14_MixVPR'

    # backbone
    backbone_arch = 'dinov2_vitb14'
    pretrained = False
    layers_to_freeze = 1
    layers_to_crop = []
    layer1 = -1
    use_cls = True
    norm_descs = True

    # Aggregator 聚合方法
    agg_arch = 'MixVPR'
    agg_config = {'in_channels': 768,
                  'in_h': 32,
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
    epochs: int = 40
    batch_size: int = 10  # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,1)  # GPU ids for training

    # Similarity Sampling
    custom_sampling: bool = True  # use custom sampling instead of random
    gps_sample: bool = True  # use gps sampling
    sim_sample: bool = True  # use similarity sampling
    neighbour_select: int = 64  # max selection size from pool
    neighbour_range: int = 128  # pool size for selection
    gps_dict_path: str = "/home/ubuntu/CVUSA/gps_dict.pkl"  # path to pre-computed distances

    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 1  # eval every n Epoch
    normalize_features: bool = True

    # Optimizer
    clip_grad = 100.  # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False  # Gradient Checkpointing
    use_sgd = True

    # Loss
    label_smoothing: float = 0.1

    # Learning Rate
    lr: float = 0.005  # 1 * 10^-4 for ViT | 1 * 10^-3 for CNN   0.0002 for adam, 0.05 for sgd (needs to change according to batch size)
    scheduler: str = "cosine"  # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001  # only for "polynomial"

    # Dataset
    data_folder = "/home/ubuntu/CVUSA/"

    # Augment Images
    prob_rotate: float = 0.75  # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5  # flipping the sat image and ground images simultaneously

    # Savepath for model checkpoints
    model_path: str = "/home/ubuntu/data/CV-cities_train/cvusa/second"

    # Eval before training
    zero_shot: bool = True

    # Checkpoint to start from
#    #todo(这个是预训练好的模型)
#     checkpoint_start_ready="/home/ubuntu/data/CV-cities_train/cvusa/dinov2_vitb14_MixVPR/093227/weights_e31_99.1896.pth"
#     #todo(这个是要训练的模型)
#     checkpoint_start=None
    #CVACT-->CVUSA
   #todo(这个是预训练好的模型)
    checkpoint_start_ready="/home/ubuntu/data/CV-cities_train/cvact/dinov2_vitb14_MixVPR/2025-03-20_050929/weights_e29_92.4471.pth"
    #todo(这个是要训练的模型)
    checkpoint_start="/home/ubuntu/data/CV-cities_train/cvact/second/dinov2_vitb14_MixVPR/2025-04-02_034941/weights_e23_93.0774.pth"
   

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 7

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

if __name__ == '__main__':

    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   time.strftime("%Y-%m-%d_%H%M%S"))
    # src_file = os.path.basename(__file__)  # 获取当前文件名
    # print(src_file )
    # if not os.path.exists(src_file):
    #     raise FileNotFoundError(f"源文件 {src_file} 不存在！请检查路径。")
    # print(src_file )
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.abspath(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    # -----------------------------------------------------------------------------#
    # Model                                                                        #
    # -----------------------------------------------------------------------------#

    print("\nModel: {}".format(config.model))

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
    # # #变成我们想要的model_first   
    # children1=list(model_first.model.children())
    # print('============================================')
    # # deal1=torch.nn.Sequential(
    # #     #    torch.nn.Sequential((children1[0]))
    # #         torch.nn.Sequential((children1[0]).dino_model.blocks[-1])
    # #         # torch.nn.Sequential(*(children1[1])[:-2])
    # #     )
   
    # # print(children1[0])

    # model=MixedModel_first(children1=children1)
    data_config = model_first.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size

    image_size_sat = (img_size, img_size)

    # new_width = config.img_size * 2
    # new_hight = round((224 / 1232) * new_width)
    new_width = config.new_width
    new_hight = config.new_hight
    img_size_ground = (new_hight, new_width)

    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model_first.set_grad_checkpointing(True)
        model.set_grad_checkpointing(True)  
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
    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    # -----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    # -----------------------------------------------------------------------------#

    # Transforms
    sat_transforms_train, ground_transforms_train = get_transforms_train(image_size_sat,
                                                                         img_size_ground,
                                                                         mean=mean,
                                                                         std=std,
                                                                         )

    # Train
    train_dataset = CVUSADatasetTrain(data_folder=config.data_folder,
                                      transforms_query=ground_transforms_train,
                                      transforms_reference=sat_transforms_train,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size
                                      )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)

    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   )

    # Reference Satellite Images
    reference_dataset_test = CVUSADatasetEval(data_folder=config.data_folder,
                                              split="test",
                                              img_type="reference",
                                              transforms=sat_transforms_val,
                                              )

    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)

    # Query Ground Images Test
    query_dataset_test = CVUSADatasetEval(data_folder=config.data_folder,
                                          split="test",
                                          img_type="query",
                                          transforms=ground_transforms_val,
                                          )

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))

    # -----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    # -----------------------------------------------------------------------------#
    if config.gps_sample:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    # -----------------------------------------------------------------------------#
    # Sim Sample                                                                  #
    # -----------------------------------------------------------------------------#

    if config.sim_sample:
        # Query Ground Images Train for simsampling
        query_dataset_train = CVUSADatasetEval(data_folder=config.data_folder,
                                               split="train",
                                               img_type="query",
                                               transforms=ground_transforms_val,
                                               )

        query_dataloader_train = DataLoader(query_dataset_train,
                                            batch_size=config.batch_size_eval,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=True)

        reference_dataset_train = CVUSADatasetEval(data_folder=config.data_folder,
                                                   split="train",
                                                   img_type="reference",
                                                   transforms=sat_transforms_val,
                                                   )

        reference_dataloader_train = DataLoader(reference_dataset_train,
                                                batch_size=config.batch_size_eval,
                                                num_workers=config.num_workers,
                                                shuffle=False,
                                                pin_memory=True)

        print("\nReference Images Train:", len(reference_dataset_train))
        print("Query Images Train:", len(query_dataset_train))

        # -----------------------------------------------------------------------------#
    # Loss                                                                        #
    # -----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn,
                            device=config.device,
                            )

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2. ** 10)
    else:
        scaler = None

    # -----------------------------------------------------------------------------#
    # optimizer                                                                   #
    # -----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    if config.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)




    # 冻结第一个模型
    #todo
    for param in model_first.parameters():
        param.requires_grad = False
    # -----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    # -----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end=config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)

    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)

    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps)

    else:
        scheduler = None

    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    # -----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    # -----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30 * "-", "Zero Shot", 30 * "-"))


        r1_test = evaluate(config=config,
                               model=model,
                               model_first=model_first,
                               reference_dataloader=reference_dataloader_test,
                               query_dataloader=query_dataloader_test,
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)
            #cal_sim使用单独特征
        if config.sim_sample:
            r1_train, sim_dict = calc_sim(config=config,
                                              model=model,
                                              model_first=model_first,
                                              reference_dataloader=reference_dataloader_train,
                                              query_dataloader=query_dataloader_train,
                                              ranks=[1, 5, 10],
                                              step_size=1000,
                                              cleanup=True)

    # # -----------------------------------------------------------------------------#
    # # Shuffle                                                                     #
    # # -----------------------------------------------------------------------------#
    if config.custom_sampling:
        train_dataloader.dataset.shuffle(sim_dict,
                                         neighbour_select=config.neighbour_select,
                                         neighbour_range=config.neighbour_range)

    # -----------------------------------------------------------------------------#
    # Train                                                                       #
    # -----------------------------------------------------------------------------#
    start_epoch = 0
    best_score = 0

    for epoch in range(1, config.epochs + 1):

        print("\n{}[Epoch: {}]{}".format(30 * "-", epoch, 30 * "-"))

        train_loss = train(config,
                           model,
                           model_first,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)

        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))

        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:

            print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))
            r1_test = evaluate(config=config,
                                model=model,
                                model_first=model_first,
                                reference_dataloader=reference_dataloader_test,
                                query_dataloader=query_dataloader_test,
                                ranks=[1, 5, 10],
                                step_size=1000,
                                cleanup=True)
                #cal_sim使用单独特征
            if config.sim_sample:
                r1_train, sim_dict = calc_sim(config=config,
                                                model=model,
                                                model_first=model_first,
                                                reference_dataloader=reference_dataloader_train,
                                                query_dataloader=query_dataloader_train,
                                                ranks=[1, 5, 10],
                                                step_size=1000,
                                                cleanup=True)

            if r1_test > best_score:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(),
                               '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))

        if config.custom_sampling:
            train_dataloader.dataset.shuffle(sim_dict,
                                             neighbour_select=config.neighbour_select,
                                             neighbour_range=config.neighbour_range)

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))            
