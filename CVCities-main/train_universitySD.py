import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import math
import shutil
import sys
import torch
from dataclasses import dataclass
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup

from cvcities_base.dataset.university import U1652DatasetEval, U1652DatasetTrain, get_transforms
from cvcities_base.utils import setup_system, Logger
from cvcities_base.trainer import train
from cvcities_base.evaluate.university import evaluate
from cvcities_base.loss import InfoNCE
from cvcities_base.model import TimmModel


def save_model(model_path,model,optimizer,epoch,best_acc,scheduler):
    
    model_checkpoint = model_path
    checkpoint = {
        'model':model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),
        'epoch':epoch,
        'best_acc':best_acc
    }
    torch.save(checkpoint, model_checkpoint)  

# @dataclass
# class Configuration:
#     # Model
#     model = 'dinov2_vitb14_MixVPR'

#     # backbone
#     backbone_arch = 'dinov2_vitb14'
#     pretrained = True
#     layer1 = 7
#     use_cls = True
#     norm_descs = True

#     # Aggregator 聚合方法
#     agg_arch = 'MixVPR'
#     agg_config = {'in_channels': 768,
#                   'in_h': 32,  # 受输入图像尺寸的影响
#                   'in_w': 32,
#                   'out_channels': 1024,
#                   'mix_depth': 2,
#                   'mlp_ratio': 1,
#                   'out_rows': 4}
#     # Override model image size
#     img_size: int = 448
#     new_hight = 448
#     new_width = 448

#     # Training
#     mixed_precision: bool = True
#     custom_sampling: bool = True  # use custom sampling instead of random
#     seed = 1
#     epochs: int = 40
#     batch_size: int = 16  # keep in mind real_batch_size = 2 * batch_size
#     verbose: bool = True
#     gpu_ids: tuple = (0,1)  # GPU ids for training

#     # Eval
#     batch_size_eval: int = 128
#     eval_every_n_epoch: int = 4  # eval every n Epoch
#     normalize_features: bool = True
#     eval_gallery_n: int = -1  # -1 for all or int

#     # Optimizer
#     clip_grad = 100.  # None | float
#     decay_exclue_bias: bool = False
#     grad_checkpointing: bool = False  # Gradient Checkpointing
#     use_sgd = True

#     # Loss
#     label_smoothing: float = 0.1

#     # Learning Rate
#     lr: float = 0.005  # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
#     scheduler: str = "cosine"  # "polynomial" | "cosine" | "constant" | None
#     warmup_epochs: int = 0.1
#     lr_end: float = 0.0001  # only for "polynomial"

#     # Dataset
#     dataset: str = 'U1652-S2D'  # 'U1652-D2S' | 'U1652-S2D'
#     data_folder: str = "/home/ubuntu/University-1652/University-Release"

#     # Augment Images
#     prob_flip: float = 0.5  # flipping the sat image and drone image simultaneously

#     # Savepath for model checkpoints
#     model_path: str = "/home/ubuntu/data/CV-cities_train/university/s2d/second"

#     # Eval before training
#     zero_shot: bool = True

#     # Checkpoint to start from
#    #todo(这个是预训练好的模型)
#     checkpoint_start_ready="/home/ubuntu/data/CV-cities_train/university/dinov2_vitb14_MixVPR/97.29/weights_e8_0.9729.pth"
#     #todo(这个是要训练的模型)
#     checkpoint_start=None
#     #s2d: 97.29  D2S:95.61
#     # set num_workers to 0 if on Windows
#     num_workers: int = 0 if os.name == 'nt' else 7

#     # train on GPU if available
#     device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # for better performance
#     cudnn_benchmark: bool = True

#     # make cudnn deterministic
#     cudnn_deterministic: bool = False

@dataclass
class Configuration:
    # Model
    model = 'dinov2_vitb14_MixVPR'

    # backbone
    backbone_arch = 'dinov2_vitb14'
    pretrained = True
    layer1 = 7
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
    custom_sampling: bool = True  # use custom sampling instead of random
    seed = 1
    epochs: int = 40
    batch_size: int = 10  # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0)  # GPU ids for training

    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 1  # eval every n Epoch
    normalize_features: bool = True
    eval_gallery_n: int = -1  # -1 for all or int

    # Optimizer
    clip_grad = 100.  # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False  # Gradient Checkpointing
    use_sgd = True

    # Loss
    label_smoothing: float = 0.1

    # Learning Rate
    lr: float = 0.005  # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"  # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 0.1
    lr_end: float = 0.0001  # only for "polynomial"

    # Dataset
    dataset: str = 'U1652-S2D'  # 'U1652-D2S' | 'U1652-S2D'
    data_folder: str = "/home/ubuntu/data/University-Release"

    # Augment Images
    prob_flip: float = 0.5  # flipping the sat image and drone image simultaneously

    # Savepath for model checkpoints
    model_path: str = "/home/ubuntu/data/CV-cities_train/university/s2d/second/weather"

    # Eval before training
    zero_shot: bool = True

    # Checkpoint to start from
    # checkpoint_start = None
   #todo(这个是预训练好的模型)
    # contra
    # checkpoint_start_ready="/home/ubuntu/data/CV-cities_train/university/d2s/first/contra/dinov2_vitb14_MixVPR/2026-01-31_154927/weights_e4_0.8132.pth"
    # checkpoint_start="/home/ubuntu/data/CV-cities_train/university/d2s/second/contra/dinov2_vitb14_MixVPR/2026-02-01_135137/weights_e8_0.8021.pth"
    # triplet
    # checkpoint_start_ready="/home/ubuntu/data/CV-cities_train/university/d2s/first/trip/dinov2_vitb14_MixVPR/2026-01-31_100410/weights_e36_0.9370.pth"
    # checkpoint_start="/home/ubuntu/data/CV-cities_train/university/d2s/second/triplet/dinov2_vitb14_MixVPR/2026-02-01_114242/weights_e24_0.9304.pth"
    # Instance
    # checkpoint_start_ready="/home/ubuntu/data/CV-cities_train/university/d2s/first/instance/dinov2_vitb14_MixVPR/2026-01-31_020105/weights_e32_0.9481.pth"
    # checkpoint_start="/home/ubuntu/data/CV-cities_train/university/d2s/second/instance/dinov2_vitb14_MixVPR/2026-01-31_153556/weights_e6_0.9559.pth"
    
    checkpoint_start_ready="/home/ubuntu/data/CV-cities_train/university/s2d/first/weather/dinov2_vitb14_MixVPR/2026-02-01_162010/weights_e4_0.9800.pth"
    checkpoint_start="/home/ubuntu/data/CV-cities_train/university/s2d/second/weather/dinov2_vitb14_MixVPR/2026-02-02_010543/weights_e9_0.9743.pth"
    
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

if config.dataset == 'U1652-D2S':
    config.query_folder_train = '/home/ubuntu/data/University-Release/train/satellite'
    config.gallery_folder_train = '/home/ubuntu/data/University-Release/train/drone'
    config.query_folder_test = '/home/ubuntu/data/University-Release/test/query_drone'
    config.gallery_folder_test = '/home/ubuntu/data/University-Release/test/gallery_satellite'
elif config.dataset == 'U1652-S2D':
    config.query_folder_train = '/home/ubuntu/data/University-Release/train/satellite'
    config.gallery_folder_train = '/home/ubuntu/data/University-Release/train/drone'
    config.query_folder_test = '/home/ubuntu/data/University-Release/test/query_satellite'
    config.gallery_folder_test = '/home/ubuntu/data/University-Release/test/gallery_drone'

if __name__ == '__main__':

    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   time.strftime("%Y-%m-%d_%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.abspath(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    # -----------------------------------------------------------------------------#
    # Model                                                                       #
    # -----------------------------------------------------------------------------#
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    print("\nModel: {}".format(config.model))

    # model = TimmModel(model_name=config.model,
    #                   pretrained=True,
    #                   img_size=config.img_size, backbone_arch=config.backbone_arch, agg_arch=config.agg_arch,
    #                   agg_config=config.agg_config, layer1=config.layer1)
    
    model_first = TimmModel(model_name=config.model,
                      pretrained=True,
                      img_size=config.img_size, backbone_arch=config.backbone_arch, agg_arch=config.agg_arch,
                      agg_config=config.agg_config, layer1=config.layer1,
                      model_type='first' )
    # print(model)
    
    model_state_dict_ready=torch.load(config.checkpoint_start_ready) 
    #这个模型保存时用了save_model方法
    model_first.load_state_dict(model_state_dict_ready, strict=False) 
    
    # print(model_state_dict_ready.keys())
    del model_state_dict_ready
    
    model = TimmModel(model_name=config.model,
                      pretrained=True,
                      img_size=config.img_size, backbone_arch=config.backbone_arch, agg_arch=config.agg_arch,
                      agg_config=config.agg_config, layer1=config.layer1,
                      model_type='second' ,checkpoint_start_ready=config.checkpoint_start_ready)

    data_config = model_first.get_config()
    print(model)

    # data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]

    img_size = (config.img_size, config.img_size)

    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
        model_first.set_grad_checkpointing(True)
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        # model.load_state_dict(model_state_dict, strict=False)
        model.load_state_dict(model_state_dict, strict=False)  

        # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model_first = torch.nn.DataParallel(model_first, device_ids=config.gpu_ids)

    # Model to device   
    model = model.to(config.device)
    model_first = model_first.to(config.device)
    
    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    # -----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    # -----------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

    # Train
    train_dataset = U1652DatasetTrain(query_folder=config.query_folder_train,
                                      gallery_folder=config.gallery_folder_train,
                                      transforms_query=train_sat_transforms,
                                      transforms_gallery=train_drone_transforms,
                                      prob_flip=config.prob_flip,
                                      shuffle_batch_size=config.batch_size,
                                      )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)

    # Reference Satellite Images
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                          mode="query",
                                          transforms=val_transforms,
                                          )

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    # Query Ground Images Test
    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test,
                                            mode="gallery",
                                            transforms=val_transforms,
                                            sample_ids=query_dataset_test.get_sample_ids(),
                                            gallery_n=config.eval_gallery_n,
                                            )

    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                         batch_size=config.batch_size_eval,
                                         num_workers=config.num_workers,
                                         shuffle=False,
                                         pin_memory=True)

    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))

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
        # if config.checkpoint_start is not None:  
        #     model_state_dict = torch.load(config.checkpoint_start)  
            # optimizer.load_state_dict(model_state_dict['optimizer']) 

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
        # if config.checkpoint_start is not None:  
        #     # model_state_dict = torch.load(config.checkpoint_start)  
        #     scheduler.load_state_dict(model_state_dict['scheduler']) 
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
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)

    # # -----------------------------------------------------------------------------#
    # # Shuffle                                                                     #
    # # -----------------------------------------------------------------------------#
    # if config.custom_sampling:
    #     train_dataloader.dataset.shuffle()

    # # -----------------------------------------------------------------------------#
    # # Train                                                                       #
    # # --------------------------------0--------------------------------------------#
    # start_epoch = 0
    # best_score = 0
    # # if config.checkpoint_start is not None:
    # #     start_epoch=model_state_dict['epoch']+1
    # #     del model_state_dict   
    # for epoch in range(1, config.epochs + 1):

    #     print("\n{}[{}/Epoch: {}]{}".format(30*"-",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),  epoch, 30*"-"))

    #     train_loss = train(config,
    #                        model,
    #                        model_first,
    #                        dataloader=train_dataloader,
    #                        loss_function=loss_function,
    #                        optimizer=optimizer,
    #                        scheduler=scheduler,
    #                        scaler=scaler)

    #     print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
    #                                                                train_loss,
    #                                                                optimizer.param_groups[0]['lr']))

    #     # evaluate
    #     if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:

    #         print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))

    #         r1_test = evaluate(config=config,
    #                            model=model,
    #                            model_first=model_first,
    #                            query_loader=query_dataloader_test,
    #                            gallery_loader=gallery_dataloader_test,
    #                            ranks=[1, 5, 10],
    #                            step_size=1000,
    #                            cleanup=True)

    #         if r1_test > best_score:

    #             best_score = r1_test

    #             if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
    #                 torch.save(model.module.state_dict(),
    #                            '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
    #             else:
    #                 torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
    #             # save_model('{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test),model,optimizer,epoch,r1_test,scheduler)    

    #     if config.custom_sampling:
    #         train_dataloader.dataset.shuffle()

    # if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
    #     torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    # else:
    #     torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))

    # # save_model('{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test),model,optimizer,epoch,r1_test,scheduler)  