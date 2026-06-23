import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.amp import autocast
import torch.nn.functional as F

def cam_loss(output):
    loss=0
    print(output.shape)
    for i in range(len(output.size(0))):
        loss=loss+output[i]
    return loss

def train(train_config, model, model_first,dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    if model ==None:
        model_first.train()
    else:
        model.train()
        model_first.eval()
    # model_first.train()
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query, reference, ids, label in bar:
        
        if scaler:
            with torch.amp.autocast(device_type='cuda'):
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                label = label.to(train_config.device)
                if model == None:
                    # features1_ready,_=model_first(query,type=False,test='instance')
                    # features2_ready,mask=model_first(reference,type=True,test='instance')
                    features1_ready,_=model_first(query,type=False)
                    features2_ready,mask=model_first(reference,type=True)
                #our
                # Forward pass
                # # features1, features2 = model_first(query, reference)
                # features1_ready,_=model_first(query,type=False)
                # # # 0--------------------------------------
                # features2_ready,mask=model_first(reference,type=True)
                
                
                # features1=model(query,type=False)
                # features2=model(reference,type=True,mask=mask)
                
                #param-matched testing
                #屏蔽掉了
                features1_ready,_=model_first(query,type=False)
                # # # 0--------------------------------------
                features2_ready,mask=model_first(reference,type=True)
                
                
                features1=model(query,type=False)
                features2=model(reference,type=True,mask=mask)
                
                # instance
                # features1_ready,_=model_first(query,type=False,test="instance")
                # # # # 0--------------------------------------
                # features2_ready,mask=model_first(reference,type=True,test="instance")
                
                
                # features1=model(query,type=False,test="instance")
                # features2=model(reference,type=True,mask=mask,test="instance")
                
                
                # # 以下修改为对卫星图像进行处理
                # # features1, features2 = model_first(query, reference)
                # features1_ready,_=model_first(reference,type=False)
                # # # 0--------------------------------------
                # features2_ready,mask=model_first(query,type=True)
                
                
                # features1=model(reference,type=False)
                # features2=model(query,type=True,mask=mask)
                
                # if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                #     loss = loss_function(features1, features2, model.module.logit_scale.exp())
                # else:
                #     loss = loss_function(features1, features2, model.logit_scale.exp()) 
                if model == None:
                    # if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    #     loss = loss_function(features1_ready, features2_ready, label,model_first.module.logit_scale.exp())
                    # else:
                    #     loss = loss_function(features1_ready, features2_ready,label,model_first.logit_scale.exp()) 
                    if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                        loss = loss_function(features1_ready, features2_ready, model_first.module.logit_scale.exp())
                    else:
                        loss = loss_function(features1_ready, features2_ready,model_first.logit_scale.exp()) 
                else:
                    if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                        loss = loss_function(features1, features2, model.module.logit_scale1.exp())
                    else:
                        loss = loss_function(features1, features2,model.logit_scale1.exp()) 
                    # if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    #     loss = loss_function(features1, features2, label,model.module.logit_scale1.exp())
                    # else:
                    #     loss = loss_function(features1, features2, label,model.logit_scale1.exp()) 
                    
                    if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                        loss1 = loss_function(features1, features2_ready, model.module.logit_scale2.exp())
                    else:
                        loss1 = loss_function(features1, features2_ready,model.logit_scale2.exp()) 
                        
                    if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                        loss2 = loss_function(features1_ready, features2, model.module.logit_scale3.exp())
                    else:
                        loss2 = loss_function(features1_ready, features2,model.logit_scale3.exp())                
                                                                                                                                                                        
                    if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                        loss_union = loss_function(features1_ready+features1,features2_ready+features2, model.module.logit_scale4.exp())
                    else:
                        loss_union = loss_function(features1_ready+features1,features2_ready+features2,  model.logit_scale4.exp()) 
                
                    # if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    #     loss1 = loss_function(features1, features2_ready, label,model.module.logit_scale2.exp())
                    # else:
                    #     loss1 = loss_function(features1, features2_ready,label,model.logit_scale2.exp()) 
                        
                    # if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    #     loss2 = loss_function(features1_ready, features2,label, model.module.logit_scale3.exp())
                    # else:
                    #     loss2 = loss_function(features1_ready, features2,label,model.logit_scale3.exp())                
                                                                                                                                                                        
                    # if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    #     loss_union = loss_function(features1_ready+features1,features2_ready+features2,label, model.module.logit_scale4.exp())
                    # else:
                    #     loss_union = loss_function(features1_ready+features1,features2_ready+features2, label, model.logit_scale4.exp()) 
                
                # loss_all=loss
                # loss_all=loss+loss_union
                loss_all=loss+loss1+loss2+loss_union
                
                # loss_all=cam_loss(features1_ready)
                
                losses.update(loss_all.item())
                # losses.update(loss.item())
                  
            # scaler.scale(loss).backward()
            scaler.scale(loss_all).backward()
            
            # Gradient clipping 
            if model ==None:
                if train_config.clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_value_(model_first.parameters(), train_config.clip_grad)
            else:  
                if train_config.clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()

        if train_config.verbose:
            
            # monitor = {"loss": "{:.4f}".format(loss.item()),
            #            "loss_avg": "{:.4f}".format(losses.avg),
            #            "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            monitor = {
                                "all": "{:.6f}".format(loss_all.item()),
                                "avg": "{:.6f}".format(losses.avg),
                                    "l": "{:.2f}".format(loss.item()),
                                    "l12": "{:.2f}".format(loss1.item()+loss2.item()),
                                    "l2": "{:.2f}".format(loss2.item()),
                                    #  "w": "{:.4f}".format(loss_weight.weight.item()),
                                    #  "w": "{:.2f}".format((loss_all-loss).item()),
                                    "u": "{:.2f}".format(loss_union.item()),
                                "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    model.eval()
    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    # output_shape = model(dummy_input).shape[1:]
    out,_=model(dummy_input)
    output_shape=out.shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size)
    img_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            img_feature,_ = model(img)
            # img_feature = model(img,type=False,mask=None)

            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids

#用于cal_sim:ref   单独特征，需要mask,所以传入model_first  
def predict2ref(train_config, model, dataloader,model_first):
    model.eval()
    model_first.eval()
    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    output_shape = model(dummy_input).shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size)
    img_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)

    with torch.no_grad(), autocast():
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            
            img_feature_first,mask= model_first(img,type=True)
            img_feature = model(img,type=True,mask=mask)

            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids

#用于cal_sim:query   单独特征，不需要mask(没有mask),所以不传入model_first  
def predict2query(train_config, model, dataloader):
    model.eval()
    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    output_shape = model(dummy_input).shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size)
    img_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)

    with torch.no_grad(), autocast():
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            img_feature = model(img,type=False)

            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids

#用于evaluate:ref   联合特征，不使用mask
def predict_double2ref(train_config, model, dataloader,model_first):
    model.eval()
    model_first.eval()
    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    output_shape = model(dummy_input).shape[1:]
    # print(output_shape)
    # Pre-allocate memory for efficiency (assuming fixed batch size)
    img_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            
            img_feature_first,_ = model_first(img,type=True)
            img_feature = model(img,type=True,mask=None)

            img_feature=img_feature+img_feature_first
            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids

#用于evaluate:query   联合特征，没有mask
def predict_double2query(train_config, model, dataloader,model_first):
    model.eval()
    
    model_first.eval()
    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    output_shape = model(dummy_input).shape[1:]
    # print(output_shape)
    # Pre-allocate memory for efficiency (assuming fixed batch size)
    img_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            # print(type(img))
            img_feature_first,_ = model_first(img,type=False)
            img_feature = model(img,type=False)

            img_feature=img_feature+img_feature_first
            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids

#用于evaluate:query(由于VIGOR的query特殊,所以单独设置)   联合特征，不使用mask
def predict_double2query_Vigor(train_config, model, dataloader,model_first):
    model.eval()
    model_first.eval()

    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    output_shape = model(dummy_input).shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size and ids_current shape)
    total_samples = len(dataloader.dataset)
    img_features = torch.zeros((total_samples, *output_shape), dtype=torch.float32, device=train_config.device)
    # Assuming each id_current has 4 elements, adjust the dimension for ids
    ids = torch.zeros((total_samples, 4), dtype=torch.long, device=train_config.device)  # 修改为二维张量以匹配ids_current的形状

    with torch.no_grad(), autocast():
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            
            img_feature_first,_ = model_first(img,type=False)
            img_feature = model(img,type=False)

            img_feature=img_feature+img_feature_first

            # Normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            # Directly assign the 2D ids_current to the corresponding slice in ids
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids


#用于cal_sim:query(由于VIGOR的query特殊,所以单独设置)   单独特征，不使用mask，所以不传入model_first
def predict2query_Vigor(train_config, model, dataloader):
    model.eval()

    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    output_shape = model(dummy_input).shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size and ids_current shape)
    total_samples = len(dataloader.dataset)
    img_features = torch.zeros((total_samples, *output_shape), dtype=torch.float32, device=train_config.device)
    # Assuming each id_current has 4 elements, adjust the dimension for ids
    ids = torch.zeros((total_samples, 4), dtype=torch.long, device=train_config.device)  # 修改为二维张量以匹配ids_current的形状

    with torch.no_grad(), autocast():
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            img_feature = model(img,type=False)

            # Normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            # Directly assign the 2D ids_current to the corresponding slice in ids
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids

# def predict_vigor(train_config, model, dataloader):
    # model.eval()

    # # Get output shape from a dummy input
    # dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    # output_shape = model(dummy_input).shape[1:]

    # # Pre-allocate memory for efficiency (assuming fixed batch size and ids_current shape)
    # total_samples = len(dataloader.dataset)
    # img_features = torch.zeros((total_samples, *output_shape), dtype=torch.float32, device=train_config.device)
    # # Assuming each id_current has 4 elements, adjust the dimension for ids
    # ids = torch.zeros((total_samples, 4), dtype=torch.long, device=train_config.device)  # 修改为二维张量以匹配ids_current的形状

    # with torch.no_grad(), autocast():
    #     for i, (img, ids_current) in enumerate(tqdm(dataloader)):
    #         img = img.to(train_config.device)
    #         img_feature = model(img)

    #         # Normalize is calculated in fp32
    #         if train_config.normalize_features:
    #             img_feature = F.normalize(img_feature, dim=-1)

    #         img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
    #         # Directly assign the 2D ids_current to the corresponding slice in ids
    #         ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    # return img_features, ids

# def predict(train_config, model, dataloader):
#     model.eval()
#
#     # wait before starting progress bar
#     time.sleep(0.1)
#
#     if train_config.verbose:
#         bar = tqdm(dataloader, total=len(dataloader))
#     else:
#         bar = dataloader
#
#     img_features_list = []
#
#     ids_list = []
#     with torch.no_grad():
#
#         for img, ids in bar:
#
#             ids_list.append(ids)
#
#             with torch.amp.autocast(device_type='cuda'):
#
#                 img = img.to(train_config.device)
#                 img_feature = model(img)
#
#                 # normalize is calculated in fp32
#                 if train_config.normalize_features:
#                     img_feature = F.normalize(img_feature, dim=-1)
#
#             # save features in fp32 for sim calculation
#             img_features_list.append(img_feature.to(torch.float32))
#
#         # keep Features on GPU
#         img_features = torch.cat(img_features_list, dim=0)
#         ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
#
#     if train_config.verbose:
#         bar.close()
#
#     return img_features, ids_list