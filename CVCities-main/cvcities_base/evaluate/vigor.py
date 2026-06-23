import time
import cv2
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict, predict_double2ref,predict_double2query_Vigor,predict2ref,predict2query_Vigor


def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader, 
             model_first=None,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    # reference_features, reference_labels = predict(config, model, reference_dataloader)
    # query_features, query_labels = predict_vigor(config, model, query_dataloader)
    
    reference_features, reference_labels = predict_double2ref(config, model, reference_dataloader,model_first=model_first)
    query_features, query_labels = predict_double2query_Vigor(config, model, query_dataloader,model_first=model_first)
    # reference_features=torch.load("/home/ubuntu/data/reference_features.pth")
    # reference_labels=torch.load("/home/ubuntu/data/reference_labels.pth")
    # query_features=torch.load("/home/ubuntu/data/query_features.pth")
    # query_labels=torch.load("/home/ubuntu/data/query_labels.pth")

    # torch.save(reference_features,"/home/ubuntu/data/reference_features.pth")
    # torch.save(reference_labels,"/home/ubuntu/data/reference_labels.pth")
    # torch.save(query_features,"/home/ubuntu/data/query_features.pth")
    # torch.save(query_labels,"/home/ubuntu/data/query_labels.pth")
    
    print("Compute Scores:")
    r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, reference_dataloader=reference_dataloader,query_dataloader=query_dataloader,step_size=step_size, ranks=ranks) 

    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1


def calc_sim(config,
                        model,
                        reference_dataloader,
                        query_dataloader, 
                        model_first=None,
                        ranks=[1, 5, 10],
                        step_size=1000,
                        cleanup=True):
    
    
    print("\nExtract Features:")
    # reference_features, reference_labels = predict(config, model, reference_dataloader) 
    # query_features, query_labels = predict_vigor(config, model, query_dataloader)
    
    reference_features, reference_labels = predict2ref(config, model, reference_dataloader,model_first=model_first)
    query_features, query_labels = predict2query_Vigor(config, model, query_dataloader)
    
    print("Compute Scores Train:")
    r1 =  calculate_scores_train(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    
    near_dict = calculate_nearest(query_features=query_features,
                                  reference_features=reference_features,
                                  query_labels=query_labels,
                                  reference_labels=reference_labels,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)
            
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1, near_dict



def calculate_scores(query_features, reference_features, query_labels, reference_labels, reference_dataloader=None,query_dataloader=None,step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    hit_rate = 0.0
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i][0]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
        
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
        # mask for semi pos
        mask = torch.ones(R)
        for near_pos in query_labels_np[i][1:]:
            mask[ref2index[near_pos]] = 0
        
        # calculate hit rate
        hit = (higher_sim * mask).sum()
        if hit < 1:
            hit_rate += 1.0
                
    
    results = results/ Q * 100.
    hit_rate = hit_rate / Q * 100
    
    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))
    string.append('Hit_Rate: {:.4f}'.format(hit_rate))             
        
    print(' - '.join(string)) 
    
    



# #==================testing=======================
#     similarity=torch.load("/home/ubuntu/data/similarity.pth") 
#     # torch.save(similarity,"/home/ubuntu/data/similarity.pth")
#     print("开始探寻")
# #     #TODO找到最相近的五个图像
#     topk_scores, topk_ids = torch.topk(similarity, k=5, dim=1,sorted=True)

#     # print(topk_ids.shape)    #torch.Size([52605, 5])
#     # print(topk_ids[0])
#     topk_references = []
    
#     for i in range(len(topk_ids)):
#         topk_references.append(reference_labels[topk_ids[i,:]])
#         # print(topk_references)
    
#     topk_references = torch.stack(topk_references, dim=0)
#     # print(topk_references[0])
#     # print(topk_references.shape)  #torch.Size([52605, 5])
#     nearest_dict = dict()
#     # print(query_labels_main.shape)
#     # print(query_labels_main[0])
#     # print(query_labels_main[0].to('cpu').tolist() )#13355
#     idx2sat_list=[]
# #     #TODO 不同街景图像可能对应同一个卫星图像 query_labels_main[i]存的是主卫星图像，怎么找街景图像呢    
# #     #TODO 创建的字典的键为sat sat1 sat2 sat3的合集 然后找街景图像的话，要全都完全匹配才是街景图像
#     for i in range(len(topk_references)):
        
#         nearest = topk_references[i]
#         # nearest_dict[query_labels_main[i].to('cpu').tolist()] = list(nearest)
#         nearest_dict[i] = list(nearest)                 #储存序号到每个街景图像对应的最困难的五个
#         idx2sat_list.append(query_labels[i].to('cpu').tolist())   #储存序号到街景图像对应的sat sat1 sat2 sat3
#         # 创建一个新的字典来存储街景图像路径到卫星图像路径列表的映射
#     # print(nearest_dict)
#     new_path_dict = {}
    
#     num=24911
#     save_path1='/home/ubuntu/Self_CVCties/show_5/ground.jpg'
#     save_path2='/home/ubuntu/Self_CVCties/show_5/sat1.jpg'
#     save_path3='/home/ubuntu/Self_CVCties/show_5/sat2.jpg'
#     save_path4='/home/ubuntu/Self_CVCties/show_5/sat3.jpg'
#     save_path5='/home/ubuntu/Self_CVCties/show_5/sat4.jpg'
#     save_path6='/home/ubuntu/Self_CVCties/show_5/sat5.jpg'
#     save_path7='/home/ubuntu/Self_CVCties/show_5/sat_true.jpg'
    
#     main_sat_list=list(nearest_dict.keys())
#     main_sat_index=main_sat_list[num]
#     sat_indices=nearest_dict[main_sat_index]
    
#     sat_list=idx2sat_list[main_sat_index]
#     # print(sat_list)
#     #todo(找街景图像的代码好像有问题)
#     street_image_path = query_dataloader.dataset.find_single_street_image_for_satellite(sat_list)
#     # print(street_image_path)
#     # 获取每个卫星图像的路径
#     satellite_image_paths = [reference_dataloader.dataset.get_imageSat_path_by_label(sat_index.item()) for sat_index in sat_indices]
#     satellite_image_paths.append(reference_dataloader.dataset.get_imageSat_path_by_label(sat_list[0]))
#     # 添加到新的字典中
#     #TODO(存在个别的两个图片位置完全一样，细节不一样，这个怎么处理？)
#     new_path_dict[str(street_image_path)] = satellite_image_paths    


#     # 遍历选定的索引并绘制图像
#     index=0
#     for key in new_path_dict:
#         # key = list(new_path_dict.keys())[index]
#         values = new_path_dict[key]
        
#         ground_img=cv2.imread(key)
#         cv2.imwrite(save_path1,ground_img)
        
#         sat_path=values[0]
#         sat_img=cv2.imread(sat_path)
#         cv2.imwrite(save_path2,sat_img)    
        
#         sat_path=values[1]
#         sat_img=cv2.imread(sat_path)
#         cv2.imwrite(save_path3,sat_img) 
        
#         sat_path=values[2]
#         sat_img=cv2.imread(sat_path)
#         cv2.imwrite(save_path4,sat_img) 
        
#         sat_path=values[3]
#         sat_img=cv2.imread(sat_path)
#         cv2.imwrite(save_path5,sat_img) 
        
#         sat_path=values[4]
#         sat_img=cv2.imread(sat_path)
#         cv2.imwrite(save_path6,sat_img) 
        
#         sat_path=values[5]
#         sat_img=cv2.imread(sat_path)
#         cv2.imwrite(save_path7,sat_img)    
        



    return results[0]

def calculate_scores_train(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    query_labels_np = query_labels[:,0].cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
        
    results = results/ Q * 100.

    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))           
        
    print(' - '.join(string)) 

    return results[0]
   

def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64, step_size=1000):

    query_labels = query_labels[:,0]
    
    Q = len(query_features)
    
    steps = Q // step_size + 1
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)


    # there might be more ground views for same sat view
    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range+2, dim=1)


    topk_references = []
    
    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i,:]])
    
    topk_references = torch.stack(topk_references, dim=0)

     
    # mask for ids without gt hits
    mask = topk_references != query_labels.unsqueeze(1)
    
    
    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()
    

    # dict that only stores ids where similiarity higher than the lowes gt hit score
    nearest_dict = dict()
    
    for i in range(len(topk_references)):
        
        nearest = topk_references[i][mask[i]][:neighbour_range]
    
        nearest_dict[query_labels[i].item()] = list(nearest)
    

    return nearest_dict
