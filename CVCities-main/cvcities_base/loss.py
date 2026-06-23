import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss  
 


# class TripletLoss(nn.Module):
#     def __init__(self, margin=0.3):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#         self.ranking_loss = nn.MarginRankingLoss(margin=margin)

#     def forward(self, image_features1, image_features2, logit_scale=None):
#         # 1. 归一化特征 (通常度量学习推荐先归一化)
#         image_features1 = F.normalize(image_features1, dim=-1)
#         image_features2 = F.normalize(image_features2, dim=-1)

#         # 2. 计算欧氏距离矩阵 (Batch size x Batch size)
#         # dist[i][j] 表示 feature1[i] 和 feature2[j] 之间的距离
#         dist_mat = torch.cdist(image_features1, image_features2, p=2)

#         # 3. 提取正样本对距离 (对角线元素: feature1[i] <-> feature2[i])
#         # shape: [batch_size]
#         pos_dist = torch.diag(dist_mat)

#         # 4. 提取难负样本对距离 (Hard Mining)
#         # 也就是在每一行中，找到除对角线外最小的那个距离
#         # 我们可以先将对角线(正样本)设为无穷大，防止被选为最小
#         mask = torch.eye(dist_mat.size(0), device=dist_mat.device).bool()
#         dist_mat_neg = dist_mat.clone()
#         dist_mat_neg[mask] = float('inf') 
        
#         # 找到每一行最小的负样本距离, shape: [batch_size]
#         neg_dist, _ = torch.min(dist_mat_neg, dim=1)

#         # 5. 计算损失
#         # y=1 表示第一个输入应该比第二个输入大（在这里反过来，我们希望 neg_dist > pos_dist）
#         # MarginRankingLoss: max(0, -y * (x1 - x2) + margin)
#         # 这里 y = -1, 所以 loss = max(0, pos_dist - neg_dist + margin)
#         y = -torch.ones_like(pos_dist)
#         loss = self.ranking_loss(neg_dist, pos_dist, y)
        
#         return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, image_features1, image_features2, labels,logit_scale=None):
        # inputs:
        #   image_features1: [batch_size, dim] (Query)
        #   image_features2: [batch_size, dim] (Gallery)
        #   labels: [batch_size] (Class Labels)
        
        # 1. 归一化
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        # 2. 计算欧氏距离矩阵
        dist_mat = torch.cdist(image_features1, image_features2, p=2)

        # 3. 提取正样本对距离 (对角线)
        pos_dist = torch.diag(dist_mat)

        # 4. 提取难负样本 (Hard Mining with Label Mask)
        # 关键修改：我们要屏蔽掉所有 label 相同的样本，而不仅仅是对角线
        
        # 创建 mask：如果 label[i] == label[j]，则 mask[i][j] 为 True
        # shape: [batch_size, batch_size]
        is_pos = labels.expand(len(labels), len(labels)).eq(labels.expand(len(labels), len(labels)).t())
        
        # 复制一份距离矩阵用于寻找负样本
        dist_mat_neg = dist_mat.clone()
        
        # 将所有正样本对（包括对角线和同类不同图）的距离设为无穷大
        # 这样 min() 就永远不会选中它们作为负样本了
        dist_mat_neg[is_pos] = float('inf')

        # 找到每一行中，非同类的最小距离（真正的 Hard Negative）
        neg_dist, _ = torch.min(dist_mat_neg, dim=1)

        # 5. 计算损失
        y = -torch.ones_like(pos_dist)
        loss = self.ranking_loss(pos_dist, neg_dist, y)
        
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, image_features1, image_features2, labels,logit_scale=None):
        # inputs:
        #   image_features1, image_features2: [Batch, Dim]
        #   labels: [Batch]  <-- 必须传入真实标签 (如: 0, 0, 1, 5...)
        
        # 1. 归一化 (保持不变，这是对的)
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        # 2. 计算欧氏距离矩阵
        dist_mat = torch.cdist(image_features1, image_features2, p=2)
        
        # 3. 生成真实的标签矩阵 Mask
        # 如果 label[i] == label[j]，则 mask[i][j] = 1 (正样本)
        # 否则为 0 (负样本)
        batch_size = image_features1.size(0)
        labels = labels.view(-1, 1) # [B, 1]
        mask = torch.eq(labels, labels.T).float().to(dist_mat.device) # [B, B]

        # 4. 计算正样本损失 (minimize distance^2)
        # 只计算 mask 为 1 的部分
        # 避免除以 0，加一个极小值 eps
        pos_dist_sq = mask * torch.pow(dist_mat, 2)
        pos_loss = pos_dist_sq.sum() / (mask.sum() + 1e-8)

        # 5. 计算负样本损失 (minimize max(0, margin - distance)^2)
        # 只计算 mask 为 0 的部分
        neg_dist = torch.clamp(self.margin - dist_mat, min=0.0)
        neg_dist_sq = (1 - mask) * torch.pow(neg_dist, 2)
        
        # 优化建议：只统计那些 loss > 0 的有效负样本的数量来进行平均，而不是除以所有负样本
        # 这样可以聚焦由于 Hard Negative 产生的梯度
        valid_negatives = ((1 - mask) * (neg_dist > 0.0).float()).sum()
        neg_loss = neg_dist_sq.sum() / (valid_negatives + 1e-8)

        # 6. 总损失
        loss = pos_loss + neg_loss
        return loss
    
class InstanceLoss(nn.Module):
    def __init__(self):
        super(InstanceLoss, self).__init__()
        # 定义一个共享的分类器层 (Linear: Feature Dim -> Num Classes)
        # bias=False 是 ID Loss 的常见做法，类似于 ArcFace 等，但加 bias 也可以
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, image_features1, image_features2, targets, logit_scale=None):
        # 注意：这里的 targets 是必须的输入，代表样本的真实类别ID (比如 tensor([0, 1, 2...]))
        
        # 1. 直接通过分类器 (不做归一化也可以，但做归一化收敛更稳)
        # 论文中通常对特征做 bottleneck 或者直接进分类器
        # 这里假设输入还是原始特征
   
        
        # 2. 计算交叉熵损失
        loss1 = self.loss_function(image_features1, targets)
        loss2 = self.loss_function(image_features2, targets)
        
        # 3. 总损失为两部分之和
        loss = loss1 + loss2
        
        return loss