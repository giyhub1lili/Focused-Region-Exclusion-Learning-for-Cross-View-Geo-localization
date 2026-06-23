import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7,6"

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
from cvcities_base.loss import InfoNCE, InstanceLoss,ContrastiveLoss,TripletLoss
from cvcities_base.model import TimmModel


import torch
import torch.nn as nn

import torch
from collections import defaultdict

# def unwrap_model(model):
#     return model.module if hasattr(model, "module") else model

# def count_params_m(model, trainable_only=False):
#     model = unwrap_model(model)
#     if trainable_only:
#         n = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     else:
#         n = sum(p.numel() for p in model.parameters())
#     return n / 1e6

# def find_dinov2_backbone(model):
#     """
#     找到真正的 DINOv2 主干模块（具有 patch_embed 和 blocks）
#     """
#     model = unwrap_model(model)

#     if hasattr(model, "patch_embed") and hasattr(model, "blocks"):
#         return model, "<root>"

#     for name, module in model.named_modules():
#         if hasattr(module, "patch_embed") and hasattr(module, "blocks"):
#             return module, name

#         if hasattr(module, "dino_model"):
#             dm = module.dino_model
#             if hasattr(dm, "patch_embed") and hasattr(dm, "blocks"):
#                 path = f"{name}.dino_model" if name else "dino_model"
#                 return dm, path

#     raise AttributeError("没有找到带有 patch_embed 和 blocks 的 DINOv2 主干")

# def print_dinov2_block_status(model, title="model"):
#     """
#     只打印 DINOv2 backbone 的 patch_embed 和 blocks 状态
#     """
#     model = unwrap_model(model)
#     backbone, path = find_dinov2_backbone(model)

#     print(f"\n[{title}] DINO backbone path: {path}")

#     patch_flag = any(p.requires_grad for p in backbone.patch_embed.parameters())
#     patch_params = sum(p.numel() for p in backbone.patch_embed.parameters()) / 1e6
#     print(f"[{title}] patch_embed: trainable={patch_flag}, params={patch_params:.4f} M")

#     for i, blk in enumerate(backbone.blocks):
#         flag = any(p.requires_grad for p in blk.parameters())
#         blk_params = sum(p.numel() for p in blk.parameters()) / 1e6
#         print(f"[{title}] block {i:02d}: trainable={flag}, params={blk_params:.4f} M")

# def print_trainable_params_full(model, title="model"):
#     """
#     打印所有 requires_grad=True 的参数，逐参数张量输出
#     """
#     model = unwrap_model(model)
#     total = 0.0

#     print(f"\n===== {title}: full trainable parameter list =====")
#     for name, p in model.named_parameters():
#         if p.requires_grad:
#             pm = p.numel() / 1e6
#             total += pm
#             print(f"{name:100s} {list(p.shape)!s:28s} {pm:10.6f} M")

#     print(f"[{title}] trainable total = {total:.4f} M")

# def print_trainable_params_grouped(model, title="model", level=2):
#     """
#     按名称前缀聚合 trainable params。
#     level=2 例如:
#       model.backbone
#       model.agg
#       model.logit_scale1
#     level=3 例如:
#       model.backbone.dino_model
#       model.agg.mix
#     """
#     model = unwrap_model(model)
#     groups = defaultdict(float)

#     for name, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         parts = name.split(".")
#         prefix = ".".join(parts[:level]) if len(parts) >= level else name
#         groups[prefix] += p.numel() / 1e6

#     print(f"\n===== {title}: grouped trainable params (level={level}) =====")
#     for k, v in sorted(groups.items(), key=lambda x: -x[1]):
#         print(f"{k:60s} {v:10.6f} M")

# def print_trainable_vs_total_by_top_module(model, title="model"):
#     """
#     按顶层子模块统计 total params 和 trainable params
#     """
#     model = unwrap_model(model)

#     print(f"\n===== {title}: total vs trainable by top submodule =====")
#     for child_name, child in model.named_children():
#         total = sum(p.numel() for p in child.parameters()) / 1e6
#         trainable = sum(p.numel() for p in child.parameters() if p.requires_grad) / 1e6
#         print(f"{child_name:40s} total={total:10.6f} M   trainable={trainable:10.6f} M")

#     # 顶层没有被子模块覆盖到的“裸参数”，例如 logit_scale
#     child_param_ids = set()
#     for _, child in model.named_children():
#         for p in child.parameters():
#             child_param_ids.add(id(p))

#     root_total = 0.0
#     root_trainable = 0.0
#     for name, p in model.named_parameters(recurse=False):
#         pm = p.numel() / 1e6
#         root_total += pm
#         if p.requires_grad:
#             root_trainable += pm

#     print(f"{'<root_params>':40s} total={root_total:10.6f} M   trainable={root_trainable:10.6f} M")

# def summarize_stage2_detailed(model_first, model):
#     """
#     适用于你当前这个 stage2 脚本:
#     - model_first 已整体冻结
#     - model 是第二阶段训练模型
#     """
#     m1 = unwrap_model(model_first)
#     m2 = unwrap_model(model)

#     stage2_trainable_m = count_params_m(m2, trainable_only=True)+count_params_m(m1, trainable_only=True)
#     inference_params_m = count_params_m(m1, trainable_only=False) + \
#                          count_params_m(m2, trainable_only=False)

#     print("\n===== Stage-2 Summary =====")
#     print(f"Stage-2 Trainable Params (M): {stage2_trainable_m:.4f}")
#     print(f"Inference-time Params (M):    {inference_params_m:.4f}")

#     return {
#         "stage2_trainable_m": stage2_trainable_m,
#         "inference_params_m": inference_params_m,
#     }

import os
import torch

# =========================================================
# 基础工具
# =========================================================

# def unwrap_model(model):
#     return model.module if hasattr(model, "module") else model

# def count_params_m(model, trainable_only=False):
#     """
#     统计参数量，单位 M（百万）

#     Params(M) = sum_i numel(p_i) / 1e6

#     其中：
#     - p_i: 第 i 个参数张量
#     - numel(p_i): 参数张量元素个数
#     """
#     model = unwrap_model(model)
#     if trainable_only:
#         return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
#     return sum(p.numel() for p in model.parameters()) / 1e6

# def count_direct_params_m(module, trainable_only=False):
#     """
#     只统计当前模块自己直接挂载的参数，不递归到子模块
#     """
#     if trainable_only:
#         return sum(p.numel() for _, p in module.named_parameters(recurse=False) if p.requires_grad) / 1e6
#     return sum(p.numel() for _, p in module.named_parameters(recurse=False)) / 1e6

# def find_dinov2_backbone(model):
#     """
#     递归查找真正的 DINOv2 主干
#     条件：模块具有 patch_embed 和 blocks 两个属性

#     返回：
#     - backbone_module
#     - backbone_path（字符串路径）
#     """
#     model = unwrap_model(model)

#     if hasattr(model, "patch_embed") and hasattr(model, "blocks"):
#         return model, "<root>"

#     for name, module in model.named_modules():
#         if hasattr(module, "patch_embed") and hasattr(module, "blocks"):
#             return module, name

#         if hasattr(module, "dino_model"):
#             dm = module.dino_model
#             if hasattr(dm, "patch_embed") and hasattr(dm, "blocks"):
#                 path = f"{name}.dino_model" if name else "dino_model"
#                 return dm, path

#     raise AttributeError("没有找到带有 patch_embed 和 blocks 的 DINOv2 主干")

# # =========================================================
# # 构造报告文本，不在函数内部直接 print，避免输出被冲乱
# # =========================================================

# def build_dino_block_report(model, title="model"):
#     lines = []
#     backbone, path = find_dinov2_backbone(model)

#     lines.append(f"[{title}] DINO backbone path: {path}")

#     patch_trainable = any(p.requires_grad for p in backbone.patch_embed.parameters())
#     patch_params = sum(p.numel() for p in backbone.patch_embed.parameters()) / 1e6
#     lines.append(
#         f"[{title}] patch_embed: trainable={patch_trainable}, params={patch_params:.4f} M"
#     )

#     trainable_block_count = 0
#     trainable_block_total = 0.0

#     for i, blk in enumerate(backbone.blocks):
#         flag = any(p.requires_grad for p in blk.parameters())
#         blk_params = sum(p.numel() for p in blk.parameters()) / 1e6
#         if flag:
#             trainable_block_count += 1
#             trainable_block_total += blk_params
#         lines.append(
#             f"[{title}] block {i:02d}: trainable={flag}, params={blk_params:.4f} M"
#         )

#     lines.append(f"[{title}] trainable DINO block count = {trainable_block_count}")
#     lines.append(f"[{title}] trainable DINO block total = {trainable_block_total:.4f} M")
#     return lines

# def build_module_tree_report(model, title="model", max_depth=4):
#     """
#     递归输出模块树：
#     - 按注册顺序
#     - 不排序
#     - 每个模块给出 total/trainable/direct 参数量
#     """
#     model = unwrap_model(model)
#     lines = []
#     lines.append(f"===== {title}: module tree breakdown (registration order) =====")

#     def _walk(module, name, depth):
#         indent = "  " * depth
#         total = sum(p.numel() for p in module.parameters()) / 1e6
#         trainable = sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6
#         direct_total = count_direct_params_m(module, trainable_only=False)
#         direct_trainable = count_direct_params_m(module, trainable_only=True)

#         lines.append(
#             f"{indent}{name}: total={total:.6f} M, "
#             f"trainable={trainable:.6f} M, "
#             f"direct={direct_total:.6f} M, "
#             f"direct_trainable={direct_trainable:.6f} M"
#         )

#         if depth >= max_depth:
#             return

#         for child_name, child in module.named_children():
#             _walk(child, child_name, depth + 1)

#     _walk(model, "<root>", 0)
#     return lines

# def build_root_param_report(model, title="model"):
#     """
#     输出顶层裸参数，例如 logit_scale 这类不挂在子模块里的参数
#     """
#     model = unwrap_model(model)
#     lines = []
#     lines.append(f"===== {title}: root direct params =====")

#     found = False
#     for name, p in model.named_parameters(recurse=False):
#         found = True
#         pm = p.numel() / 1e6
#         lines.append(
#             f"{name}: shape={list(p.shape)}, trainable={p.requires_grad}, params={pm:.6f} M"
#         )

#     if not found:
#         lines.append("<none>")

#     return lines

# def build_keyword_report(model, keywords=("agg", "aggregator", "logit", "head", "norm"), title="model", trainable_only=False):
#     """
#     按关键词筛选参数，不排序，按 named_parameters 的原始顺序输出
#     """
#     model = unwrap_model(model)
#     lines = []
#     mode = "trainable" if trainable_only else "all"
#     lines.append(f"===== {title}: keyword-matched {mode} params (registration order) =====")

#     found = False
#     total = 0.0
#     for name, p in model.named_parameters():
#         if trainable_only and (not p.requires_grad):
#             continue
#         if any(k in name.lower() for k in keywords):
#             found = True
#             pm = p.numel() / 1e6
#             total += pm
#             lines.append(
#                 f"{name}: shape={list(p.shape)}, trainable={p.requires_grad}, params={pm:.6f} M"
#             )

#     if not found:
#         lines.append("<none>")

#     lines.append(f"[{title}] keyword-matched total = {total:.6f} M")
#     return lines

# def build_full_param_report(model, title="model", trainable_only=False):
#     """
#     全量逐参数输出，不排序，按 named_parameters 的原始顺序
#     输出会很长，只建议写文件，不建议总是打印到终端
#     """
#     model = unwrap_model(model)
#     lines = []
#     mode = "trainable" if trainable_only else "all"
#     lines.append(f"===== {title}: full {mode} parameter list (registration order) =====")

#     total = 0.0
#     for name, p in model.named_parameters():
#         if trainable_only and (not p.requires_grad):
#             continue
#         pm = p.numel() / 1e6
#         total += pm
#         lines.append(
#             f"{name}: shape={list(p.shape)}, trainable={p.requires_grad}, params={pm:.6f} M"
#         )

#     lines.append(f"[{title}] {mode} total = {total:.6f} M")
#     return lines

# def build_stage2_summary_report(model_first, model, title="stage2"):
#     """
#     第二阶段汇总：
#     - Stage-2 Trainable Params
#     - Inference-time Params
#     - 额外可训练参数（总 trainable 减去可训练 DINO blocks）
#     """
#     lines = []

#     stage2_trainable_m = count_params_m(model, trainable_only=True)
#     inference_params_m = count_params_m(model_first, trainable_only=False) + \
#                          count_params_m(model, trainable_only=False)

#     backbone, _ = find_dinov2_backbone(model)
#     trainable_block_total = 0.0
#     for blk in backbone.blocks:
#         if any(p.requires_grad for p in blk.parameters()):
#             trainable_block_total += sum(p.numel() for p in blk.parameters()) / 1e6

#     extra_trainable_m = stage2_trainable_m - trainable_block_total

#     lines.append(f"===== {title}: summary =====")
#     lines.append(f"Stage-2 Trainable Params (M): {stage2_trainable_m:.4f}")
#     lines.append(f"Inference-time Params (M):    {inference_params_m:.4f}")
#     lines.append(f"Trainable DINO Blocks (M):   {trainable_block_total:.4f}")
#     lines.append(f"Extra Trainable Params (M):  {extra_trainable_m:.4f}")

#     return lines

# def build_stage1_summary_report(model_first_stage1, title="stage1"):
#     """
#     第一阶段汇总：
#     - Stage-1 Trainable Params
#     """
#     lines = []
#     stage1_trainable_m = count_params_m(model_first_stage1, trainable_only=True)

#     backbone, _ = find_dinov2_backbone(model_first_stage1)
#     trainable_block_total = 0.0
#     for blk in backbone.blocks:
#         if any(p.requires_grad for p in blk.parameters()):
#             trainable_block_total += sum(p.numel() for p in blk.parameters()) / 1e6

#     extra_trainable_m = stage1_trainable_m - trainable_block_total

#     lines.append(f"===== {title}: summary =====")
#     lines.append(f"Stage-1 Trainable Params (M): {stage1_trainable_m:.4f}")
#     lines.append(f"Trainable DINO Blocks (M):    {trainable_block_total:.4f}")
#     lines.append(f"Extra Trainable Params (M):   {extra_trainable_m:.4f}")
#     return lines

# def write_report(lines, save_path=None, also_print=True):
#     """
#     一次性输出，避免终端被 tqdm / logger / 多次 print 搅乱
#     """
#     text = "\n".join(lines)

#     if save_path is not None:
#         with open(save_path, "w", encoding="utf-8") as f:
#             f.write(text)

#     if also_print:
#         print(text)

#     return text
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
from thop import profile
# def unwrap_model(model):
#     return model.module if hasattr(model, "module") else model


# class VigorQueryInferenceWrapper(nn.Module):
#     """
#     VIGOR query 路径：
#     model_first(x, type=False) + model(x, type=False)
#     """
#     def __init__(self, model_first, model, normalize=True):
#         super().__init__()
#         self.model_first = unwrap_model(model_first).eval()
#         # self.model = unwrap_model(model).eval()
#         self.normalize = normalize

#     def forward(self, x):
#         f1, _ = self.model_first(x, type=False)
#         # f2 = self.model(x, type=False)
#         # y = f1 + f2
#         if self.normalize:
#             y = F.normalize(f1, dim=-1)
#         return y


# class VigorRefInferenceWrapper(nn.Module):
#     """
#     VIGOR reference 路径：
#     model_first(x, type=True) + model(x, type=True, mask=None)
#     注意：这里按你 eval 代码的真实路径来写，mask=None
#     """
#     def __init__(self, model_first, model, normalize=True):
#         super().__init__()
#         self.model_first = unwrap_model(model_first).eval()
#         # self.model = unwrap_model(model).eval()
#         self.normalize = normalize

#     def forward(self, x):
#         f1, _ = self.model_first(x, type=True)
#         # f2 = self.model(x, type=True, mask=None)
#         # y = f1 + f2
#         if self.normalize:
#             y = F.normalize(f1, dim=-1)
#         return y


# @torch.no_grad()
# def measure_descriptor_dim(wrapper, input_shape=(1, 3, 448, 448), device="cuda"):
#     wrapper = wrapper.to(device).eval()
#     x = torch.randn(*input_shape, device=device)
#     y = wrapper(x)

#     # 若输出是 [B, D]，则维度是 D
#     # 若输出不是二维，就取单样本元素总数
#     if y.ndim == 2:
#         return int(y.shape[1])
#     return int(y[0].numel())


# @torch.no_grad()
# def measure_gflops(wrapper, input_shape=(1, 3, 448, 448), device="cuda"):
#     wrapper = wrapper.to(device).eval()
#     x = torch.randn(*input_shape, device=device)

#     # flops = FlopCountAnalysis(wrapper, x).total()
#     # return flops / 1e9
#     macs, params = profile(wrapper, inputs=(x,), verbose=False)

#     return {
#         "macs": macs,
#         "params": params,
#         "gmacs": macs / 1e9,
#         "gflops_2mac": (2 * macs) / 1e9,   # 常见 FLOPs 口径
#     }

# @torch.no_grad()
# def measure_with_thop(wrapper, input_shape=(1, 3, 448, 448), device="cuda"):
#     wrapper = wrapper.to(device).eval()
#     x = torch.randn(*input_shape, device=device)

#     macs, params = profile(wrapper, inputs=(x,), verbose=False)

#     return {
#         "macs": macs,
#         "params": params,
#         "gmacs": macs / 1e9,
#         "gflops_2mac": (2 * macs) / 1e9,   # 常见 FLOPs 口径
#     }

# def measure_vigor_inference_metrics(model_first, model=None, img_size=448, device="cuda"):
#     # q_wrapper = VigorQueryInferenceWrapper(model_first, model, normalize=True)
#     # r_wrapper = VigorRefInferenceWrapper(model_first, model, normalize=True)

#     # input_shape = (1, 3, img_size, img_size)

#     # query_dim = measure_descriptor_dim(q_wrapper, input_shape=input_shape, device=device)
#     # ref_dim = measure_descriptor_dim(r_wrapper, input_shape=input_shape, device=device)

#     # query_gflops = measure_gflops(q_wrapper, input_shape=input_shape, device=device)
#     # ref_gflops = measure_gflops(r_wrapper, input_shape=input_shape, device=device)

#     # result = {
#     #     "query_dim": query_dim,
#     #     "ref_dim": ref_dim,
#     #     "final_descriptor_dim": query_dim,   # 按当前实现，二者应相同
#     #     "query_gflops": query_gflops,
#     #     "ref_gflops": ref_gflops,
#     #     "pair_gflops": query_gflops + ref_gflops
#     # }

#     # print("\n===== VIGOR Inference Metrics =====")
#     # print(f"Query Descriptor Dim: {query_dim}")
#     # print(f"Ref Descriptor Dim:   {ref_dim}")
#     # print(f"Final Descriptor Dim: {query_dim}")
#     # print(f"Query GFLOPs:         {query_gflops:.4f}")
#     # print(f"Ref GFLOPs:           {ref_gflops:.4f}")
#     # print(f"Pair GFLOPs:          {query_gflops + ref_gflops:.4f}")

#     # return result
    
#     input_shape = (1, 3, img_size, img_size)

#     first_query = VigorQueryInferenceWrappe(model_first, type_flag=False)
#     dual_query = VigorQueryInferenceWrappe(model_first, model, type_flag=False)
    
#     r1 = measure_with_thop(first_query, input_shape=input_shape, device=device)
#     r2 = measure_with_thop(dual_query, input_shape=input_shape, device=device)

#     print("\n===== THOP naive results =====")
#     print(f"First-only Query GMACs: {r1['gmacs']:.4f}")
#     print(f"First-only Query GFLOPs (2*MACs): {r1['gflops_2mac']:.4f}")
#     print(f"Naive Dual Query GMACs: {r2['gmacs']:.4f}")
#     print(f"Naive Dual Query GFLOPs (2*MACs): {r2['gflops_2mac']:.4f}")

#     return r1, r2

def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

# ---------------------------------------------------------
# Wrapper 1: 单路完整前向 (Baseline)
# ---------------------------------------------------------
class FirstOnlyWrapper(nn.Module):
    """只测 model_first 单路完整前向"""
    def __init__(self, model_first, type_flag=False, normalize=True):
        super().__init__()
        self.model_first = unwrap_model(model_first).eval()
        self.type_flag = type_flag
        self.normalize = normalize

    def forward(self, x):
        f, _ = self.model_first(x, type=self.type_flag)
        if self.normalize:
            f = F.normalize(f, dim=-1)
        return f

# ---------------------------------------------------------
# Wrapper 2: 朴素双路前向
# ---------------------------------------------------------
class NaiveDualWrapper(nn.Module):
    """
    朴素双路：model_first(x) + model(x)
    注意：这里会把共享前端重复计算两次
    """
    def __init__(self, model_first, model, type_flag=False, normalize=True):
        super().__init__()
        self.model_first = unwrap_model(model_first).eval()
        self.model = unwrap_model(model).eval()
        self.type_flag = type_flag
        self.normalize = normalize

    def forward(self, x):
        if self.type_flag is False:
            f1, _ = self.model_first(x, type=False)
            f2 = self.model(x, type=False)
        else:
            f1, _ = self.model_first(x, type=True)
            f2 = self.model(x, type=True, mask=None)

        y = f1 + f2
        if self.normalize:
            y = F.normalize(y, dim=-1)
        return y

# ---------------------------------------------------------
# 测量执行函数
# ---------------------------------------------------------
@torch.no_grad()
def measure_with_thop(wrapper, input_shape=(1, 3, 448, 448), device="cuda"):
    wrapper = wrapper.to(device).eval()
    x = torch.randn(*input_shape, device=device)

    macs, params = profile(wrapper, inputs=(x,), verbose=False)

    return {
        "macs": macs,
        "params": params,
        "gmacs": macs / 1e9,
        "gflops_2mac": (2 * macs) / 1e9,   # 常见 FLOPs 口径
    }

def run_naive_thop_tests(model_first, model, img_size=448, device="cuda"):
    input_shape = (1, 3, img_size, img_size)

    first_query = FirstOnlyWrapper(model_first, type_flag=False)
    dual_query = NaiveDualWrapper(model_first, model, type_flag=False)

    r1 = measure_with_thop(first_query, input_shape=input_shape, device=device)
    r2 = measure_with_thop(dual_query, input_shape=input_shape, device=device)

    print("\n===== THOP naive results =====")
    print(f"First-only Query GMACs: {r1['gmacs']:.4f}")
    print(f"First-only Query GFLOPs (2*MACs): {r1['gflops_2mac']:.4f}")
    print(f"Naive Dual Query GMACs: {r2['gmacs']:.4f}")
    print(f"Naive Dual Query GFLOPs (2*MACs): {r2['gflops_2mac']:.4f}")

    return r1, r2

@dataclass
class Configuration:
    # Model
    model = 'dinov2_vitb14_MixVPR'

    # backbone
    backbone_arch = 'dinov2_vitb14'
    pretrained = True
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
    custom_sampling: bool = True  # use custom sampling instead of random
    seed = 1
    epochs: int = 40
    batch_size: int = 1  # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,1)  # GPU ids for training

    # Eval
    batch_size_eval: int = 100
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
    dataset: str = 'U1652-D2S'  # 'U1652-D2S' | 'U1652-S2D'
    data_folder: str = "/home/ubuntu/data/University-Release"

    # Augment Images
    prob_flip: float = 0.5  # flipping the sat image and drone image simultaneously

    # Savepath for model checkpoints
    model_path: str ="/home/ubuntu/data/CV-cities_train/university/d2s/second/weather"
    # Eval before training
    zero_shot: bool = True  

    # Checkpoint to start from
   #todo(这个是预训练好的模型)
    checkpoint_start_ready="/home/ubuntu/data/CV-cities_train/university/d2s/first/weather/dinov2_vitb14_MixVPR/2026-02-01_161616/weights_e4_0.9619.pth"
    checkpoint_start= "/home/ubuntu/data/CV-cities_train/university/d2s/second/weather/dinov2_vitb14_MixVPR/2026-02-02_010018/weights_e11_0.9649.pth"
    # checkpoint_start_ready="/home/ubuntu/data/CV-cities_train/university/d2s/first/weather/dinov2_vitb14_MixVPR/2026-02-01_161616/weights_e4_0.9619.pth"
    # checkpoint_start= None
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

    model_first = TimmModel(model_name=config.model,
                      pretrained=True,
                      img_size=config.img_size, backbone_arch=config.backbone_arch, agg_arch=config.agg_arch,
                      agg_config=config.agg_config, layer1=config.layer1,
                      model_type='first' )
    # print(model)
    
    model_state_dict_ready=torch.load(config.checkpoint_start_ready) 
    #这个模型保存时用了save_model方法
    # model_first.load_state_dict(model_state_dict_ready['model'], strict=False) 
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

    img_size = (config.img_size, config.img_size)

    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
        model_first.set_grad_checkpointing(True)

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
    # loss_function = InstanceLoss().to(config.device)
    # loss_function = TripletLoss().to(config.device)
    # loss_function = ContrastiveLoss().to(config.device)

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
        
        
    import torch

    # 测试参数 #todo
    # print_dinov2_block_status(model_first, "model_first")
    # print_dinov2_block_status(model, "model")

    # summary = summarize_stage2_detailed(model_first, model)

    # report_lines = []

    # # 1) DINO block 状态
    # report_lines += build_dino_block_report(model_first, title="model_first")
    # report_lines.append("")
    # report_lines += build_dino_block_report(model, title="model")
    # report_lines.append("")

    # # 2) 模块树：能看到 backbone / agg / 其他子模块
    # report_lines += build_module_tree_report(model, title="model", max_depth=2)
    # report_lines.append("")

    # # 3) 顶层裸参数，例如 logit_scale
    # report_lines += build_root_param_report(model, title="model")
    # report_lines.append("")

    # # 4) 关键词过滤：快速查看 agg / logit / norm / head
    # report_lines += build_keyword_report(
    #     model,
    #     keywords=("agg", "aggregator", "logit", "head", "norm"),
    #     title="model",
    #     trainable_only=True
    # )
    # report_lines.append("")

    # # 5) 如果你需要完整可训练参数明细，再打开这一段
    # # report_lines += build_full_param_report(model, title="model", trainable_only=True)
    # # report_lines.append("")

    # # 6) 第二阶段总汇总
    # report_lines += build_stage2_summary_report(model_first, model, title="stage2")

    # # 7) 一次性写文件 + 打印
    # # 注意：请确保运行环境中 model_path 变量已经定义
    # save_report_path = os.path.join("/home/ubuntu/Self_CVCties/CVCities-main", "parameter_report_stage2.txt")
    # write_report(report_lines, save_path=save_report_path, also_print=True)

    # print(f"\n参数统计报告已保存到: {save_report_path}")
    
    
    # report_lines = []

    # report_lines += build_dino_block_report(model_first, title="model_first")
    # report_lines.append("")

    # report_lines += build_stage1_summary_report(model_first, title="stage1")

    # save_report_path = os.path.join("/home/ubuntu/Self_CVCties/CVCities-main", "parameter_report_stage1.txt")
    # write_report(report_lines, save_path=save_report_path, also_print=True)
    
    metrics = run_naive_thop_tests(
        model_first=model_first,
        model=model,
        img_size=config.img_size,
        device=config.device
    )

    # ---------------------------------------
    
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
    # # -----------------------------------------------------------------------------#
    # start_epoch = 0
    # best_score = 0

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

    #     if config.custom_sampling:
    #         train_dataloader.dataset.shuffle()

    # if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
    #     torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    # else:
    #     torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))
