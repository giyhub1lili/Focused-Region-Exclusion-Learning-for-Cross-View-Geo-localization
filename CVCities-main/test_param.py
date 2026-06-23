import torch

def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

def count_params_m(model, trainable_only=False):
    """
    统计参数量，单位 M（百万）
    """
    model = unwrap_model(model)
    if trainable_only:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        n = sum(p.numel() for p in model.parameters())
    return n / 1e6

def count_named_params_m(model, keyword=None, trainable_only=False):
    """
    按名称筛选统计参数量，便于核查 backbone / aggregator / block 等部分
    """
    model = unwrap_model(model)
    total = 0
    for name, p in model.named_parameters():
        if keyword is not None and keyword not in name:
            continue
        if trainable_only and (not p.requires_grad):
            continue
        total += p.numel()
    return total / 1e6

def print_dinov2_block_status(model, title="model"):
    """
    逐块打印 DINOv2（视觉 Transformer 主干）每个 block 是否被冻结
    """
    model = unwrap_model(model)

    # 这里假设你的 TimmModel 内部路径是 model.backbone.dino_model
    backbone = model.backbone.dino_model

    print(f"\n[{title}] patch_embed trainable:",
          any(p.requires_grad for p in backbone.patch_embed.parameters()))

    for i, blk in enumerate(backbone.blocks):
        flag = any(p.requires_grad for p in blk.parameters())
        blk_params = sum(p.numel() for p in blk.parameters()) / 1e6
        print(f"[{title}] block {i:02d}: trainable={flag}, params={blk_params:.4f} M")

def summarize_two_stage_params(model_first, model):
    """
    返回三类最关键的参数量：
    1) 第一阶段可学习参数
    2) 第二阶段可学习参数
    3) 测试时总参数
    """
    m1 = unwrap_model(model_first)
    m2 = unwrap_model(model)

    stage1_trainable_m = count_params_m(m1, trainable_only=True)
    stage2_trainable_m = count_params_m(m2, trainable_only=True)

    # 注意：测试时参数量统计的是“所有参与前向的参数”，不是 trainable params
    inference_params_m = count_params_m(m1, trainable_only=False) + \
                         count_params_m(m2, trainable_only=False)

    print("\n===== Parameter Summary =====")
    print(f"Stage-1 Trainable Params (M): {stage1_trainable_m:.4f}")
    print(f"Stage-2 Trainable Params (M): {stage2_trainable_m:.4f}")
    print(f"Inference-time Params (M):    {inference_params_m:.4f}")

    return {
        "stage1_trainable_m": stage1_trainable_m,
        "stage2_trainable_m": stage2_trainable_m,
        "inference_params_m": inference_params_m,
    }
    
print_dinov2_block_status(model_first, "model_first")
print_dinov2_block_status(model, "model")

summary = summarize_two_stage_params(model_first, model)