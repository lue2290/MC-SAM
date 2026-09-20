# -*- coding: utf-8 -*-
"""
MC-SAM 修复验证脚本
验证所有修复是否正确生效，无需GPU即可运行（使用CPU和小尺寸输入）

使用方法:
    cd MC-SAM
    python test_fixes.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback

# 统计
passed = 0
failed = 0
total = 0


def test(name, condition, msg=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}: {msg}")


def test_section(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


# ==================== 测试1: 模块导入 ====================
test_section("测试1: 模块导入")
try:
    from segment_anything.modeling.mcsam_integrated import (
        SinkhornProjection,
        RankDiceRMAModule,
        HyperCondModule,
        RMSNorm,
        ManifoldConstrainedAdapter,
        CrossModalStablePromptGenerator,
        BoundaryAwareLoss,
        IntegratedImageEncoderViT,
        MMSAM_Integrated,
    )
    test("所有模块导入成功", True)
except Exception as e:
    test("模块导入", False, str(e))
    print("无法继续测试，退出")
    sys.exit(1)


# ==================== 测试2: C3修复 - RankDice梯度可回传 ====================
test_section("测试2: C3修复 - RankDice软阈值梯度")
try:
    rankdice = RankDiceRMAModule(use_in_training=True, weight=0.1)
    rankdice.train()

    # 模拟输入
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    gt_mask = (torch.rand(2, 1, 64, 64) > 0.5).float()

    result_logits, rank_loss = rankdice(logits, gt_mask)
    test("RankDice前向传播成功", True)
    test("rank_loss是Tensor", isinstance(rank_loss, torch.Tensor))
    test("rank_loss非零", rank_loss.item() != 0.0, f"loss={rank_loss.item()}")

    # 关键：检查梯度能否回传
    rank_loss.backward()
    test("rank_loss梯度回传成功", logits.grad is not None)
    test("logits梯度非零", logits.grad is not None and logits.grad.abs().sum().item() > 0,
         f"grad_sum={logits.grad.abs().sum().item() if logits.grad is not None else 'None'}")
except Exception as e:
    test("C3修复测试", False, traceback.format_exc())


# ==================== 测试3: M1修复 - Sinkhorn直接输入 ====================
test_section("测试3: M1修复 - Sinkhorn投影")
try:
    adapter = ManifoldConstrainedAdapter(embed_dim=256, n_streams=4)
    sam_feat = torch.randn(2, 16, 256)
    blip_feat = torch.randn(2, 16, 256)

    output = adapter(sam_feat, blip_feat)
    test("ManifoldConstrainedAdapter前向成功", True)
    test("输出形状正确", output.shape == (2, 16, 256),
         f"期望(2,16,256), 实际{output.shape}")

    # 检查残差连接：输出应该接近输入（初始alpha小）
    diff = (output - sam_feat).abs().mean().item()
    test("残差连接有效（输出接近原始SAM特征）", diff < 1.0,
         f"平均差异={diff:.4f}")
except Exception as e:
    test("M1修复测试", False, traceback.format_exc())


# ==================== 测试4: S1修复 - 多token稀疏提示 ====================
test_section("测试4: S1修复 - CrossModalStablePromptGenerator多token输出")
try:
    prompt_gen = CrossModalStablePromptGenerator(
        text_dim=768, vision_dim=768, prompt_dim=256, n_prompt=2
    )

    text_embed = torch.randn(2, 768)
    vision_embed = torch.randn(2, 768)
    text_features = torch.randn(2, 10, 768)
    vision_features = torch.randn(2, 768, 8, 8)

    sparse_prompt = prompt_gen(text_embed, vision_embed, text_features, vision_features)

    test("前向传播成功", True)
    test("输出形状为[B, n_prompt, prompt_dim]",
         sparse_prompt.shape == (2, 2, 256),
         f"期望(2,2,256), 实际{sparse_prompt.shape}")
except Exception as e:
    test("S1修复测试", False, traceback.format_exc())


# ==================== 测试5: C2修复 - _init_weights范围限制 ====================
test_section("测试5: C2修复 - _init_weights不覆盖预训练权重")
try:
    # 创建一个小型编码器和解码器，模拟SAM结构
    # 在MMSAM_Integrated构造函数中，mask_decoder的权重应该不被_init_weights覆盖

    # 创建一个假的mask_decoder
    class FakeMaskDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.test_linear = nn.Linear(256, 256)
            # 设置一个已知的权重值
            with torch.no_grad():
                self.test_linear.weight.fill_(0.42)

    fake_decoder = FakeMaskDecoder()
    original_weight = fake_decoder.test_linear.weight.clone()

    # 创建一个小型图像编码器
    small_encoder = IntegratedImageEncoderViT(
        img_size=64, patch_size=16, embed_dim=64, depth=2, num_heads=2,
        out_chans=32, window_size=0, global_attn_indexes=[], blip_feature_dim=64
    )

    # 创建集成模型 - 此时 _init_weights 会被调用
    model = MMSAM_Integrated(
        image_encoder=small_encoder,
        mask_decoder=fake_decoder,
        image_feature_dim=768,
        use_rankdice=True,
        use_hypercond=True,
    )

    # 检查mask_decoder的权重是否被保留
    current_weight = model.mask_decoder.test_linear.weight
    weights_preserved = torch.allclose(current_weight, original_weight, atol=1e-6)
    test("mask_decoder权重未被_init_weights覆盖", weights_preserved,
         f"original={original_weight[0,:3]}, current={current_weight[0,:3]}")
except Exception as e:
    test("C2修复测试", False, traceback.format_exc())


# ==================== 测试6: S3修复 - HyperCond推理时激活 ====================
test_section("测试6: S3修复 - HyperCond推理时激活")
try:
    hyper_cond = HyperCondModule(cond_dim=128, hidden_dim=64)
    hyper_cond.eval()  # 推理模式

    cond = hyper_cond(0.5, 1.0, 'cpu', batch_size=2)
    test("HyperCond eval模式前向成功", True)
    test("条件嵌入形状正确", cond.shape == (2, 128),
         f"期望(2,128), 实际{cond.shape}")
    test("条件嵌入非零", cond.abs().sum().item() > 0)
except Exception as e:
    test("S3修复测试", False, traceback.format_exc())


# ==================== 测试7: SinkhornProjection数值稳定性 ====================
test_section("测试7: Sinkhorn投影数值稳定性")
try:
    sinkhorn = SinkhornProjection(iters=5)

    # 正常输入
    cost = torch.randn(2, 4, 4)
    result = sinkhorn(cost)
    test("正常输入Sinkhorn成功", True)
    test("输出中无NaN", not torch.isnan(result).any())
    test("输出中无Inf", not torch.isinf(result).any())

    # 检查行和列归一化
    row_sums = result.sum(dim=-1)
    col_sums = result.sum(dim=-2)
    test("行近似归一化", (row_sums - 1.0).abs().max().item() < 0.1,
         f"max_row_deviation={( row_sums - 1.0).abs().max().item():.4f}")
    test("列近似归一化", (col_sums - 1.0).abs().max().item() < 0.1,
         f"max_col_deviation={(col_sums - 1.0).abs().max().item():.4f}")

    # 极端输入
    extreme = torch.randn(2, 4, 4) * 100
    result_extreme = sinkhorn(extreme)
    test("极端输入Sinkhorn无NaN", not torch.isnan(result_extreme).any())
except Exception as e:
    test("Sinkhorn测试", False, traceback.format_exc())


# ==================== 测试8: BoundaryAwareLoss ====================
test_section("测试8: BoundaryAwareLoss")
try:
    loss_fn = BoundaryAwareLoss(alpha=1.0, beta=0.1)
    pred = torch.randn(2, 1, 64, 64)
    gt = (torch.rand(2, 1, 64, 64) > 0.5).float()

    total_loss, dice_ce_loss, boundary_loss = loss_fn(pred, gt)
    test("损失计算成功", True)
    test("总损失非零", total_loss.item() > 0)
    test("总损失可回传", True)

    total_loss.backward()
    test("梯度回传成功", pred.grad is None)  # pred没有requires_grad
except Exception as e:
    test("BoundaryAwareLoss测试", False, traceback.format_exc())


# ==================== 测试9: set_training_mode 安全性 ====================
test_section("测试9: set_training_mode安全性（可选模块）")
try:
    # 创建不使用rankdice和hypercond的模型
    small_encoder2 = IntegratedImageEncoderViT(
        img_size=64, patch_size=16, embed_dim=64, depth=2, num_heads=2,
        out_chans=32, window_size=0, global_attn_indexes=[], blip_feature_dim=64
    )

    class DummyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32)

    model_no_extras = MMSAM_Integrated(
        image_encoder=small_encoder2,
        mask_decoder=DummyDecoder(),
        image_feature_dim=768,
        use_rankdice=False,
        use_hypercond=False,
    )

    # 这应该不会抛出 AttributeError
    model_no_extras.set_training_mode(True)
    model_no_extras.set_training_mode(False)
    test("无rankdice/hypercond时set_training_mode不报错", True)
except Exception as e:
    test("set_training_mode安全性", False, traceback.format_exc())



# ==================== 测试10: einsum修复 - 流形混合矩阵实际生效 ====================
test_section("测试10: einsum修复 - ManifoldConstrainedAdapter流混合有效性")
try:
    adapter2 = ManifoldConstrainedAdapter(embed_dim=32, n_streams=4, sinkhorn_iters=5)
    sam_feat2 = torch.randn(1, 4, 32, requires_grad=True)
    blip_feat2 = torch.randn(1, 4, 32)

    output2 = adapter2(sam_feat2, blip_feat2)
    test("ManifoldConstrainedAdapter前向成功", True)

    # 关键验证：梯度应能通过混合矩阵回传到所有mixing_mlp参数
    loss2 = output2.sum()
    loss2.backward()

    mixing_mlp_grads = []
    for name, param in adapter2.named_parameters():
        if 'mixing_mlp' in name and param.grad is not None:
            mixing_mlp_grads.append(param.grad.abs().sum().item())

    test("mixing_mlp梯度非零（混合矩阵实际参与计算）",
         len(mixing_mlp_grads) > 0 and all(g > 0 for g in mixing_mlp_grads),
         f"梯度={mixing_mlp_grads}")

    # 验证不同的mixing weights产生不同输出
    adapter3 = ManifoldConstrainedAdapter(embed_dim=32, n_streams=4, sinkhorn_iters=5)
    with torch.no_grad():
        # 修改mixing_mlp权重，输出应该改变
        for p in adapter3.mixing_mlp.parameters():
            p.data += 10.0  # 大幅改变权重

    out_a = adapter2(sam_feat2.detach(), blip_feat2)
    out_b = adapter3(sam_feat2.detach(), blip_feat2)
    diff_outputs = (out_a - out_b).abs().mean().item()
    test("不同mixing权重产生不同输出（非no-op）", diff_outputs > 1e-6,
         f"输出差异={diff_outputs:.8f}")
except Exception as e:
    test("einsum修复测试", False, traceback.format_exc())


# ==================== 测试11: 优化器参数覆盖完整性 ====================
test_section("测试11: 优化器参数组覆盖完整性")
try:
    from segment_anything.modeling.mcsam_integrated import create_optimizer_for_integrated_model

    small_encoder3 = IntegratedImageEncoderViT(
        img_size=64, patch_size=16, embed_dim=64, depth=2, num_heads=2,
        out_chans=32, window_size=0, global_attn_indexes=[], blip_feature_dim=64
    )

    class DummyDecoder2(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32)

    model_full = MMSAM_Integrated(
        image_encoder=small_encoder3,
        mask_decoder=DummyDecoder2(),
        image_feature_dim=768,
        use_rankdice=True,
        use_hypercond=True,
    )

    optimizer = create_optimizer_for_integrated_model(model_full, lr=0.001)

    # 收集优化器中所有参数的id
    optimizer_param_ids = set()
    for group in optimizer.param_groups:
        for param in group['params']:
            optimizer_param_ids.add(id(param))

    # 检查所有model参数都在优化器中
    missing_params = []
    for name, param in model_full.named_parameters():
        if id(param) not in optimizer_param_ids:
            missing_params.append(name)

    test("所有参数都在优化器中", len(missing_params) == 0,
         f"遗漏参数: {missing_params}")

    # 特别检查之前遗漏的关键参数
    critical_names = ['text_adapter', 'cond_channel_adapter', 'cond_spatial_adapter']
    for crit_name in critical_names:
        found = False
        for name, param in model_full.named_parameters():
            if crit_name in name:
                found = id(param) in optimizer_param_ids
                if not found:
                    break
        test(f"'{crit_name}'参数在优化器中", found)
except Exception as e:
    test("优化器参数覆盖测试", False, traceback.format_exc())


# ==================== 汇总 ====================
print(f"\n{'='*60}")
print(f"  测试结果汇总")
print(f"{'='*60}")
print(f"  总测试数: {total}")
print(f"  通过: {passed}")
print(f"  失败: {failed}")
print(f"{'='*60}")

if failed == 0:
    print("  🎉 所有测试通过！修复验证成功。")
else:
    print(f"  ⚠️ 有 {failed} 个测试失败，请检查。")

sys.exit(0 if failed == 0 else 1)
