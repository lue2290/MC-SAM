# #!/usr/bin/env python3
# """
# 调试维度问题
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# def debug_sam_dimensions():
#     """调试SAM模型维度"""
#     print("=== SAM模型维度调试 ===")
#
#     # SAM模型配置
#     sam_configs = {
#         "vit_b": {
#             "embed_dim": 768,
#             "num_heads": 12,
#             "depth": 12,
#             "patch_size": 16,
#             "image_size": 1024
#         },
#         "vit_l": {
#             "embed_dim": 1024,
#             "num_heads": 16,
#             "depth": 24,
#             "patch_size": 16,
#             "image_size": 1024
#         },
#         "vit_h": {
#             "embed_dim": 1280,
#             "num_heads": 16,
#             "depth": 32,
#             "patch_size": 16,
#             "image_size": 1024
#         }
#     }
#
#     model_type = "vit_l"
#     config = sam_configs[model_type]
#
#     print(f"模型类型: {model_type}")
#     print(f"embed_dim: {config['embed_dim']}")
#     print(f"num_heads: {config['num_heads']}")
#     print(f"depth: {config['depth']}")
#     print(f"patch_size: {config['patch_size']}")
#     print(f"image_size: {config['image_size']}")
#
#     # 计算特征图大小
#     patch_grid = config['image_size'] // config['patch_size']
#     print(f"\n特征网格大小: {patch_grid} x {patch_grid} = {patch_grid * patch_grid}")
#
#     # 计算注意力维度
#     head_dim = config['embed_dim'] // config['num_heads']
#     print(f"每个注意力头维度: {head_dim}")
#
#     # 计算qkv维度
#     qkv_dim = 3 * config['embed_dim']
#     print(f"QKV总维度: {qkv_dim}")
#
#     # 模拟输入
#     batch_size = 1
#     x = torch.randn(batch_size, patch_grid, patch_grid, config['embed_dim'])
#     print(f"\n模拟输入形状: {x.shape}")
#
#     # 重塑为注意力计算需要的形状
#     B, H, W, C = x.shape
#     x_reshaped = x.reshape(B, H * W, C)
#     print(f"重塑后形状 (注意力计算): {x_reshaped.shape}")
#
#     # 计算qkv投影
#     qkv_proj = nn.Linear(C, 3 * C)
#     qkv = qkv_proj(x_reshaped)
#     print(f"QKV投影后形状: {qkv.shape}")
#
#     # 重塑为多头注意力格式
#     qkv_reshaped = qkv.reshape(B, H * W, 3, config['num_heads'], head_dim)
#     print(f"多头注意力重塑后形状: {qkv_reshaped.shape}")
#
#     # 验证维度
#     total_elements = qkv_reshaped.numel()
#     expected_elements = B * H * W * 3 * config['num_heads'] * head_dim
#     print(f"\n总元素数: {total_elements}")
#     print(f"预期元素数: {expected_elements}")
#     print(f"匹配: {total_elements == expected_elements}")
#
#     return config
#
#
# def debug_blip_features():
#     """调试BLIP特征维度"""
#     print("\n=== BLIP特征维度调试 ===")
#
#     # BLIP配置
#     blip_config = {
#         "hidden_dim": 768,
#         "spatial_size": 24,  # 24x24
#         "seq_len": 576,  # 24*24
#         "target_size": 64  # 目标上采样大小
#     }
#
#     print(f"BLIP隐藏维度: {blip_config['hidden_dim']}")
#     print(f"BLIP空间大小: {blip_config['spatial_size']}x{blip_config['spatial_size']}")
#     print(f"BLIP序列长度: {blip_config['seq_len']}")
#
#     # 模拟BLIP特征
#     batch_size = 1
#     blip_features_3d = torch.randn(batch_size, blip_config['seq_len'], blip_config['hidden_dim'])
#     print(f"\nBLIP 3D特征形状: {blip_features_3d.shape}")
#
#     # 重塑为4D
#     blip_features_4d = blip_features_3d.reshape(
#         batch_size,
#         blip_config['spatial_size'],
#         blip_config['spatial_size'],
#         blip_config['hidden_dim']
#     ).permute(0, 3, 1, 2)
#
#     print(f"BLIP 4D特征形状: {blip_features_4d.shape}")
#
#     # 上采样到目标大小
#     blip_upsampled = F.interpolate(
#         blip_features_4d,
#         size=(blip_config['target_size'], blip_config['target_size']),
#         mode='bilinear',
#         align_corners=False
#     )
#
#     print(f"上采样后BLIP特征形状: {blip_upsampled.shape}")
#
#     return blip_config
#
#
# def debug_integration():
#     """调试集成模型维度"""
#     print("\n=== 集成模型维度调试 ===")
#
#     # SAM ViT-L配置
#     sam_config = {
#         "embed_dim": 1024,
#         "num_heads": 16,
#         "patch_grid": 64,  # 1024/16
#     }
#
#     # BLIP配置
#     blip_config = {
#         "hidden_dim": 768,
#         "target_size": 64
#     }
#
#     print(f"SAM ViT-L embed_dim: {sam_config['embed_dim']}")
#     print(f"SAM ViT-L num_heads: {sam_config['num_heads']}")
#     print(f"SAM ViT-L patch_grid: {sam_config['patch_grid']}")
#     print(f"BLIP hidden_dim: {blip_config['hidden_dim']}")
#
#     # 维度转换需求
#     print(f"\n维度转换需求:")
#     print(f"BLIP -> SAM: {blip_config['hidden_dim']} -> {sam_config['embed_dim']}")
#
#     # 创建调整层
#     adjust_layer = nn.Sequential(
#         nn.Conv2d(blip_config['hidden_dim'], sam_config['embed_dim'], 1),
#         nn.GELU(),
#         nn.Conv2d(sam_config['embed_dim'], sam_config['embed_dim'], 1),
#         nn.GELU()
#     )
#
#     # 模拟调整
#     batch_size = 1
#     blip_input = torch.randn(batch_size, blip_config['hidden_dim'],
#                              blip_config['target_size'], blip_config['target_size'])
#     print(f"\nBLIP输入形状: {blip_input.shape}")
#
#     adjusted = adjust_layer(blip_input)
#     print(f"调整后形状: {adjusted.shape}")
#
#     # 验证与SAM兼容
#     if adjusted.shape[1] == sam_config['embed_dim']:
#         print("✅ 调整后维度与SAM兼容")
#     else:
#         print("❌ 调整后维度与SAM不兼容")
#
#     return True
#
#
# if __name__ == "__main__":
#     debug_sam_dimensions()
#     debug_blip_features()
#     debug_integration()
#     print("\n✅ 调试完成")
# test_training.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys


def test_training_mode():
    """测试训练模式"""
    print("=== 测试训练模式 ===")

    # 导入模型
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from segment_anything.modeling.mcsam_integrated import create_integrated_model

    device = torch.device("cuda:0")

    # 创建模型
    model = create_integrated_model(
        sam_checkpoint_path="/root/autodl-tmp/MMsam/sam/sam_vit_l_0b3195.pth",
        model_type="vit_l",
        image_size=1024,
        use_rankdice=True,
        use_hypercond=True,
        n_streams=4,
        device=device,
        blip_feature_dim=768
    ).to(device)

    # 设置为训练模式
    model.train()
    print(f"训练模式: {model.training}")

    # 创建测试输入
    batch_size = 1
    image = torch.randn(batch_size, 3, 1024, 1024).to(device)
    text_embeddings = torch.randn(batch_size, 77, 768).to(device)
    image_features = torch.randn(batch_size, 576, 768).to(device)
    gt_mask = torch.randn(batch_size, 1, 1024, 1024).to(device)

    # 清空缓存
    torch.cuda.empty_cache()
    print(f"开始内存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    # 测试训练前向
    print("\n测试训练前向传播...")
    logits, rank_dice_loss, hyper_cond = model(
        image=image,
        text_embeddings=text_embeddings,
        image_features=image_features,
        gt_mask=gt_mask,
        hyper_cond={'threshold': 0.5, 'boundary_weight': 1.0},
        return_logits=True
    )

    print(f"训练前向后内存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"最大内存: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    print(f"logits形状: {logits.shape}, 损失: {rank_dice_loss}")

    # 测试反向传播
    print("\n测试反向传播...")
    total_loss = rank_dice_loss + F.binary_cross_entropy_with_logits(logits, gt_mask)

    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
    optimizer.zero_grad()

    total_loss.backward()
    optimizer.step()

    print(f"反向传播后内存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"最大内存: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")

    return model


def test_amp_training():
    """测试混合精度训练"""
    print("\n=== 测试混合精度训练 ===")

    from segment_anything.modeling.mcsam_integrated import create_integrated_model

    device = torch.device("cuda:0")

    # 创建模型
    model = create_integrated_model(
        sam_checkpoint_path="/root/autodl-tmp/MMsam/sam/sam_vit_l_0b3195.pth",
        model_type="vit_l",
        image_size=1024,
        use_rankdice=True,
        use_hypercond=True,
        n_streams=4,
        device=device,
        blip_feature_dim=768
    ).to(device)

    model.train()

    # 创建输入
    batch_size = 1
    image = torch.randn(batch_size, 3, 1024, 1024).to(device)
    text_embeddings = torch.randn(batch_size, 77, 768).to(device)
    image_features = torch.randn(batch_size, 576, 768).to(device)
    gt_mask = torch.randn(batch_size, 1, 1024, 1024).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
    scaler = torch.cuda.amp.GradScaler()

    # 清空缓存
    torch.cuda.empty_cache()
    print(f"开始内存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    # 使用AMP
    print("使用AMP进行训练...")
    optimizer.zero_grad()

    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        logits, rank_dice_loss, hyper_cond = model(
            image=image,
            text_embeddings=text_embeddings,
            image_features=image_features,
            gt_mask=gt_mask,
            hyper_cond={'threshold': 0.5, 'boundary_weight': 1.0},
            return_logits=True
        )

        total_loss = rank_dice_loss + F.binary_cross_entropy_with_logits(logits, gt_mask)

    print(f"前向传播后内存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"反向传播后内存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"最大内存: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")

    return model


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"总内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

    # 测试1: 普通训练模式
    print("\n" + "=" * 50)
    model1 = test_training_mode()

    # 清空缓存
    del model1
    torch.cuda.empty_cache()

    # 测试2: AMP训练模式
    print("\n" + "=" * 50)
    model2 = test_amp_training()


if __name__ == "__main__":
    main()