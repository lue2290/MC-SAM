# debug_memory.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc


def print_memory_stats(label=""):
    allocated = torch.cuda.memory_allocated() / 1e9
    cached = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    print(f"{label}: 已分配={allocated:.2f}GB, 缓存={cached:.2f}GB, 最大已分配={max_allocated:.2f}GB")


def test_sam_baseline():
    """测试基础SAM模型"""
    from segment_anything import sam_model_registry

    print("=== 测试基础SAM模型 ===")
    torch.cuda.empty_cache()

    model = sam_model_registry["vit_l"](checkpoint="/root/autodl-tmp/MMsam/sam/sam_vit_l_0b3195.pth").to("cuda:0")
    model.eval()

    print_memory_stats("模型加载后")

    # 测试输入
    x = torch.randn(1, 3, 1024, 1024).cuda()
    print_memory_stats("创建输入后")

    with torch.no_grad():
        image_embedding = model.image_encoder(x)
        print_memory_stats("图像编码器后")

        # 测试完整推理
        output = model(x, multimask_output=False)
        print_memory_stats("完整推理后")

    return model


def test_integrated_step_by_step():
    """逐步测试集成模型"""
    print("\n=== 逐步测试集成模型 ===")

    # 导入模块
    from segment_anything.modeling.mcsam_integrated import create_integrated_model

    # 清空缓存
    torch.cuda.empty_cache()
    gc.collect()

    # 创建模型
    model = create_integrated_model(
        sam_checkpoint_path="/root/autodl-tmp/MMsam/sam/sam_vit_l_0b3195.pth",
        model_type="vit_l",
        image_size=1024,
        use_rankdice=True,
        use_hypercond=True,
        n_streams=4,
        device="cuda:0"
    ).to("cuda:0")

    model.eval()
    print_memory_stats("集成模型加载后")

    # 创建模拟输入
    batch_size = 1

    # 图像输入
    image = torch.randn(batch_size, 3, 1024, 1024).cuda()
    print_memory_stats("创建图像输入后")

    # 文本特征 (模拟BLIP输出)
    text_embeddings = torch.randn(batch_size, 77, 768).cuda()
    print_memory_stats("创建文本特征后")

    # 图像特征 (模拟VLM输出)
    image_features = torch.randn(batch_size, 576, 768).cuda()  # 24x24=576
    print_memory_stats("创建图像特征后")

    # 逐步测试
    print("\n=== 测试图像编码器 ===")
    with torch.no_grad():
        # 单独测试图像编码器
        print_memory_stats("开始前")
        image_embedding = model.image_encoder(image, image_features)
        print_memory_stats("图像编码器后")
        print(f"图像嵌入形状: {image_embedding.shape}")

        # 测试prompt生成器
        print("\n=== 测试Prompt生成器 ===")
        text_global = text_embeddings.mean(dim=1)
        vision_global = image_features.mean(dim=1)

        sparse_prompt = model.prompt_generator(
            text_global,
            vision_global,
            text_embeddings,
            image_features
        )
        print_memory_stats("Prompt生成器后")
        print(f"稀疏提示形状: {sparse_prompt.shape}")

        # 测试超参数条件化
        print("\n=== 测试超参数条件化 ===")
        if model.use_hypercond:
            cond_embedding = model.hyper_cond(0.5, 1.0, "cuda:0")
            print_memory_stats("超参数条件化后")
            print(f"条件嵌入形状: {cond_embedding.shape}")

        # 完整前向（不使用梯度）
        print("\n=== 完整前向传播 ===")
        model.eval()
        pred_mask, rank_dice_loss, hyper_cond = model(
            image=image,
            text_embeddings=text_embeddings,
            image_features=image_features,
            gt_mask=None,
            hyper_cond={'threshold': 0.5, 'boundary_weight': 1.0},
            return_logits=False
        )
        print_memory_stats("完整前向后")
        print(f"预测掩码形状: {pred_mask.shape}")


def test_attention_memory():
    """测试注意力内存使用"""
    print("\n=== 测试注意力内存 ===")

    # 计算ViT-L注意力矩阵大小
    image_size = 1024
    patch_size = 16
    seq_len = (image_size // patch_size) ** 2  # 4096
    num_heads = 16
    batch_size = 1

    # 注意力矩阵大小: [batch_size, num_heads, seq_len, seq_len]
    attn_size = batch_size * num_heads * seq_len * seq_len

    # float32: 4字节每个元素
    attn_memory_gb = attn_size * 4 / 1e9

    print(f"序列长度: {seq_len}")
    print(f"注意力头数: {num_heads}")
    print(f"注意力矩阵元素数: {attn_size:,}")
    print(f"注意力矩阵内存 (float32): {attn_memory_gb:.2f} GB")
    print(f"注意力矩阵内存 (bfloat16): {attn_memory_gb / 2:.2f} GB")

    # 测试实际分配
    torch.cuda.empty_cache()
    print_memory_stats("开始前")

    try:
        # 尝试分配注意力矩阵
        attn_matrix = torch.randn(batch_size, num_heads, seq_len, seq_len).cuda()
        print_memory_stats("分配注意力矩阵后")
        print("✅ 可以分配完整注意力矩阵")
    except Exception as e:
        print(f"❌ 无法分配完整注意力矩阵: {e}")


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"总内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 1. 测试注意力内存
    test_attention_memory()

    # 2. 测试基础SAM
    # test_sam_baseline()

    # 3. 逐步测试集成模型
    test_integrated_step_by_step()

    print("\n=== 内存测试完成 ===")