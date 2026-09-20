# inference_integrated.py
# !/usr/bin/env python3
"""
集成模型的推理脚本
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm

from segment_anything.modeling.mcsam_integrated import create_integrated_model
from MCsam_train import NpyDataset, eval_psnr
from transformers import AutoTokenizer, BlipProcessor, BlipForConditionalGeneration, MambaModel


def main():
    parser = argparse.ArgumentParser(description="集成模型推理")
    parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--data_path", type=str, required=True, help="测试数据路径")
    parser.add_argument("--sam_checkpoint", type=str, default="sam/sam_vit_l_0b3195.pth", help="SAM预训练权重")
    parser.add_argument("--model_type", type=str, default="vit_l", help="模型类型")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--threshold", type=float, default=0.5, help="阈值")
    parser.add_argument("--boundary_weight", type=float, default=1.0, help="边界权重")
    parser.add_argument("--save_dir", type=str, default="result", help="结果保存目录")
    parser.add_argument("--visualize", action="store_true", help="是否可视化结果")

    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    print("加载模型...")
    model = create_integrated_model(
        sam_checkpoint_path=args.sam_checkpoint,
        model_type=args.model_type,
        image_size=1024,
        use_rankdice=True,
        use_hypercond=True,
        n_streams=4,
        device=device
    ).to(device)

    # 加载检查点
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()

    # 加载VLM和Mamba模型
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    processor = BlipProcessor.from_pretrained("/root/autodl-tmp/MMsam/Blip")
    vlm_model = BlipForConditionalGeneration.from_pretrained("/root/autodl-tmp/MMsam/Blip").to(device)
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/MMsam/mamba")
    mamba_model = MambaModel.from_pretrained("/root/autodl-tmp/MMsam/mamba").to(device)

    vlm_model.eval()
    mamba_model.eval()

    # 加载数据
    test_dataset = NpyDataset(args.data_path)
    print(f"测试样本数: {len(test_dataset)}")

    # 评估指标
    metrics = {
        'sm': [], 'em': [], 'wfm': [], 'mae': [],
        'dice': [], 'iou': [], 'ber': []
    }

    # 推理
    for idx in tqdm(range(len(test_dataset)), desc="推理进度"):
        image, gt2D, img_1024_ori = test_dataset[idx]

        # 添加批次维度
        image = image.unsqueeze(0).to(device)
        gt2D = gt2D.unsqueeze(0).to(device)
        img_1024_ori = torch.from_numpy(img_1024_ori).unsqueeze(0).to(device)

        with torch.no_grad():
            # 获取VLM描述
            vlm_inputs = processor(img_1024_ori, return_tensors="pt").to(device)
            vlm_outputs = vlm_model.generate(**vlm_inputs)
            description = processor.decode(vlm_outputs[0], skip_special_tokens=True)

            # 提取文本信息
            mamba_inputs = tokenizer(description, padding=True, return_tensors="pt").to(device)
            mamba_outputs = mamba_model(**mamba_inputs)
            vision_outputs = vlm_model.vision_model(**vlm_inputs)
            image_features = vision_outputs.last_hidden_state[:, 1:, :]

            # 处理图像特征
            batch_size, seq_len, hidden_dim = image_features.shape
            if seq_len == 576:
                image_features = image_features.reshape(batch_size, 24, 24, hidden_dim)
                image_features = image_features.permute(0, 3, 1, 2)
                image_features = F.interpolate(image_features, size=(64, 64), mode='bilinear', align_corners=False)
            elif seq_len == 64 * 64:
                image_features = image_features.reshape(batch_size, 64, 64, hidden_dim).permute(0, 3, 1, 2)
            else:
                image_features = image_features.mean(dim=1, keepdim=True)
                image_features = image_features.unsqueeze(-1)
                image_features = F.interpolate(image_features, size=(64, 64))

            text_features = mamba_outputs.last_hidden_state

            # 超参数条件
            hyper_cond = {
                'threshold': args.threshold,
                'boundary_weight': args.boundary_weight
            }

            # 模型推理
            pred_mask, _, _ = model(
                image=image,
                text_embeddings=text_features,
                image_features=image_features,
                gt_mask=None,
                hyper_cond=hyper_cond,
                return_logits=False
            )

            # 转换为numpy
            pred_np = pred_mask.squeeze().cpu().numpy()
            gt_np = gt2D.squeeze().cpu().numpy()

            # 保存结果
            if args.visualize:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # 原始图像
                axes[0].imshow(img_1024_ori.squeeze().cpu().numpy())
                axes[0].set_title("原始图像")
                axes[0].axis('off')

                # 预测掩码
                axes[1].imshow(pred_np, cmap='gray')
                axes[1].set_title("预测掩码")
                axes[1].axis('off')

                # 真实掩码
                axes[2].imshow(gt_np, cmap='gray')
                axes[2].set_title("真实掩码")
                axes[2].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(args.save_dir, f"result_{idx:04d}.png"), dpi=150, bbox_inches='tight')
                plt.close()

            # 保存预测掩码
            pred_img = Image.fromarray((pred_np * 255).astype(np.uint8))
            pred_img.save(os.path.join(args.save_dir, f"pred_{idx:04d}.png"))

    print(f"\n推理完成！结果保存到: {args.save_dir}")

    # 计算总体指标
    print("\n=== 总体性能指标 ===")
    for metric_name, values in metrics.items():
        if values:
            avg_value = np.mean(values)
            print(f"{metric_name}: {avg_value:.4f}")


if __name__ == "__main__":
    main()