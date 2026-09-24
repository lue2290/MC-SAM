# -*- coding: utf-8 -*-
"""
训练集成的四合一MMSAM模型
包含: 1)流形约束超连接适配器 2)RankDice-RMA模块 3)超参数条件化 4)跨空间稳定Prompt生成器
"""

# %% 设置环境
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import shutil
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加当前目录到路径，确保能导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Beta

from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from transformers import AutoTokenizer, BlipProcessor, BlipForConditionalGeneration, MambaModel

# 导入我们的集成模型
from segment_anything.modeling.mcsam_integrated import (
    create_integrated_model,
    create_optimizer_for_integrated_model,
    BoundaryAwareLoss,
    evaluate_model
)

from utils_downstream.saliency_metric import (
    cal_mae, cal_sm, cal_em, cal_wfm, cal_dice, cal_iou, cal_ber, cal_acc,
)

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '0'  # 使用在线模式


# %% 随机种子设置
def set_seed(seed: int = 42, deterministic: bool = False):
    """
    统一设置所有随机种子，保证实验可复现。

    Args:
        seed: 随机种子
        deterministic: 是否启用 cuDNN 确定性（会牺牲训练速度，
                       但能显著提高可复现性）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 让 Python 的哈希也稳定
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 更严格但更慢；某些算子不支持时会报错，可先不开
        # torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"[set_seed] seed={seed}, deterministic={deterministic}")


def seed_worker(worker_id: int):
    """DataLoader worker 的随机种子初始化函数"""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """创建带固定种子的 torch.Generator"""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# %% 数据集定义
class NpyDataset(Dataset):
    """训练数据集类"""

    def __init__(self, data_root, img_size=1024):
        self.data_root = data_root
        self.img_size = img_size

        # 获取图像和掩码路径
        self.gt_path = os.path.join(data_root, "GT/")
        self.img_path = os.path.join(data_root, "Imgs/")

        self.gt_path_files = sorted([
            os.path.join(self.gt_path, f)
            for f in os.listdir(self.gt_path)
            if f.endswith('.png')
        ])
        self.img_path_files = sorted([
            os.path.join(self.img_path, f)
            for f in os.listdir(self.img_path)
            if f.endswith('.jpg') or f.endswith('.png')
        ])

        # 确保文件数量匹配
        assert len(self.gt_path_files) == len(self.img_path_files), \
            f"图像和掩码数量不匹配: {len(self.img_path_files)} vs {len(self.gt_path_files)}"

        print(f"数据集: {data_root}")
        print(f"图像数量: {len(self.img_path_files)}")

        # 图像转换
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 掩码转换
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])

    def __len__(self):
        return len(self.img_path_files)

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.img_path_files[idx]
        img_ori = Image.open(img_path).convert('RGB')

        # 加载掩码
        gt_path = self.gt_path_files[idx]
        gt = Image.open(gt_path).convert('L')

        # 应用转换
        img_tensor = self.img_transform(img_ori)
        gt_tensor = self.mask_transform(gt)

        # 返回原始图像用于VLM描述生成
        img_ori_array = np.array(img_ori)

        return img_tensor, gt_tensor, img_ori_array


# %% 评估函数
def eval_psnr(loader, model, vlm_model, processor, mamba_model, tokenizer, device, use_rankdice=True):
    """评估模型性能"""
    model.eval()
    model.set_training_mode(False)

    print(f"\n=== 开始评估 ===")
    pbar = tqdm(total=len(loader), leave=False, desc='评估进度')

    # 初始化评估指标
    mae, sm, em, wfm, m_dice, m_iou, ber = cal_mae(), cal_sm(), cal_em(), cal_wfm(), cal_dice(), cal_iou(), cal_ber()

    with torch.no_grad():
        for step, (image, gt2D, img_1024_ori) in enumerate(loader):
            image, gt2D = image.to(device), gt2D.to(device)
            img_1024_ori = img_1024_ori.to(device)

            # 获取VLM描述
            vlm_inputs = processor(img_1024_ori, return_tensors="pt").to(device)
            vlm_outputs = vlm_model.generate(**vlm_inputs)
            description = processor.decode(vlm_outputs[0], skip_special_tokens=True)

            # 提取文本特征
            mamba_inputs = tokenizer(description, padding=True, return_tensors="pt").to(device)
            mamba_outputs = mamba_model(**mamba_inputs)
            text_features = mamba_outputs.last_hidden_state

            # 提取图像特征
            vision_outputs = vlm_model.vision_model(**vlm_inputs)
            image_features = vision_outputs.last_hidden_state[:, 1:, :]

            # 处理图像特征形状
            batch_size, seq_len, hidden_dim = image_features.shape
            if seq_len == 576:  # BLIP 24x24
                image_features = image_features.reshape(batch_size, 24, 24, hidden_dim)
                image_features = image_features.permute(0, 3, 1, 2)
                image_features = F.interpolate(image_features, size=(64, 64), mode='bilinear', align_corners=False)
            elif seq_len == 64 * 64:  # 64x64
                image_features = image_features.reshape(batch_size, 64, 64, hidden_dim).permute(0, 3, 1, 2)
            else:
                image_features = image_features.mean(dim=1, keepdim=True)
                image_features = image_features.unsqueeze(-1)
                image_features = F.interpolate(image_features, size=(64, 64))

            # 模型推理
            pred_mask, _, _ = model(
                image=image,
                text_embeddings=text_features,
                image_features=image_features,
                gt_mask=None,
                hyper_cond={'threshold': 0.5},
                return_logits=False
            )

            # 转换为numpy
            pred = pred_mask.squeeze().cpu().numpy()
            gt = gt2D.squeeze().cpu().numpy()

            # 处理维度
            if pred.ndim == 0:
                pred = np.expand_dims(pred, 0)
            if gt.ndim == 0:
                gt = np.expand_dims(gt, 0)

            # 更新指标
            mae.update(pred, gt)
            sm.update(pred, gt)
            em.update(pred, gt)
            wfm.update(pred, gt)
            m_dice.update(pred, gt)
            m_iou.update(pred, gt)
            ber.update(pred, gt)

            pbar.update(1)
            pbar.set_description(f"评估步骤 {step + 1}/{len(loader)}")

    pbar.close()

    # 获取评估结果
    metrics = {
        'mae': mae.show(),
        'sm': sm.show(),
        'em': em.show(),
        'wfm': wfm.show(),
        'dice': m_dice.show(),
        'iou': m_iou.show(),
        'ber': ber.show(),
    }

    return metrics


# %% 参数解析器
def parse_args():
    parser = argparse.ArgumentParser(description="训练集成的MMSAM模型")

    # 数据参数
    parser.add_argument("--train_data", type=str,
                        default="/root/autodl-tmp/data/COD10K+CAMO/COD10K_CAMO_CombinedTrainingDataset",
                        help="训练数据路径")
    parser.add_argument("--val_data", type=str,
                        default="/root/autodl-tmp/data/COD10K+CAMO/COD10K_CAMO_CombinedTestingDataset/TestingDataset",
                        help="官方测试集路径（训练时不读取，仅保留用于最终测试）")
    parser.add_argument("--val_sample_size", type=int, default=404,
                        help="从训练集中固定划出的验证样本数（默认404）")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="训练/验证集固定划分种子；不同训练seed和对比方法应保持一致")

    # 模型参数
    parser.add_argument("--sam_checkpoint", type=str,
                        default="/root/autodl-tmp/MMsam/sam/sam_vit_h_4b8939.pth",
                        help="SAM预训练权重路径")
    parser.add_argument("--model_type", type=str, default="vit_h",
                        choices=["vit_b", "vit_l", "vit_h"],
                        help="SAM模型类型")
    parser.add_argument("--blip_path", type=str,
                        default="/root/autodl-tmp/MMsam/Blip",
                        help="BLIP模型路径")
    parser.add_argument("--mamba_path", type=str,
                        default="/root/autodl-tmp/MMsam/mamba",
                        help="Mamba模型路径")

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")
    parser.add_argument("--lr", type=float, default=0.00005,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")

    # 随机种子
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，默认42；多seed实验时分别传入 42/123/2024/3407/0 等")
    parser.add_argument("--deterministic", action="store_true",
                        help="启用 cuDNN 确定性模式，牺牲速度换可复现性")

    # 其他参数
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="训练设备")
    parser.add_argument("--use_amp", action="store_true",
                        help="使用混合精度训练")
    parser.add_argument("--use_wandb", action="store_true",
                        help="使用WandB记录")
    parser.add_argument("--resume", type=str, default="",
                        help="恢复训练的检查点路径")
    parser.add_argument("--work_dir", type=str, default=".\\work_dir",
                        help="工作目录")
    parser.add_argument("--task_name", type=str, default="mmsam_integrated",
                        help="任务名称")

    # 模型特定参数
    parser.add_argument("--n_streams", type=int, default=4,
                        help="流形约束适配器的流数量")
    parser.add_argument("--mca_bottleneck_dim", type=int, default=128,
                        help="三个插入位置共享的MCA瓶颈维度")
    parser.add_argument("--projection_rank", type=int, default=48,
                        help="BLIP对齐与跨模态提示投影的低秩维度")
    parser.add_argument("--use_rankdice", action="store_true", default=True,
                        help="使用RankDice-RMA模块")
    parser.add_argument("--use_hypercond", action="store_true", default=True,
                        help="使用超参数条件化")

    parser.add_argument("--cspg_temperature", type=float, default=1.0, help="Gram-affinity temperature; new implementation default")
    parser.add_argument("--cspg_iters", type=int, default=5, help="CSPG row/column normalization pairs")
    return parser.parse_args()


# %% 主函数
def main():
    args = parse_args()

    # === 设置随机种子（最早执行） ===
    set_seed(args.seed, deterministic=args.deterministic)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建工作目录：在 task_name 下按 seed 分子目录，便于多 seed 实验
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    seed_tag = f"seed{args.seed}"
    model_save_path = os.path.join(args.work_dir, args.task_name, seed_tag, run_id)
    os.makedirs(model_save_path, exist_ok=True)

    # 保存当前脚本
    shutil.copyfile(__file__, os.path.join(model_save_path, f"train_{run_id}.py"))

    # 保存参数与种子信息
    with open(os.path.join(model_save_path, "run_config.txt"), "w", encoding="utf-8") as f:
        f.write(f"seed = {args.seed}\n")
        f.write(f"deterministic = {args.deterministic}\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k} = {v}\n")

    # 初始化WandB（如果使用）
    if args.use_wandb:
        import wandb
        wandb.login()
        wandb.init(
            project=args.task_name,
            name=f"{args.task_name}_{seed_tag}_{run_id}",
            config=vars(args)
        )

    # %% 创建数据集和数据加载器
    print("加载数据集...")
    full_train_dataset = NpyDataset(args.train_data)

    if not 0 < args.val_sample_size < len(full_train_dataset):
        raise ValueError(
            f"val_sample_size 必须在 1 到 {len(full_train_dataset) - 1} 之间，"
            f"当前为 {args.val_sample_size}"
        )

    # 仅从训练集做一次固定划分；官方测试集不参与训练或选模。
    split_generator = make_generator(args.split_seed)
    indices = torch.randperm(len(full_train_dataset), generator=split_generator).tolist()
    val_indices = indices[:args.val_sample_size]
    train_indices = indices[args.val_sample_size:]
    train_dataset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)

    print(f"训练集划分: {len(train_dataset)} 张")
    print(f"验证集划分: {len(val_subset)} 张（split_seed={args.split_seed}，固定不变）")
    print(f"官方测试集: {args.val_data}（训练阶段不读取）")

    # 训练集 shuffle 使用独立 generator，避免影响固定的数据划分。
    train_generator = make_generator(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    # %% 加载VLM和文本模型
    print("加载VLM和文本模型...")
    processor = BlipProcessor.from_pretrained(args.blip_path)
    vlm_model = BlipForConditionalGeneration.from_pretrained(args.blip_path).to(device)
    vlm_model.eval()
    vlm_model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.mamba_path)
    mamba_model = MambaModel.from_pretrained(args.mamba_path).to(device)
    mamba_model.eval()
    mamba_model.requires_grad_(False)

    # %% 创建集成模型
    print("创建集成模型...")
    model = create_integrated_model(
        sam_checkpoint_path=args.sam_checkpoint,
        model_type=args.model_type,
        cspg_temperature=args.cspg_temperature,
        cspg_iters=args.cspg_iters,
        image_size=1024,
        use_rankdice=args.use_rankdice,
        use_hypercond=args.use_hypercond,
        n_streams=args.n_streams,
        mca_bottleneck_dim=args.mca_bottleneck_dim,
        projection_rank=args.projection_rank,
        device=device,
        blip_feature_dim=768
    )

    # 冻结SAM图像编码器的大部分参数（除了适配器）
    for name, param in model.image_encoder.named_parameters():
        if "adapter" in name or "blip_feature_adjust" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"训练参数比例: {trainable_params / total_params * 100:.2f}%")

    # 分模块核对参数预算，vit_h 默认配置目标约 5.1M。
    budget_modules = {
        "SAM mask decoder": model.mask_decoder,
        "共享 MCA": model.image_encoder.shared_manifold_adapter,
        "BLIP 低秩对齐": model.image_encoder.blip_feature_adjust,
        "CSPG 低秩提示": model.prompt_generator,
        "Dense prompt": model.pseudo_mask_embed,
    }
    if args.use_hypercond:
        budget_modules.update({
            "HyperCond": model.hyper_cond,
            "HyperCond channel": model.cond_channel_adapter,
            "HyperCond spatial": model.cond_spatial_adapter,
        })
    print("可训练参数分模块统计:")
    for module_name, module in budget_modules.items():
        module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {module_name}: {module_params:,}")

    # %% 创建优化器和损失函数
    print("创建优化器和损失函数...")
    optimizer = create_optimizer_for_integrated_model(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * len(train_loader),
        eta_min=1e-6
    )

    # 损失函数
    seg_loss = BoundaryAwareLoss(alpha=1.0, beta=0.1)

    # Beta分布用于超参数采样（使用带 seed 的 generator，保证采样一致）
    beta_dist = Beta(
        torch.tensor([2.0]),
        torch.tensor([2.0])
    )

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # %% 恢复训练（如果指定）
    start_epoch = 0
    best_val_score = 0.0
    train_history = {
        'train_loss': [],
        'train_rank_dice_loss': [],
        'train_boundary_loss': [],
        'val_metrics': [],
        'learning_rates': []
    }

    if args.resume and os.path.isfile(args.resume):
        print(f"恢复训练从: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        start_epoch = checkpoint.get('epoch', 0) + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'best_val_score' in checkpoint:
            best_val_score = checkpoint['best_val_score']

        if 'train_history' in checkpoint:
            train_history = checkpoint['train_history']

        print(f"恢复训练: epoch {start_epoch}, 最佳验证分数: {best_val_score:.4f}")

    # %% 训练循环
    print("开始训练...")
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}  (seed={args.seed})")
        print(f"{'=' * 50}")

        # 训练阶段
        model.train()
        model.set_training_mode(True)

        epoch_train_loss = 0.0
        epoch_rank_dice_loss = 0.0
        epoch_boundary_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f'训练 (Epoch {epoch + 1}/{args.num_epochs})',
                          ncols=120, leave=False)

        for step, (image, gt2D, img_1024_ori) in enumerate(train_pbar):
            optimizer.zero_grad()

            # 移动数据到设备
            image, gt2D = image.to(device), gt2D.to(device)
            img_1024_ori = img_1024_ori.to(device)

            # === 超参数条件采样 ===
            tau_sample = beta_dist.sample().item()
            tau = 0.4 + tau_sample * 0.2  # [0.4, 0.6]

            lambda_sample = beta_dist.sample().item()
            boundary_weight = 0.3 + lambda_sample * 0.7  # [0.3, 1.0]

            hyper_cond = {
                'threshold': tau,
            }

            # === 获取VLM描述和特征 ===
            with torch.no_grad():
                # 生成描述
                vlm_inputs = processor(img_1024_ori, return_tensors="pt").to(device)
                vlm_outputs = vlm_model.generate(**vlm_inputs)
                description = processor.decode(vlm_outputs[0], skip_special_tokens=True)

                # 提取文本特征
                mamba_inputs = tokenizer(description, padding=True, return_tensors="pt").to(device)
                mamba_outputs = mamba_model(**mamba_inputs)
                text_features = mamba_outputs.last_hidden_state

                # 提取图像特征
                vision_outputs = vlm_model.vision_model(**vlm_inputs)
                image_features_raw = vision_outputs.last_hidden_state[:, 1:, :]

            # 处理图像特征形状
            batch_size, seq_len, hidden_dim = image_features_raw.shape
            if seq_len == 576:  # BLIP 24x24
                image_features = image_features_raw.reshape(batch_size, 24, 24, hidden_dim)
                image_features = image_features.permute(0, 3, 1, 2)
                image_features = F.interpolate(image_features, size=(64, 64), mode='bilinear', align_corners=False)
            elif seq_len == 64 * 64:  # 64x64
                image_features = image_features_raw.reshape(batch_size, 64, 64, hidden_dim).permute(0, 3, 1, 2)
            else:
                image_features = image_features_raw.mean(dim=1, keepdim=True)
                image_features = image_features.unsqueeze(-1)
                image_features = F.interpolate(image_features, size=(64, 64))

            # === 模型前向传播 ===
            if args.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, rank_dice_loss, _ = model(
                        image=image,
                        text_embeddings=text_features,
                        image_features=image_features,
                        gt_mask=gt2D,
                        hyper_cond=hyper_cond,
                        return_logits=True
                    )

                    # 计算损失
                    total_loss, dice_ce_loss, boundary_loss = seg_loss(
                        logits, gt2D, boundary_weight
                    )


                # 反向传播
                scaler.scale(total_loss).backward()

                # 应用梯度约束
                model.apply_gradient_constraints()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                logits, rank_dice_loss, _ = model(
                    image=image,
                    text_embeddings=text_features,
                    image_features=image_features,
                    gt_mask=gt2D,
                    hyper_cond=hyper_cond,
                    return_logits=True
                )

                # 计算损失
                total_loss, dice_ce_loss, boundary_loss = seg_loss(
                    logits, gt2D, boundary_weight
                )


                # 反向传播
                total_loss.backward()

                # 应用梯度约束
                model.apply_gradient_constraints()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            # 清除梯度
            optimizer.zero_grad()

            # 更新学习率
            scheduler.step()

            # 记录损失
            epoch_train_loss += total_loss.item()
            epoch_rank_dice_loss += rank_dice_loss.item() if isinstance(rank_dice_loss,
                                                                        torch.Tensor) else rank_dice_loss
            epoch_boundary_loss += boundary_loss.item()

            # 更新进度条
            current_lr = optimizer.param_groups[0]['lr']
            # 确保 rank_dice_loss 是标量
            if isinstance(rank_dice_loss, torch.Tensor):
                rank_dice_value = rank_dice_loss.item()
            else:
                rank_dice_value = rank_dice_loss

            # 确保 boundary_loss 是标量
            if isinstance(boundary_loss, torch.Tensor):
                boundary_value = boundary_loss.item()
            else:
                boundary_value = boundary_loss

            # 简化显示，减少字段数量
            train_pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'r_dice': f'{rank_dice_value:.4f}',  # 缩写为 r_dice
                'b_loss': f'{boundary_value:.4f}',  # 缩写为 b_loss
                'lr': f'{current_lr:.2e}',
                # 移除 tau 和 lambda 以节省空间
            })

        # 计算平均训练损失
        epoch_train_loss /= len(train_loader)
        epoch_rank_dice_loss /= len(train_loader)
        epoch_boundary_loss /= len(train_loader)

        # 记录训练历史
        train_history['train_loss'].append(epoch_train_loss)
        train_history['train_rank_dice_loss'].append(epoch_rank_dice_loss)
        train_history['train_boundary_loss'].append(epoch_boundary_loss)
        train_history['learning_rates'].append(current_lr)

        print(f"\n训练统计:")
        print(f"  总损失: {epoch_train_loss:.4f}")
        print(f"  RankDice损失: {epoch_rank_dice_loss:.4f}")
        print(f"  边界损失: {epoch_boundary_loss:.4f}")
        print(f"  学习率: {current_lr:.2e}")

        # === 验证阶段 ===
        print(f"\n开始验证...")
        val_metrics = eval_psnr(
            val_loader, model, vlm_model, processor,
            mamba_model, tokenizer, device, args.use_rankdice
        )

        train_history['val_metrics'].append(val_metrics)

        print(f"\n验证指标 (Epoch {epoch + 1}):")
        print(f"  S-measure: {val_metrics['sm']:.4f}")
        print(f"  E-measure: {val_metrics['em']:.4f}")
        print(f"  wF-measure: {val_metrics['wfm']:.4f}")
        print(f"  MAE: {val_metrics['mae']:.4f}")
        print(f"  Dice: {val_metrics['dice']:.4f}")
        print(f"  IoU: {val_metrics['iou']:.4f}")
        print(f"  BER: {val_metrics['ber']:.4f}")

        # 计算综合分数
        val_score = (val_metrics['sm'] + val_metrics['em'] + val_metrics['wfm']) / 3

        # 记录到WandB
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "train_rank_dice_loss": epoch_rank_dice_loss,
                "train_boundary_loss": epoch_boundary_loss,
                "learning_rate": current_lr,
                "val_sm": val_metrics['sm'],
                "val_em": val_metrics['em'],
                "val_wfm": val_metrics['wfm'],
                "val_mae": val_metrics['mae'],
                "val_dice": val_metrics['dice'],
                "val_iou": val_metrics['iou'],
                "val_ber": val_metrics['ber'],
                "val_score": val_score,
                "seed": args.seed,
            })

        # === 保存模型 ===
        previous_best_score = best_val_score
        is_best = val_score > best_val_score
        if is_best:
            best_val_score = val_score

        # 保存最新模型
        checkpoint_latest = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': epoch_train_loss,
            'val_metrics': val_metrics,
            'val_score': val_score,
            'best_val_score': best_val_score,
            'train_history': train_history,
            'args': vars(args),
            'seed': args.seed,
        }

        torch.save(checkpoint_latest, os.path.join(model_save_path, "model_latest.pth"))

        # 保存最佳模型
        if is_best:
            checkpoint_best = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': epoch_train_loss,
                'val_metrics': val_metrics,
                'val_score': val_score,
                'best_val_score': best_val_score,
                'train_history': train_history,
                'args': vars(args),
                'seed': args.seed,
            }

            torch.save(checkpoint_best, os.path.join(model_save_path, "model_best.pth"))
            print(f"✅ 保存最佳模型，验证分数: {val_score:.4f} (之前最佳: {previous_best_score:.4f})")

        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_epoch = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': epoch_train_loss,
                'val_metrics': val_metrics,
                'val_score': val_score,
                'best_val_score': best_val_score,
                'train_history': train_history,
                'args': vars(args),
                'seed': args.seed,
            }

            torch.save(checkpoint_epoch, os.path.join(model_save_path, f"model_epoch_{epoch + 1}.pth"))

        # === 绘制训练曲线 ===
        plt.figure(figsize=(15, 10))

        # 损失曲线
        plt.subplot(2, 3, 1)
        plt.plot(train_history['train_loss'], label='Total Loss')
        plt.plot(train_history['train_rank_dice_loss'], label='RankDice Loss')
        plt.plot(train_history['train_boundary_loss'], label='Boundary Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss (seed={args.seed})')
        plt.legend()
        plt.grid(True)

        # 验证指标曲线
        plt.subplot(2, 3, 2)
        val_sm = [m['sm'] for m in train_history['val_metrics']]
        val_em = [m['em'] for m in train_history['val_metrics']]
        val_wfm = [m['wfm'] for m in train_history['val_metrics']]
        plt.plot(val_sm, label='S-measure')
        plt.plot(val_em, label='E-measure')
        plt.plot(val_wfm, label='wF-measure')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(True)

        # MAE曲线
        plt.subplot(2, 3, 3)
        val_mae = [m['mae'] for m in train_history['val_metrics']]
        plt.plot(val_mae, label='MAE', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Mean Absolute Error')
        plt.legend()
        plt.grid(True)

        # Dice和IoU曲线
        plt.subplot(2, 3, 4)
        val_dice = [m['dice'] for m in train_history['val_metrics']]
        val_iou = [m['iou'] for m in train_history['val_metrics']]
        plt.plot(val_dice, label='Dice Coefficient')
        plt.plot(val_iou, label='IoU')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Segmentation Metrics')
        plt.legend()
        plt.grid(True)

        # 学习率曲线
        plt.subplot(2, 3, 5)
        plt.plot(train_history['learning_rates'], label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')

        # BER曲线
        plt.subplot(2, 3, 6)
        val_ber = [m['ber'] for m in train_history['val_metrics']]
        plt.plot(val_ber, label='BER', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('BER')
        plt.title('Balanced Error Rate')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(model_save_path, "training_curves.png"), dpi=150)
        plt.close()

        print(f"Epoch {epoch + 1} 完成，训练曲线已保存")

    # 训练完成
    print(f"\n{'=' * 50}")
    print(f"训练完成!")
    print(f"Seed: {args.seed}")
    print(f"最佳验证分数: {best_val_score:.4f}")
    print(f"模型保存至: {model_save_path}")
    print(f"{'=' * 50}")

    # 关闭WandB
    if args.use_wandb:
        wandb.finish()


# %% 执行
if __name__ == "__main__":
    main()
