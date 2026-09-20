# MC-SAM：流形约束增强的 Segment Anything Model

> **M**anifold-**C**onstrained **SAM** — 基于 Vision-Language SAM (VLSAM) 框架，集成四项核心改进模块，面向伪装目标检测（Camouflaged Object Detection, COD）任务。

---

## 目录

- [项目总览](#项目总览)
- [架构设计](#架构设计)
- [环境依赖](#环境依赖)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
  - [训练](#训练)
  - [推理](#推理)
  - [验证修复](#验证修复)
- [核心模块详解](#核心模块详解)
  - [1. 流形约束多层适配器](#1-流形约束多层适配器-manifoldconstrainedadapter)
  - [2. 跨空间稳定 Vision-Language Prompt 生成器](#2-跨空间稳定-vision-language-prompt-生成器-crossmodalstablepromptgenerator)
  - [3. RankDice-RMA 后处理模块](#3-rankdice-rma-后处理模块-rankdicermamodule)
  - [4. 超参数条件化模块](#4-超参数条件化模块-hypercondmodule)
- [训练策略](#训练策略)
- [项目结构](#项目结构)
- [配置参数](#配置参数)
- [常见问题](#常见问题)

---

## 项目总览

MC-SAM 在原始 VLSAM（SAM + BLIP + Mamba）的基础上，引入四项改进：

| 模块 | 核心思想 | 作用位置 |
|------|----------|----------|
| **流形约束适配器** | Sinkhorn 双随机矩阵约束的多流残差融合 | ViT 编码器的 1/4, 1/2, 3/4 深度层 |
| **跨模态 Prompt 生成器** | 文本/视觉特征经流形约束混合生成多 token sparse prompt | Mask Decoder 输入 |
| **RankDice-RMA** | 基于排序的 Dice 损失 + 推理时自适应阈值 | 训练损失 & 推理后处理 |
| **超参数条件化** | 将阈值/边界权重编码为条件嵌入注入 dense prompt | Dense Embedding 融合 |

### 数据流全景

```
输入图像 [B,3,1024,1024]
    │
    ├──→ BLIP Vision → image_features [B,768,64,64]
    │                         │
    │                    ┌────┴────┐
    │                    ↓         ↓
    │          IntegratedImageEncoderViT
    │          (SAM ViT + ManifoldConstrainedAdapter ×3)
    │                    │
    │              image_embedding [B,256,64,64]
    │                    │
    ├──→ BLIP Caption → Mamba → text_features [B,seq,768]
    │         │                       │
    │         │    ┌──────────────────┘
    │         ↓    ↓
    │   CrossModalStablePromptGenerator
    │         │
    │    sparse_prompt [B,2,256]
    │         │
    │    ┌────┴────┐
    │    │ HyperCondModule(τ, λ) → cond_embedding
    │    │         │
    │    │   dense_embeddings + 0.1·cond
    │    │         │
    │    └────┬────┘
    │         ↓
    │    SAM MaskDecoder
    │         │
    │    logits [B,1,1024,1024]
    │         │
    └──→ RankDice-RMA (训练: loss / 推理: 自适应阈值)
              │
         pred_mask [B,1,1024,1024]
```

---

## 架构设计

```
mcsam_integrated.py 模块依赖关系
─────────────────────────────────

SinkhornProjection          ← ManifoldConstrainedAdapter (内部调用)
                            ← CrossModalStablePromptGenerator (内部调用)

RMSNorm                     ← ManifoldConstrainedAdapter (归一化)
                            ← CrossModalStablePromptGenerator (归一化)

ManifoldConstrainedAdapter  ← IntegratedImageEncoderViT (在 blocks 间注入)

CrossModalStablePromptGenerator ← MMSAM_Integrated (生成 sparse prompt)

HyperCondModule             ← MMSAM_Integrated (生成条件嵌入)

RankDiceRMAModule           ← MMSAM_Integrated (训练 loss / 推理阈值)

BoundaryAwareLoss           ← MCsam_train.py (训练损失函数)

IntegratedImageEncoderViT   ← MMSAM_Integrated (图像编码器)

MMSAM_Integrated            ← create_integrated_model() 工厂函数创建
```

---

## 环境依赖

### 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU 显存 | 16 GB (vit_b) | 24+ GB (vit_l / vit_h) |
| 内存 | 16 GB | 32 GB |
| 存储 | 10 GB | 20 GB（含数据集） |

### 软件依赖

```bash
# Python >= 3.8
pip install torch torchvision          # PyTorch >= 1.12
pip install monai                       # 医学图像 Dice 损失
pip install transformers                # BLIP + Mamba
pip install tqdm matplotlib Pillow numpy
pip install wandb                       # (可选) 实验记录
```

### 预训练模型

| 模型 | 路径（AutoDL 示例） | 用途 |
|------|---------------------|------|
| SAM ViT-L | `/root/autodl-tmp/MMsam/sam/sam_vit_l_0b3195.pth` | 图像编码器 + Mask Decoder |
| BLIP | `/root/autodl-tmp/MMsam/Blip` | 图像描述 + 视觉特征 |
| Mamba | `/root/autodl-tmp/MMsam/mamba` | 文本特征提取 |

---

## 数据准备

数据集需要按以下结构组织：

```
Dataset/
├── Imgs/           # 输入图像 (.jpg 或 .png)
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
└── GT/             # Ground Truth 掩码 (.png, 灰度图)
    ├── 0001.png
    ├── 0002.png
    └── ...
```

> **注意**：图像和掩码文件按字典序排列后必须一一对应。掩码为灰度 PNG，前景=白色(255)，背景=黑色(0)。

支持的数据集：COD10K、CAMO、NC4K、CHAMELEON 等伪装目标检测数据集。

---

## 快速开始

### 训练

```bash
cd MC-SAM

python MCsam_train.py \
    --train_data /path/to/train/dataset \
    --val_data /path/to/val/dataset \
    --sam_checkpoint /path/to/sam_vit_l_0b3195.pth \
    --model_type vit_l \
    --blip_path /path/to/Blip \
    --mamba_path /path/to/mamba \
    --num_epochs 20 \
    --batch_size 1 \
    --lr 0.00005 \
    --device cuda:0 \
    --work_dir ./work_dir \
    --task_name mcsam_cod
```

#### 关键训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr` | 5e-5 | 基础学习率（各模块有差异化倍率） |
| `--num_epochs` | 20 | 训练轮数 |
| `--batch_size` | 1 | 批次大小（受限于显存） |
| `--use_rankdice` | True | 启用 RankDice-RMA 模块 |
| `--use_hypercond` | True | 启用超参数条件化 |
| `--n_streams` | 4 | 流形适配器的残差流数量 |
| `--use_amp` | False | 启用混合精度训练（节省显存） |
| `--use_wandb` | False | 启用 Weights & Biases 日志 |
| `--resume` | "" | 从检查点恢复训练 |

#### 恢复训练

```bash
python MCsam_train.py \
    --resume ./work_dir/mcsam_cod/20260210-1300/best_model.pth \
    ... # 其余参数同上
```

### 推理

```bash
python inference_mcsam.py \
    --model_path ./work_dir/mcsam_cod/best_model.pth \
    --data_path /path/to/test/dataset \
    --sam_checkpoint /path/to/sam_vit_l_0b3195.pth \
    --model_type vit_l \
    --save_dir ./results \
    --threshold 0.5 \
    --boundary_weight 1.0 \
    --visualize
```

输出保存在 `--save_dir` 目录，包括预测掩码 PNG 和（可选）可视化对比图。

### 验证修复

运行修复验证脚本，确认所有模块正常工作（无需 GPU）：

```bash
python test_fixes.py
```

期望输出：所有 11 个测试区段通过（`✅`），共约 30 项检查。

---

## 核心模块详解

### 1. 流形约束多层适配器 (`ManifoldConstrainedAdapter`)

**核心文件**: `segment_anything/modeling/mcsam_integrated.py`

**设计目标**: 在 SAM ViT 编码器内部，将 BLIP 视觉特征与 SAM 特征进行受约束的多流残差融合，保持信号范数守恒。

**工作流程**:

```
SAM特征 [B,N,C] ──┐
                   ├──→ concat [B,N,2C] → input_adjust → RMSNorm → combined [B,N,C]
BLIP特征 [B,N,C] ─┘
                   │
            ┌──────┴──────┐
            ↓              ↓
      adapter_mlp      mixing_mlp
      [B,N,4,C]        [B,N,4,4]
      (4路残差流)        │
            │        Sinkhorn(20次迭代)
            │           ↓
            │        H_res [B,N,4,4] (双随机矩阵)
            │           │
            └─── einsum('bnsa,bnac->bnsc') ───→ mixed [B,N,4,C]
                        │
                  softmax(alpha) 门控 → final_residual [B,N,C]
                        │
              SAM特征 + 0.1 × final_residual → 输出
```

**关键设计点**:
- **Sinkhorn 投影**: 生成双随机矩阵 H_res，确保行列和均为 1，保证信号范数守恒
- **小残差权重 0.1**: 防止初始阶段适配器干扰 SAM 预训练特征
- **门控机制 alpha**: 初始化为 0.01，训练过程中自适应学习各流重要性
- **注入深度**: 在 ViT 的 1/4、1/2、3/4 深度处各注入一个 adapter，共 3 个

### 2. 跨空间稳定 Vision-Language Prompt 生成器 (`CrossModalStablePromptGenerator`)

**设计目标**: 将文本（Mamba）和视觉（BLIP）全局特征融合为 SAM Mask Decoder 所需的 sparse prompt，使用流形约束保证数值稳定。

**工作流程**:

```
text_global [B,768] ──→ W_text ──→ text_proj [B,256]  ──┐
                                                          ├→ stack [B,2,256]
vision_global [B,768] → W_vision → vision_proj [B,256] ──┘
                                                          │
                      modality_controller ──→ H_pre [B,2,2] (Sinkhorn约束)
                                              │
                                  einsum('bnk,bkd->bnd')
                                              │
                                  mixed_streams [B,2,256]
                                              │
                                  H_post (stream_weights softmax)
                                              │
                                  RMSNorm per token
                                              │
                                  sparse_prompt [B, 2, 256]
```

**关键设计点**:
- **双 token 输出**: 生成 `n_prompt=2` 个 prompt token，而非合并为单一向量，与 SAM Mask Decoder 的多 token 设计对齐
- **模态权重控制器**: 自适应学习文本 vs 视觉特征的混合比例
- **范围限制 [0.4, 0.6]**: 通过 sigmoid 约束，防止某一模态完全主导

### 3. RankDice-RMA 后处理模块 (`RankDiceRMAModule`)

**训练时**: 计算基于排序的 Dice loss（可微分软阈值），作为辅助损失引导模型学习更好的概率分布。

```
logits → sigmoid → prob_map
                     │
              sorted desc → cumsum → RMA(τ) → argmax → τ*
                     │                                    │
              sigmoid((prob - τ*) / temperature) → soft_mask
                     │
              Dice(soft_mask, gt) × weight → rank_dice_loss
```

**推理时**: 自动搜索最优阈值 τ*，替代固定的 0.5 阈值，对小目标和边界区域特别有效。

**关键设计点**:
- **可学习温度**: `self.temperature` 控制软阈值的锐度，初始 0.1
- **梯度可回传**: 使用 `sigmoid` 软阈值而非硬阈值 `(p >= τ).float()`

### 4. 超参数条件化模块 (`HyperCondModule`)

**设计目标**: 将阈值 τ 和边界权重 λ 编码为条件向量，注入 dense prompt，使模型对不同超参数组合具备适应性。

```
τ ──→ threshold_encoder ──→ [B, 64]  ──┐
                                        ├→ concat [B, 128]
λ ──→ boundary_encoder ──→ [B, 64]  ──┘
                                        │
                                  fusion_net → Tanh
                                        │
                              cond_embedding [B, 128]
                                        │
                          Conv2d(128→256) → interpolate → Conv2d+GELU
                                        │
                          dense_embeddings + 0.1 × cond_dense
```

**训练策略**: 每个 batch 从 Beta(2,2) 分布采样 τ ∈ [0.4, 0.6]、λ ∈ [0.3, 1.0]，使模型学习对不同超参数的鲁棒响应。

---

## 训练策略

### 差异化学习率

| 参数组 | 学习率倍率 | 说明 |
|--------|-----------|------|
| encoder_adapters | 1.0× | 流形适配器 + blip_feature_adjust |
| prompt_generator | 0.5× | 跨模态 Prompt 生成器 |
| hyper_cond | 0.1× | 超参数条件化 + cond 适配器 |
| rankdice | 0.1× | RankDice-RMA 模块 |
| others | 0.01× | 其余可训练参数 |

### 损失函数

```
Total Loss = BoundaryAwareLoss(logits, gt) + RankDice_loss
           = [DiceLoss + BCEWithLogitsLoss + β·BoundaryLoss] + RankDice_loss
```

### 参数冻结策略

- **完全冻结**: SAM ViT 编码器的 patch_embed、pos_embed、transformer blocks、neck
- **可训练**: manifold_adapters、blip_feature_adjust、prompt_generator、hyper_cond、rankdice_module、text_adapter、mask_decoder 的部分参数

---

## 项目结构

```
MC-SAM/
├── MCsam_train.py                    # 训练脚本（主入口）
├── inference_mcsam.py                # 推理脚本
├── config_mcsam.py                   # 配置文件（IntegratedModelConfig）
├── test_fixes.py                     # 修复验证脚本（11个测试区段）
├── README.md                         # 本文件
│
├── segment_anything/                 # SAM 核心库
│   ├── __init__.py
│   ├── build_sam.py                  # SAM 模型注册表
│   ├── predictor.py                  # SAM 预测器
│   ├── automatic_mask_generator.py   # 自动掩码生成器
│   ├── modeling/
│   │   ├── mcsam_integrated.py       # ★ MC-SAM 核心（所有改进模块）
│   │   ├── image_encoder.py          # SAM ViT 编码器（原版）
│   │   ├── mask_decoder.py           # SAM Mask Decoder
│   │   ├── prompt_encoder.py         # SAM Prompt Encoder
│   │   ├── transformer.py            # SAM 双向 Transformer
│   │   └── common.py                 # 公共组件（LayerNorm2d, MLPBlock）
│   └── utils/
│       ├── transforms.py             # 输入变换
│       └── amg.py                    # 自动掩码生成工具
│
├── utils_downstream/                 # 下游评估工具
│   ├── saliency_metric.py            # 评估指标（MAE, S-m, E-m, wF-m, Dice, IoU, BER）
│   ├── dataset_rgbd_strategy2.py     # 数据集加载策略
│   ├── ssim_loss.py                  # SSIM 损失
│   ├── config.py                     # 下游配置
│   └── misc.py                       # 工具函数
│
└── work_dir/                         # 训练输出（自动创建）
    └── <task_name>/
        └── <run_id>/
            ├── best_model.pth        # 最优模型
            ├── latest_model.pth      # 最新模型
            ├── train_*.py            # 训练脚本备份
            └── training_curves.png   # 损失曲线
```

---

## 配置参数

`config_mcsam.py` 中的 `IntegratedModelConfig` 数据类定义了全部配置：

```python
from config_mcsam import IntegratedModelConfig

config = IntegratedModelConfig(
    model_type="vit_l",       # SAM 骨干：vit_b / vit_l / vit_h
    image_size=1024,          # 输入图像尺寸
    use_rankdice=True,        # 启用 RankDice-RMA
    use_hypercond=True,       # 启用超参数条件化
    n_streams=4,              # 适配器残差流数量
    prompt_dim=256,           # Prompt 嵌入维度
    cond_dim=128,             # 条件嵌入维度
    batch_size=1,
    num_epochs=20,
    learning_rate=0.0002,
    weight_decay=0.01,
)
```

---

## 常见问题

### Q: 训练时显存不足？

1. 使用 `--use_amp` 启用混合精度（节省约 30% 显存）
2. 确认 `--batch_size 1`
3. 降级为 `--model_type vit_b`（显存需求 ~16 GB）

### Q: 如何确认 SAM 权重正确加载？

训练开始时查看日志：
```
[C1修复] 从SAM预训练模型成功迁移了 X/Y 个参数
```
- `vit_l` 应迁移约 600+ 个参数
- 如果看到大量跳过，请检查编码器配置是否与 checkpoint 匹配

### Q: 评估指标模块未安装？

训练脚本会自动降级为简化指标。如需完整指标，确保 `utils_downstream/saliency_metric.py` 中的 `cal_mae` 等类正确导出。

### Q: 如何单独关闭某个改进模块？

```bash
# 关闭 RankDice
python MCsam_train.py --no-use_rankdice ...

# 关闭 HyperCond
python MCsam_train.py --no-use_hypercond ...
```

或在代码中：
```python
model = create_integrated_model(
    ...,
    use_rankdice=False,
    use_hypercond=False,
)
```

### Q: 如何在新数据集上微调？

1. 按上述格式组织 `Imgs/` 和 `GT/` 目录
2. 修改 `--train_data` 和 `--val_data` 路径
3. 建议从已有检查点恢复：`--resume /path/to/best_model.pth`
4. 适当降低学习率：`--lr 0.00001`

---

## 引用

本项目基于以下工作：

- **SAM**: Kirillov et al., "Segment Anything," *ICCV 2023*
- **BLIP**: Li et al., "BLIP: Bootstrapping Language-Image Pre-training," *ICML 2022*
- **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," *2023*


## Source snapshot

This upload preserves the supplied MCsam source version. Datasets, pretrained weights, training outputs and IDE caches are not included. Configure dataset and checkpoint paths before running. GPU training and paper-result reproduction have not been verified for this snapshot. Original third-party source notices are retained.
