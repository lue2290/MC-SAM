# MC-SAM

**用于伪装目标分割的稳定性约束 SAM 适配研究。**

A research implementation of **MC-SAM: A Stability-Constrained Coupled Adaptation Framework for SAM in Camouflaged Scene Segmentation**.

伪装目标与背景具有相似的纹理和外观，稳定地融合视觉与文本信息是分割过程中的关键问题。本项目在 SAM 视觉分割框架中引入多流融合适配、跨模态提示、超参数条件化以及 RankDice-RMA 模块，提供模型训练、检查点加载与掩码推理源码。

## 核心组成

| 组件 | 代码中的功能 |
| --- | --- |
| SAM | 图像编码与掩码解码 |
| BLIP 与 Mamba | 视觉/文本描述及文本特征处理 |
| Sinkhorn 约束融合 | 用双随机混合权重构建多流特征融合 |
| 跨模态提示生成器 | 融合文本和视觉信息，生成分割提示 |
| HyperCond | 将阈值与边界权重编码到模型条件信息中 |
| RankDice-RMA | 排序相关损失与推理后处理 |

这些模块的主要实现位于 `segment_anything/modeling/mcsam_integrated.py`。源码版本的详细结构见 [实现笔记](docs/IMPLEMENTATION_NOTES.md)；笔记中的预期测试结果不代表本次已执行验证。

## 仓库结构

```text
MCsam_train.py          # 训练、数据读取、验证与检查点保存
inference_mcsam.py      # 模型加载、掩码预测与可视化
config_mcsam.py         # 集成模型配置
segment_anything/      # SAM 组件及集成模型
utils_downstream/      # 数据工具、损失与评估指标
test_fixes.py          # 原项目的模块检查脚本
docs/                  # 原版本实现笔记
```

## 环境与模型准备

请使用独立 Python 环境，并根据 CUDA 环境安装 PyTorch / torchvision。训练还使用 `transformers`、`monai`、NumPy、SciPy、Pillow、matplotlib 和 tqdm；可选实验记录需要 wandb。原环境未提供锁定依赖文件，因此这里不声称任意最新版依赖都兼容。

训练前准备：

1. SAM 预训练权重，与 `--model_type` 一致。
2. BLIP 和 Mamba 的本地模型目录。
3. 配对图像与二值分割标签。

```text
dataset/
├── Imgs/
│   ├── 0001.jpg
│   └── 0002.jpg
└── GT/
    ├── 0001.png
    └── 0002.png
```

图像与掩码必须在文件排序后正确对应；运行前请检查配对关系。数据集和预训练/训练权重不包含在仓库内。

## 训练

下面为 Bash 示例；Windows PowerShell 可将参数写为单行：

```bash
python MCsam_train.py \
  --train_data /path/to/train \
  --val_data /path/to/val \
  --sam_checkpoint /path/to/sam_vit_l_0b3195.pth \
  --model_type vit_l \
  --blip_path /path/to/Blip \
  --mamba_path /path/to/mamba \
  --num_epochs 20 --batch_size 1 --lr 0.00005 \
  --device cuda:0 --work_dir ./work_dir --task_name mcsam_cod
```

可使用 `--resume` 指定检查点恢复训练，使用 `--use_amp` 启用混合精度。完整参数以 `python MCsam_train.py --help` 为准。

## 推理

```bash
python inference_mcsam.py \
  --model_path /path/to/trained_model.pth \
  --data_path /path/to/test \
  --sam_checkpoint /path/to/sam_vit_l_0b3195.pth \
  --model_type vit_l --save_dir ./results --visualize
```

在运行前检查推理脚本中 BLIP/Mamba 的模型加载设置，使其与本地模型目录及训练配置一致。预测掩码与可视化输出保存到指定结果目录。

## 版本与验证说明

此仓库保存作者指定的 `MCsam` 源码版本，并保留原算法实现。本次整理通过 Python 语法检查，尚未执行完整 GPU 训练或论文指标复现。未提供经本次验证的性能表，不将申请材料中的指标作为该源码快照的复现结果。

训练脚本含有评估模块导入失败时的占位指标回退逻辑；正式评估前必须确认真实指标模块成功加载，不能将回退值用于报告模型性能。

## Acknowledgements

The implementation builds on SAM and uses the PyTorch and Hugging Face ecosystems. Original source notices are retained; third-party components remain subject to their respective licenses.
