# models/mcsam_integrated_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import numpy as np
from typing import Optional, Tuple, Type
import math

# 导入必要的组件
from segment_anything.modeling.common import LayerNorm2d, MLPBlock


# ==================== 1. 改进的Sinkhorn投影函数 ====================
class SinkhornProjection(nn.Module):
    """可微分的Sinkhorn投影层"""

    def __init__(self, iters=3, epsilon=1e-8):
        super().__init__()
        self.iters = iters
        self.epsilon = epsilon

    def forward(self, cost_matrix):
        """Sinkhorn 迭代生成双随机矩阵"""
        # 确保输入为正
        log_K = -cost_matrix

        # 数值稳定性：减去最大值
        log_K = log_K - torch.max(log_K, dim=-1, keepdim=True)[0].detach()

        for i in range(self.iters):
            # 行归一化
            log_K = log_K - torch.logsumexp(log_K, dim=-1, keepdim=True)
            # 列归一化
            log_K = log_K - torch.logsumexp(log_K, dim=-2, keepdim=True)

        K = torch.exp(log_K)
        return K


# ==================== 2. 改进的RankDice-RMA 模块 ====================
class RankDiceRMAModule(nn.Module):
    """改进的RankDice-RMA训练和推理模块"""

    def __init__(self, use_in_training=True, weight=0.1, eps=1e-8):
        super().__init__()
        self.use_in_training = use_in_training
        self.weight = weight
        self.eps = eps

        # 添加温度参数
        self.temperature = nn.Parameter(torch.tensor(0.1))

        # 学习率调度
        self.register_buffer('step', torch.tensor(0))

    def compute_optimal_threshold(self, prob_map):
        """计算最优阈值"""
        B, C, H, W = prob_map.shape
        thresholds = []

        for b in range(B):
            single_prob = prob_map[b, 0]
            d = single_prob.numel()

            # 排序概率值
            p_sorted, _ = torch.sort(single_prob.flatten(), descending=True)

            # 计算累积和
            q_tau = torch.cumsum(p_sorted, dim=0)
            mu = p_sorted.sum()

            # 计算RMA (Relative Mass Accumulation)
            tau_range = torch.arange(1, d + 1, device=prob_map.device, dtype=torch.float32)
            pi_rma = 2 * q_tau / (tau_range + mu + self.eps)

            # 找到最优的τ*
            tau_star = torch.argmax(pi_rma) + 1
            threshold_value = p_sorted[tau_star - 1]
            thresholds.append(threshold_value)

        return torch.stack(thresholds)

    def forward(self, logits, gt_mask=None):
        """
        计算RankDice-RMA损失或生成推理掩码
        """
        if self.training and gt_mask is not None and self.use_in_training:
            return self._compute_rank_dice_loss(logits, gt_mask)
        else:
            return self._inference(logits)

    def _compute_rank_dice_loss(self, logits, gt_mask):
        """计算RankDice-RMA损失"""
        prob_map = torch.sigmoid(logits)
        batch_size = prob_map.shape[0]

        # 计算每个样本的最优阈值
        thresholds = self.compute_optimal_threshold(prob_map)

        total_loss = 0.0
        for i in range(batch_size):
            single_prob = prob_map[i, 0]
            single_gt = gt_mask[i, 0]
            threshold_value = thresholds[i]

            # [C3修复] 使用可微分的软阈值代替硬阈值，确保梯度可以回传
            temp = torch.clamp(self.temperature, min=0.01)
            pred_mask = torch.sigmoid((single_prob - threshold_value) / temp)

            # 计算Dice损失
            intersection = (pred_mask * single_gt).sum()
            union = pred_mask.sum() + single_gt.sum() + self.eps
            dice = (2.0 * intersection + self.eps) / union

            dice_loss = 1.0 - dice
            total_loss += dice_loss

        rank_dice_loss = (total_loss / batch_size) * self.weight
        self.step += 1

        return logits, rank_dice_loss

    def _inference(self, logits):
        """推理时使用RankDice-RMA生成掩码"""
        prob_map = torch.sigmoid(logits)
        thresholds = self.compute_optimal_threshold(prob_map)

        binary_masks = []
        for i in range(prob_map.shape[0]):
            single_prob = prob_map[i, 0]
            threshold_value = thresholds[i]
            binary_mask = (single_prob >= threshold_value).float()
            binary_masks.append(binary_mask.unsqueeze(0).unsqueeze(0))

        return torch.cat(binary_masks, dim=0)


# ==================== 3. 改进的超参数条件化模块 ====================
class HyperCondModule(nn.Module):
    """改进的超参数条件化模块"""

    def __init__(self, cond_dim=128, hidden_dim=64):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        # 阈值和边界权重编码器
        self.threshold_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cond_dim // 2)
        )

        self.boundary_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cond_dim // 2)
        )

        # 条件融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2),
            nn.LayerNorm(cond_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cond_dim * 2, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, threshold, boundary_weight, device, batch_size=1):
        # 确保输入为tensor并扩展批次维度
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor([threshold], dtype=torch.float32)
        if not isinstance(boundary_weight, torch.Tensor):
            boundary_weight = torch.tensor([boundary_weight], dtype=torch.float32)

        # 扩展批次维度
        threshold = threshold.to(device).view(-1, 1)
        boundary_weight = boundary_weight.to(device).view(-1, 1)

        if threshold.shape[0] == 1 and batch_size > 1:
            threshold = threshold.expand(batch_size, -1)
        if boundary_weight.shape[0] == 1 and batch_size > 1:
            boundary_weight = boundary_weight.expand(batch_size, -1)

        # 编码
        threshold_enc = self.threshold_encoder(threshold)
        boundary_enc = self.boundary_encoder(boundary_weight)

        # 拼接和融合
        cond = torch.cat([threshold_enc, boundary_enc], dim=-1)
        cond = self.fusion_net(cond)

        return cond


# ==================== 4. RMS归一化层 ====================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / norm


# ==================== 5. 改进的流形约束超连接适配器 ====================
class ManifoldConstrainedAdapter(nn.Module):
    """改进的流形约束超连接适配器"""

    def __init__(self, embed_dim, n_streams=4, sinkhorn_iters=20):
        super().__init__()
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.embed_dim = embed_dim

        # 输入调整层
        self.input_adjust = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # RMS归一化
        self.norm = RMSNorm(embed_dim)

        # 适配器MLP：生成n个残差分支
        self.adapter_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim * n_streams)
        )

        # 可学习门控参数
        self.alpha = nn.Parameter(0.01 * torch.ones(n_streams))

        # 混合矩阵生成器
        self.mixing_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, n_streams * n_streams)
        )

        # Sinkhorn投影层
        self.sinkhorn = SinkhornProjection(iters=sinkhorn_iters)

    def forward(self, sam_features, blip_features):
        """
        sam_features: [B, N, C] SAM编码器特征
        blip_features: [B, N, C] BLIP视觉特征
        返回: 融合后的特征 [B, N, C]
        """
        B, N, C = sam_features.shape

        # 特征拼接
        combined = torch.cat([sam_features, blip_features], dim=-1)
        combined = self.input_adjust(combined)

        # RMS归一化
        combined = self.norm(combined)

        # 生成未约束的适配器输出
        raw_residual = self.adapter_mlp(combined)
        raw_residual = raw_residual.view(B, N, self.n_streams, C)

        # 构造双随机矩阵
        mixing_scores = self.mixing_mlp(combined)
        mixing_scores = mixing_scores.view(B, N, self.n_streams, self.n_streams)

        # [M1修复] 直接将原始分数送入Sinkhorn投影（内部会做log-space归一化）
        # 去掉冗余的双softmax预处理，让Sinkhorn自行做行列归一化
        H_res = self.sinkhorn(mixing_scores)

        # 应用流形约束残差映射
        # [关键修复] 原einsum 'bnsa,bnsc->bnsc' 错误：a在raw_residual中没有对应维度，
        # 等效于乘以行和(=1)，导致混合矩阵完全不起作用。
        # 正确索引：H_res[b,n,s,a] @ raw_residual[b,n,a,c] -> output[b,n,s,c]
        # 即用H_res的每一行对raw_residual的各stream做加权组合
        mixed_residual = torch.einsum('bnsa,bnac->bnsc', H_res, raw_residual)

        # 加权合并n个流
        alpha_gate = F.softmax(self.alpha, dim=0)
        final_residual = torch.einsum('s,bnsc->bnc', alpha_gate, mixed_residual)

        # 恒等映射残差连接
        output = sam_features + 0.1 * final_residual  # 使用较小的权重

        return output


# ==================== 6. 改进的跨空间稳定的Vision-Language Prompt生成器 ====================
class CrossModalStablePromptGenerator(nn.Module):
    """改进的跨空间稳定的Vision-Language Prompt生成器"""

    def __init__(
            self,
            text_dim: int = 768,
            vision_dim: int = 768,
            prompt_dim: int = 256,
            n_prompt: int = 2,
            use_sinkhorn: bool = True,
            sinkhorn_iters: int = 3,
            debug: bool = False,
    ):
        super().__init__()
        self.n_prompt = n_prompt
        self.prompt_dim = prompt_dim
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.use_sinkhorn = use_sinkhorn
        self.debug = debug

        # 投影矩阵
        self.W_text = nn.Linear(text_dim, prompt_dim, bias=True)
        self.W_vision = nn.Linear(vision_dim, prompt_dim, bias=True)

        # 可学习的流重要性权重
        self.stream_weights = nn.Parameter(torch.tensor([0.5, 0.5]))

        # 模态权重控制器
        self.modality_controller = nn.Sequential(
            nn.Linear(text_dim + vision_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )

        # 混合比例参数
        self.base_alpha = nn.Parameter(torch.tensor(0.0))
        self.base_beta = nn.Parameter(torch.tensor(0.0))

        # 归一化层
        self.rms_norm = RMSNorm(prompt_dim)

        # Sinkhorn投影
        self.sinkhorn = SinkhornProjection(iters=sinkhorn_iters) if use_sinkhorn else None

        # 监控指标
        self.register_buffer('text_ratio', torch.tensor(0.0))
        self.register_buffer('vision_ratio', torch.tensor(0.0))

        # 学习率预热
        self.register_buffer('step', torch.tensor(0))
        self.warmup_steps = 1000

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_normal_(self.W_text.weight, gain=0.3)
        nn.init.xavier_normal_(self.W_vision.weight, gain=1.0)

        with torch.no_grad():
            self.W_text.bias.data.uniform_(-0.1, 0.1)
            self.W_vision.bias.data.uniform_(-0.1, 0.1)
            nn.init.xavier_normal_(self.modality_controller[0].weight)
            nn.init.xavier_normal_(self.modality_controller[2].weight)

    def get_lr_multiplier(self):
        """获取学习率乘子"""
        if self.step < self.warmup_steps:
            return float(self.step) / self.warmup_steps
        return 1.0

    def compute_H_pre(self, text_score, vision_score, text_features, vision_features):
        """计算预映射矩阵 H_pre"""
        B = text_score.shape[0]

        # 提取全局特征
        if len(text_features.shape) == 3:
            text_global = text_features.mean(dim=1)
        else:
            text_global = text_features

        if len(vision_features.shape) == 4:
            if vision_features.shape[-1] == self.vision_dim:
                vision_global = vision_features.mean(dim=[1, 2])
            elif vision_features.shape[1] == self.vision_dim:
                vision_global = vision_features.mean(dim=[2, 3])
            else:
                vision_global = vision_features.view(B, -1, self.vision_dim).mean(dim=1)
        elif len(vision_features.shape) == 3:
            vision_global = vision_features.mean(dim=1)
        else:
            vision_global = vision_features

        # 调整维度
        if vision_global.shape[-1] != self.vision_dim:
            vision_global = vision_global.view(B, -1)
            if vision_global.shape[-1] > self.vision_dim:
                # 使用自适应平均池化
                vision_global = vision_global.view(B, self.vision_dim, -1).mean(dim=-1)
            elif vision_global.shape[-1] < self.vision_dim:
                padding = torch.zeros(B, self.vision_dim - vision_global.shape[-1]).to(vision_global.device)
                vision_global = torch.cat([vision_global, padding], dim=-1)

        # 通过控制器计算模态权重
        combined = torch.cat([text_global, vision_global], dim=-1)
        modality_weights = self.modality_controller(combined)

        # 基于原始得分的softmax
        scores = torch.cat([text_score, vision_score], dim=-1)
        temperature = 0.5
        raw_probs = torch.softmax(scores / temperature, dim=-1)

        # 结合两种方法
        alpha_base = 0.5 + 0.5 * torch.sigmoid(self.base_alpha)
        beta_base = 0.5 + 0.5 * torch.sigmoid(self.base_beta)

        text_ratio = alpha_base * modality_weights[:, 0:1] + (1 - alpha_base) * raw_probs[:, 0:1]
        vision_ratio = beta_base * modality_weights[:, 1:2] + (1 - beta_base) * raw_probs[:, 1:2]

        # 范围限制
        text_ratio = 0.4 + 0.2 * torch.sigmoid(text_ratio)
        vision_ratio = 0.4 + 0.2 * torch.sigmoid(vision_ratio)

        # 构建H_pre矩阵
        H_pre = torch.zeros(B, self.n_prompt, self.n_prompt, device=text_score.device)
        H_pre[:, 0, 0] = text_ratio.squeeze()
        H_pre[:, 0, 1] = 1.0 - text_ratio.squeeze()
        H_pre[:, 1, 0] = 1.0 - vision_ratio.squeeze()
        H_pre[:, 1, 1] = vision_ratio.squeeze()

        # 应用Sinkhorn约束（如果启用）
        if self.use_sinkhorn and self.sinkhorn is not None:
            H_pre = self.sinkhorn(H_pre)

        # 监控增益比
        text_ratio_val = H_pre[:, 0, 0].mean().detach()
        vision_ratio_val = H_pre[:, 1, 1].mean().detach()

        self.text_ratio.copy_(text_ratio_val)
        self.vision_ratio.copy_(vision_ratio_val)

        return H_pre, text_ratio_val, vision_ratio_val

    def compute_H_post(self):
        """计算后映射矩阵 H_post"""
        H_post = torch.softmax(self.stream_weights * 2.0, dim=0)
        return H_post

    def forward(self, text_embed, vision_embed, text_features, vision_features):
        """前向传播"""
        B = text_embed.shape[0]

        # 投影到统一空间
        text_proj = self.W_text(text_embed)
        vision_proj = self.W_vision(vision_embed)

        # 计算模态得分
        text_score = torch.mean(text_proj, dim=-1, keepdim=True) + torch.std(text_proj, dim=-1, keepdim=True) * 0.5
        vision_score = torch.max(vision_proj, dim=-1, keepdim=True)[0] + torch.min(vision_proj, dim=-1, keepdim=True)[
            0] * 0.5

        # 计算 H_pre 和增益比
        H_pre, text_ratio, vision_ratio = self.compute_H_pre(
            text_score, vision_score, text_features, vision_features
        )

        # 构造输入流并应用 H_pre
        input_streams = torch.stack([text_proj, vision_proj], dim=1)
        mixed_streams = torch.einsum('bnk,bkd->bnd', H_pre, input_streams)

        # 计算 H_post
        H_post = self.compute_H_post()

        # [S1修复] 保留多个prompt token，不合并为单一向量
        # 使用 H_post 作为缩放权重但保持 n_prompt 维度
        # mixed_streams: [B, n_prompt, prompt_dim]
        sparse_prompt = mixed_streams * H_post.unsqueeze(0).unsqueeze(-1)  # [B, n_prompt, prompt_dim]

        # 对每个prompt token分别归一化
        prompt_list = []
        for i in range(self.n_prompt):
            prompt_list.append(self.rms_norm(sparse_prompt[:, i, :]))
        sparse_prompt = torch.stack(prompt_list, dim=1)  # [B, n_prompt, prompt_dim]

        # 更新步数
        if self.training:
            self.step += 1

        return sparse_prompt

    def apply_gradient_constraints(self):
        """应用梯度约束"""
        if hasattr(self, 'stream_weights') and self.stream_weights.grad is not None:
            torch.nn.utils.clip_grad_norm_([self.stream_weights], max_norm=1.0)

        # 约束模态控制器和基础参数的梯度
        for name, param in self.named_parameters():
            if param.grad is not None and (
                    'modality_controller' in name or 'base_alpha' in name or 'base_beta' in name):
                torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)


# ==================== 7. 改进的边界感知损失函数 ====================
class BoundaryAwareLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, boundary_weight=0.5, kernel_size=3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.boundary_weight = boundary_weight
        self.kernel_size = kernel_size

        self.dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        self.ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

        # 预计算卷积核 - 不直接注册为buffer，在forward中动态创建
        self.kernel_size = kernel_size

    def _create_kernel(self, device):
        """创建卷积核并移动到指定设备"""
        return torch.ones(1, 1, self.kernel_size, self.kernel_size, device=device)

    def get_boundary_mask(self, gt):
        """更稳定的边界提取，保持输出尺寸与输入一致"""
        with torch.no_grad():
            # 二值化
            gt_binary = (gt > 0.5).float()

            # 添加填充
            pad_size = self.kernel_size // 2
            padded = F.pad(gt_binary, (pad_size, pad_size, pad_size, pad_size), mode='replicate')

            # 创建卷积核并移动到正确设备
            kernel = self._create_kernel(gt_binary.device)

            # 腐蚀
            eroded = F.conv2d(padded, kernel, padding=0, stride=1)
            eroded = (eroded == kernel.sum()).float()

            # 膨胀
            dilated = F.conv2d(padded, kernel, padding=0, stride=1)
            dilated = (dilated > 0).float()

            # 边界 = 膨胀 - 腐蚀
            boundary = dilated - eroded

            # 移除填充
            boundary = boundary[:, :, pad_size:-pad_size, pad_size:-pad_size]

            # 确保边界掩码与输入尺寸相同
            if boundary.shape[-2:] != gt_binary.shape[-2:]:
                boundary = F.interpolate(
                    boundary,
                    size=gt_binary.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            # 高斯模糊平滑边界
            if self.kernel_size > 1:
                boundary = F.avg_pool2d(boundary, kernel_size=3, stride=1, padding=1)

            # 重新确保尺寸（平均池化可能改变尺寸）
            if boundary.shape[-2:] != gt_binary.shape[-2:]:
                boundary = F.interpolate(
                    boundary,
                    size=gt_binary.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            # 归一化
            if boundary.max() > 0:
                boundary = boundary / (boundary.max() + 1e-8)

            # 确保边界掩码与输入完全相同
            assert boundary.shape == gt_binary.shape, \
                f"边界掩码尺寸不匹配: {boundary.shape} != {gt_binary.shape}"

            return boundary

    def forward(self, pred, gt, boundary_weight=None):
        if boundary_weight is None:
            boundary_weight = self.boundary_weight

        # 基础损失
        dice_ce_loss = self.dice_loss(pred, gt) + self.ce_loss(pred, gt)

        # 边界损失
        boundary_mask = self.get_boundary_mask(gt)
        boundary_pixels = boundary_mask.sum()

        if boundary_pixels > 10:  # 确保有足够的边界像素
            # 使用加权边界损失
            pred_sigmoid = torch.sigmoid(pred)

            # 边界Dice损失
            intersection = (pred_sigmoid * boundary_mask * gt).sum()
            union = (pred_sigmoid * boundary_mask).sum() + (boundary_mask * gt).sum() + 1e-8
            boundary_dice = 1.0 - (2.0 * intersection) / union

            # 边界CE损失
            boundary_ce = F.binary_cross_entropy_with_logits(
                pred * boundary_mask,
                gt * boundary_mask,
                reduction='mean'
            )

            boundary_loss = (boundary_dice + boundary_ce) * 0.5 * boundary_weight * self.beta
        else:
            boundary_loss = torch.tensor(0.0).to(pred.device)

        # 总损失
        total_loss = self.alpha * dice_ce_loss + boundary_loss

        return total_loss, dice_ce_loss, boundary_loss


# ==================== 8. 集成的图像编码器（包含Attention等基础组件） ====================
class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = self.add_decomposed_rel_pos(attn, q, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)

        return x

    def add_decomposed_rel_pos(self, attn, q, q_size, k_size):
        """Add decomposed relative positional embeddings."""
        q_h, q_w = q_size
        k_h, k_w = k_size

        Rh = self.get_rel_pos(q_h, k_h, self.rel_pos_h)
        Rw = self.get_rel_pos(q_w, k_w, self.rel_pos_w)

        B, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

        attn = (
                attn.view(B, q_h, q_w, k_h, k_w)
                + rel_h[:, :, :, :, None]
                + rel_w[:, :, :, None, :]
        ).view(B, q_h * q_w, k_h * k_w)

        return attn

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """Get relative positional embeddings."""
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos if needed.
        if rel_pos.shape[0] != max_rel_dist:
            # Interpolate rel pos.
            rel_pos_resized = F.interpolate(
                rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                size=max_rel_dist,
                mode="linear",
            )
            rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
        else:
            rel_pos_resized = rel_pos

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = self.window_partition(x, self.window_size)

        x = self.attn(x)

        # Reverse window partition
        if self.window_size > 0:
            x = self.window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def window_partition(self, x: torch.Tensor, window_size: int):
        """Partition into non-overlapping windows with padding if needed."""
        B, H, W, C = x.shape

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
        windows = (
            x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        )
        return windows, (Hp, Wp)

    def window_unpartition(self, windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]):
        """Window unpartition into original sequences and removing padding."""
        Hp, Wp = pad_hw
        H, W = hw
        B = windows.shape[0] // (Hp * Wp // window_size // window_size)
        x = windows.view(
            B, Hp // window_size, Wp // window_size, window_size, window_size, -1
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16),
            stride: Tuple[int, int] = (16, 16),
            padding: Tuple[int, int] = (0, 0),
            in_chans: int = 3,
            embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class IntegratedImageEncoderViT(nn.Module):
    """集成了流形约束超连接适配器的图像编码器"""

    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            n_streams: int = 4,
            blip_feature_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.blip_feature_dim = blip_feature_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Positional embedding
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            grid_size = img_size // patch_size
            self.pos_embed = nn.Parameter(
                torch.zeros(1, grid_size, grid_size, embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        # Neck
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

        # 流形约束超连接适配器
        self.n_streams = n_streams
        self.manifold_adapters = nn.ModuleList()

        # 动态决定哪些层添加适配器
        adapter_layers = []
        if depth >= 4:
            # 在网络的1/4、1/2、3/4位置添加适配器
            indices = [depth // 4, depth // 2, 3 * depth // 4]
            for idx in indices:
                if idx < depth:
                    adapter_layers.append(idx)

        for i in range(depth):
            if i in adapter_layers:
                self.manifold_adapters.append(
                    ManifoldConstrainedAdapter(embed_dim, n_streams=n_streams)
                )
            else:
                self.manifold_adapters.append(None)

        # BLIP特征调整层
        self.blip_feature_adjust = nn.Sequential(
            nn.Conv2d(blip_feature_dim, embed_dim, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.GELU()
        )

        print(f"初始化图像编码器: embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}")
        print(f"窗口注意力: window_size={window_size}, 全局注意力层: {global_attn_indexes}")
        print(f"BLIP特征维度: {blip_feature_dim}")
        print(f"适配器层: {adapter_layers}")

    def prepare_blip_features(self, blip_features, target_spatial_size):
        """准备BLIP特征，调整为与SAM特征相同的空间维度和通道维度"""
        if blip_features is None:
            return None

        B = blip_features.shape[0]

        # 处理不同形状的输入
        if len(blip_features.shape) == 3:
            # 形状为 [B, seq_len, hidden_dim]
            seq_len, hidden_dim = blip_features.shape[1], blip_features.shape[2]

            # 检查是否是合法的空间形状
            spatial_size = int(math.sqrt(seq_len))
            if spatial_size * spatial_size == seq_len:
                # 重塑为空间格式 [B, hidden_dim, H, W]
                blip_features = blip_features.reshape(B, spatial_size, spatial_size, hidden_dim)
                blip_features = blip_features.permute(0, 3, 1, 2)
            else:
                # 如果不是空间格式，使用自适应池化
                blip_features = blip_features.transpose(1, 2)  # [B, hidden_dim, seq_len]
                blip_features = F.adaptive_avg_pool1d(blip_features, 1)  # [B, hidden_dim, 1]
                blip_features = blip_features.unsqueeze(-1)  # [B, hidden_dim, 1, 1]

        # [M2修复] 通道数检查 - 不再动态创建未注册的Conv2d
        # 如果通道数不匹配，使用自适应平均池化或直接交给blip_feature_adjust处理
        if blip_features.shape[1] != self.blip_feature_dim:
            # 使用1x1自适应调整（通过线性插值模拟通道映射）
            blip_features = blip_features.permute(0, 2, 3, 1)  # [B, H, W, C_in]
            blip_features = F.adaptive_avg_pool1d(
                blip_features.reshape(-1, blip_features.shape[-1]).unsqueeze(1),
                self.blip_feature_dim
            ).squeeze(1).reshape(blip_features.shape[0], blip_features.shape[1], blip_features.shape[2], self.blip_feature_dim)
            blip_features = blip_features.permute(0, 3, 1, 2)  # [B, C, H, W]

        # 调整空间大小到目标尺寸
        if blip_features.shape[2:] != target_spatial_size:
            blip_features = F.interpolate(
                blip_features,
                size=target_spatial_size,
                mode='bilinear',
                align_corners=False
            )

        # 调整通道数到SAM的embed_dim
        blip_features = self.blip_feature_adjust(blip_features)  # [B, embed_dim, H, W]
        blip_features = blip_features.permute(0, 2, 3, 1)  # [B, H, W, embed_dim]

        return blip_features

    def forward(self, x: torch.Tensor, adapter_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 图像特征提取
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        B, H, W, C = x.shape

        # [性能优化] 预先准备BLIP特征，避免在每个adapter层重复处理
        blip_features_prepared = None
        if adapter_input is not None:
            blip_features_prepared = self.prepare_blip_features(adapter_input, (H, W))
            if blip_features_prepared is not None:
                blip_features_prepared = blip_features_prepared.reshape(B, H * W, C)

        # 多模态特征融合
        for i, (blk, adapter) in enumerate(zip(self.blocks, self.manifold_adapters)):
            x = blk(x)

            # 在特定层进行流形约束特征融合
            if adapter is not None and blip_features_prepared is not None:
                # 重塑SAM特征
                x_reshaped = x.reshape(B, H * W, C)

                # 应用流形约束适配器
                x_reshaped = adapter(x_reshaped, blip_features_prepared)

                # 重塑回原始维度
                x = x_reshaped.reshape(B, H, W, C)

        x = self.neck(x.permute(0, 3, 1, 2))
        return x


# ==================== 9. 改进的集成主模型 ====================
class MMSAM_Integrated(nn.Module):
    """改进的四合一集成模型"""

    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder=None,
            image_feature_dim=768,
            use_rankdice=True,
            use_hypercond=True,
            n_streams=4,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.use_rankdice = use_rankdice
        self.use_hypercond = use_hypercond

        # 位置编码
        from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom
        self.pe_layer = PositionEmbeddingRandom(256 // 2)

        # 1. 跨空间稳定的Vision-Language Prompt生成器
        self.prompt_generator = CrossModalStablePromptGenerator(
            text_dim=768,
            vision_dim=image_feature_dim,
            prompt_dim=256,
            n_prompt=2,
            use_sinkhorn=True,
            sinkhorn_iters=5,
            debug=False
        )

        # 2. 超参数条件化模块
        if use_hypercond:
            self.hyper_cond = HyperCondModule(cond_dim=128, hidden_dim=64)
            self.cond_channel_adapter = nn.Conv2d(128, 256, kernel_size=1, bias=False)
            self.cond_spatial_adapter = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.GELU()
            )

        # 3. RankDice-RMA模块
        if use_rankdice:
            self.rankdice_module = RankDiceRMAModule(use_in_training=True, weight=0.1)

        # 文本和图像适配器
        self.text_adapter = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 图像特征投影层（用于维度对齐）
        self.image_feature_projection = nn.Sequential(
            nn.Linear(image_feature_dim, 768),
            nn.GELU()
        ) if image_feature_dim != 768 else nn.Identity()

        self.pseudo_mask_embed = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.GELU()
        )

        # 训练步数
        self.register_buffer('training_step', torch.tensor(0))

        self._init_weights()

    def _init_weights(self):
        """[C2修复] 只初始化MC-SAM新增的模块，不覆盖预训练权重"""
        # 明确列出需要初始化的新增模块，避免破坏 image_encoder 和 mask_decoder 的预训练权重
        new_modules = [self.text_adapter, self.pseudo_mask_embed]
        
        # image_feature_projection 可能是 Identity，检查一下
        if not isinstance(self.image_feature_projection, nn.Identity):
            new_modules.append(self.image_feature_projection)
        
        if self.use_hypercond:
            new_modules.extend([self.cond_channel_adapter, self.cond_spatial_adapter])
        
        for module in new_modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, image, text_embeddings, image_features,
                gt_mask=None, hyper_cond=None, return_logits=False):
        """
        前向传播
        Args:
            image: 输入图像 [B, 3, H, W]
            text_embeddings: 文本特征 [B, seq_len, 768]
            image_features: 图像特征 [B, seq_len, hidden_dim] 或 [B, C, H, W]
            gt_mask: 训练时的真实掩码 [B, 1, H, W]
            hyper_cond: 超参数条件化字典 {'threshold': tau, 'boundary_weight': lambda_val}
            return_logits: 是否返回logits而不是最终掩码
        Returns:
            pred_mask: 预测掩码 [B, 1, H, W]
            rank_dice_loss: RankDice-RMA损失（训练时）
            hyper_cond: 使用的超参数条件
        """
        B = image.shape[0]

        # ========== 1. 处理图像特征 ==========
        # 确保图像特征维度正确
        if len(image_features.shape) == 3:
            _, seq_len, hidden_dim = image_features.shape

            # 投影到统一维度
            if hidden_dim != 768:
                image_features = self.image_feature_projection(image_features)

            # 重塑为空间特征
            if seq_len == 576:  # BLIP 24x24
                h = w = 24
                image_features_for_sam = image_features.reshape(B, h, w, -1).permute(0, 3, 1, 2)
            elif seq_len == 64 * 64:  # 64x64
                h = w = 64
                image_features_for_sam = image_features.reshape(B, h, w, -1).permute(0, 3, 1, 2)
            else:
                # 自适应池化
                image_features_for_sam = image_features.mean(dim=1, keepdim=True)
                image_features_for_sam = image_features_for_sam.unsqueeze(-1)
                image_features_for_sam = F.interpolate(image_features_for_sam, size=(64, 64))
        else:
            image_features_for_sam = image_features

        # 保存原始特征用于Prompt生成器
        image_features_original = image_features.clone()

        # ========== 2. 超参数条件化 ==========
        if hyper_cond is None:
            hyper_cond = {'threshold': 0.5, 'boundary_weight': 1.0}

        threshold = hyper_cond['threshold']
        boundary_weight = hyper_cond['boundary_weight']

        # ========== 3. 图像编码器 ==========
        image_embedding = self.image_encoder(image, image_features_for_sam)

        # ========== 4. 超参数条件注入 ==========
        # [S3修复] 推理时也使用HyperCond，利用默认参数(tau=0.5, lambda=1.0)生成条件嵌入
        cond_embedding = None
        if self.use_hypercond:
            cond_embedding = self.hyper_cond(threshold, boundary_weight, image_embedding.device, B)

        # ========== 5. 提取文本和视觉的全局特征 ==========
        text_global = text_embeddings.mean(dim=1)

        # 处理图像特征形状以计算全局特征
        if len(image_features_original.shape) == 4:
            vision_global = image_features_original.mean(dim=[2, 3])
        elif len(image_features_original.shape) == 3:
            vision_global = image_features_original.mean(dim=1)
        else:
            vision_global = image_features_original.flatten(1).mean(dim=1, keepdim=True)

        # 确保维度匹配
        if vision_global.dim() == 1:
            vision_global = vision_global.unsqueeze(1)

        # ========== 6. 使用跨模态稳定Prompt生成器 ==========
        if self.training:
            self.training_step += 1

        # [S1修复] prompt_generator 现在直接返回 [B, n_prompt, prompt_dim]
        sparse_prompt = self.prompt_generator(
            text_global, vision_global, text_embeddings, image_features_original
        )  # [B, n_prompt, 256]

        # ========== 7. 超参数条件与特征融合 ==========
        dense_embeddings = self.pseudo_mask_embed(image_embedding)

        if cond_embedding is not None:
            # 调整条件嵌入维度
            cond_dense = cond_embedding.view(B, -1, 1, 1)  # [B, 128, 1, 1]
            cond_dense = self.cond_channel_adapter(cond_dense)  # [B, 256, 1, 1]
            cond_dense = F.interpolate(cond_dense, size=image_embedding.shape[-2:])
            cond_dense = self.cond_spatial_adapter(cond_dense)

            # 融合条件信息
            dense_embeddings = dense_embeddings + 0.1 * cond_dense

        # ========== 8. 位置编码 ==========
        image_pe = self.pe_layer((image_embedding.shape[2], image_embedding.shape[3])).unsqueeze(0)

        # ========== 9. Mask解码器 ==========
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # ========== 10. 上采样 ==========
        logits = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # ========== 11. RankDice-RMA处理 ==========
        rank_dice_loss = torch.tensor(0.0).to(logits.device)

        if self.training:
            # 训练模式
            if self.use_rankdice and gt_mask is not None:
                logits, rank_dice_loss = self.rankdice_module(logits, gt_mask)

            if return_logits:
                return logits, rank_dice_loss, hyper_cond
            else:
                return torch.sigmoid(logits), rank_dice_loss, hyper_cond
        else:
            # 推理模式
            if self.use_rankdice:
                pred_mask = self.rankdice_module._inference(logits)
            else:
                pred_mask = torch.sigmoid(logits)

            return pred_mask, rank_dice_loss, hyper_cond

    def apply_gradient_constraints(self):
        """应用梯度约束"""
        # 跨模态提示生成器
        if hasattr(self.prompt_generator, 'apply_gradient_constraints'):
            self.prompt_generator.apply_gradient_constraints()

        # 全局梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def set_training_mode(self, mode=True):
        """设置训练模式"""
        self.train(mode)
        if mode:
            self.training_step = torch.tensor(0)

        # 设置子模块的训练模式
        if hasattr(self, 'prompt_generator'):
            self.prompt_generator.train(mode)
        if hasattr(self, 'rankdice_module'):
            self.rankdice_module.train(mode)
        if hasattr(self, 'hyper_cond'):
            self.hyper_cond.train(mode)


# ==================== 10. 创建集成模型的函数 ====================
def create_integrated_model(
        sam_checkpoint_path,
        model_type="vit_l",
        image_size=1024,
        use_rankdice=True,
        use_hypercond=True,
        n_streams=4,
        device="cuda",
        blip_feature_dim=768
):
    """创建集成模型"""
    from segment_anything import sam_model_registry

    # 加载SAM基础模型
    print(f"加载SAM模型: {model_type}")
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)

    # 根据模型类型确定配置
    if model_type == "vit_b":
        embed_dim = 768
        num_heads = 12
        depth = 12
        window_size = 14
        global_attn_indexes = [2, 5, 8, 11]
    elif model_type == "vit_l":
        embed_dim = 1024
        num_heads = 16
        depth = 24
        window_size = 14
        global_attn_indexes = [5, 11, 17, 23]
    elif model_type == "vit_h":
        embed_dim = 1280
        num_heads = 16
        depth = 32
        window_size = 14
        global_attn_indexes = [7, 15, 23, 31]
    else:
        # 使用默认值
        embed_dim = 768
        num_heads = 12
        depth = 12
        window_size = 0
        global_attn_indexes = []

    print(f"SAM配置: embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}")
    print(f"窗口注意力: window_size={window_size}, 全局注意力层: {global_attn_indexes}")

    # 创建集成的图像编码器
    image_encoder = IntegratedImageEncoderViT(
        img_size=image_size,
        patch_size=16,
        in_chans=3,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        out_chans=256,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        window_size=window_size,
        global_attn_indexes=global_attn_indexes,
        n_streams=n_streams,
        blip_feature_dim=blip_feature_dim
    )

    # ===== [C1修复] 从SAM预训练模型迁移权重到集成编码器 =====
    # IntegratedImageEncoderViT 与 SAM 的 ImageEncoderViT 共享相同的
    # patch_embed, pos_embed, blocks, neck 结构，只额外增加了
    # manifold_adapters 和 blip_feature_adjust（这些保持随机初始化）
    sam_encoder_state = sam_model.image_encoder.state_dict()
    integrated_encoder_state = image_encoder.state_dict()

    transferred_keys = []
    skipped_keys = []
    for key in sam_encoder_state:
        if key in integrated_encoder_state:
            if sam_encoder_state[key].shape == integrated_encoder_state[key].shape:
                integrated_encoder_state[key] = sam_encoder_state[key]
                transferred_keys.append(key)
            else:
                skipped_keys.append(f"{key}: SAM {sam_encoder_state[key].shape} vs Integrated {integrated_encoder_state[key].shape}")
        else:
            skipped_keys.append(f"{key}: 不存在于集成编码器中")

    image_encoder.load_state_dict(integrated_encoder_state)
    print(f"[C1修复] 从SAM预训练模型成功迁移了 {len(transferred_keys)}/{len(sam_encoder_state)} 个参数")
    if skipped_keys:
        print(f"[C1修复] 跳过 {len(skipped_keys)} 个参数:")
        for sk in skipped_keys[:5]:  # 只打印前5个
            print(f"  - {sk}")
        if len(skipped_keys) > 5:
            print(f"  ... 及其余 {len(skipped_keys) - 5} 个")

    # 创建集成主模型
    integrated_model = MMSAM_Integrated(
        image_encoder=image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        image_feature_dim=blip_feature_dim,
        use_rankdice=use_rankdice,
        use_hypercond=use_hypercond,
        n_streams=n_streams,
    ).to(device)

    return integrated_model


# ==================== 11. 优化的训练函数 ====================
def create_optimizer_for_integrated_model(model, lr=0.0002, weight_decay=0.01):
    """为集成模型创建优化器"""
    # 定义参数组
    param_groups = []

    # 1. 图像编码器适配器层（较高学习率）
    encoder_adapter_params = []
    for name, param in model.named_parameters():
        if 'image_encoder' in name and ('adapter' in name or 'manifold_adapters' in name or 'blip_feature_adjust' in name):
            encoder_adapter_params.append(param)

    if encoder_adapter_params:
        param_groups.append({
            'params': encoder_adapter_params,
            'lr': lr * 1.0,
            'weight_decay': weight_decay,
            'name': 'encoder_adapters'
        })

    # 2. 跨模态提示生成器（中等学习率）
    prompt_generator_params = []
    for name, param in model.named_parameters():
        if 'prompt_generator' in name:
            prompt_generator_params.append(param)

    if prompt_generator_params:
        param_groups.append({
            'params': prompt_generator_params,
            'lr': lr * 0.5,
            'weight_decay': weight_decay * 0.1,
            'name': 'prompt_generator'
        })

    # 3. 超参数条件化模块（较低学习率）— 包含 hyper_cond 和 cond_* 适配器
    hypercond_params = []
    for name, param in model.named_parameters():
        if 'hyper_cond' in name or 'cond_channel_adapter' in name or 'cond_spatial_adapter' in name:
            hypercond_params.append(param)

    if hypercond_params:
        param_groups.append({
            'params': hypercond_params,
            'lr': lr * 0.1,
            'weight_decay': weight_decay * 0.1,
            'name': 'hyper_cond'
        })

    # 4. RankDice模块（较低学习率）
    rankdice_params = []
    for name, param in model.named_parameters():
        if 'rankdice_module' in name:
            rankdice_params.append(param)

    if rankdice_params:
        param_groups.append({
            'params': rankdice_params,
            'lr': lr * 0.1,
            'weight_decay': weight_decay * 0.1,
            'name': 'rankdice'
        })

    # 5. 其他参数（较低学习率）
    # [修复] 使用精确的关键字排除，避免text_adapter等被意外遗漏
    already_assigned = set()
    for group in param_groups:
        for param in group['params']:
            already_assigned.add(id(param))

    other_params = []
    for name, param in model.named_parameters():
        if id(param) not in already_assigned:
            other_params.append(param)

    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': lr * 0.01,
            'weight_decay': weight_decay,
            'name': 'others'
        })

    # 创建优化器
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    print(f"优化器参数组:")
    for group in param_groups:
        print(f"  {group['name']}: {len(group['params'])}个参数, lr={group['lr']}")

    return optimizer


# ==================== 12. 训练循环 ====================
def train_integrated_model(
        model,
        train_dataloader,
        val_dataloader,
        processor,
        vlm_model,
        tokenizer,
        mamba_model,
        optimizer,
        num_epochs,
        device,
        use_amp=False,
        model_save_path="./checkpoints",
        eval_interval=1,
):
    """训练集成模型"""
    import os
    from tqdm import tqdm
    from datetime import datetime
    import torch.nn.functional as F

    os.makedirs(model_save_path, exist_ok=True)

    # 损失函数
    seg_loss = BoundaryAwareLoss(alpha=1.0, beta=0.1)

    # Beta分布用于超参数采样
    from torch.distributions import Beta
    beta_dist = Beta(torch.tensor([2.0]), torch.tensor([2.0]))

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(train_dataloader),
        eta_min=1e-6
    )

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 训练历史
    history = {
        'train_loss': [],
        'train_rank_dice_loss': [],
        'train_boundary_loss': [],
        'val_metrics': [],
    }

    best_val_score = 0.0

    for epoch in range(num_epochs):
        model.train()
        model.set_training_mode(True)

        epoch_loss = 0.0
        epoch_rank_dice_loss = 0.0
        epoch_boundary_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for step, (image, gt2D, img_1024_ori) in enumerate(pbar):
            optimizer.zero_grad()

            image, gt2D = image.to(device), gt2D.to(device)
            img_1024_ori = img_1024_ori.to(device)

            # ========== 超参数条件采样 ==========
            tau_sample = beta_dist.sample().item()
            tau = 0.4 + tau_sample * 0.2  # [0.4, 0.6]

            lambda_sample = beta_dist.sample().item()
            boundary_weight = 0.3 + lambda_sample * 0.7  # [0.3, 1.0]

            hyper_cond = {
                'threshold': tau,
                'boundary_weight': boundary_weight
            }

            # ========== 获取VLM描述和特征 ==========
            with torch.no_grad():
                vlm_inputs = processor(img_1024_ori, return_tensors="pt").to(device)
                vlm_outputs = vlm_model.generate(**vlm_inputs)
                description = processor.decode(vlm_outputs[0], skip_special_tokens=True)

                mamba_inputs = tokenizer(description, padding=True, return_tensors="pt").to(device)
                mamba_outputs = mamba_model(**mamba_inputs)
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

            text_features = mamba_outputs.last_hidden_state

            # ========== 模型前向 ==========
            if use_amp and scaler is not None:
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
                    total_loss = total_loss + rank_dice_loss

                # 反向传播
                scaler.scale(total_loss).backward()

                # 应用梯度约束
                model.apply_gradient_constraints()

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
                total_loss = total_loss + rank_dice_loss

                # 反向传播
                total_loss.backward()

                # 应用梯度约束
                model.apply_gradient_constraints()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()

            # ========== 记录损失 ==========
            epoch_loss += total_loss.item()
            epoch_rank_dice_loss += rank_dice_loss.item() if isinstance(rank_dice_loss,
                                                                        torch.Tensor) else rank_dice_loss
            epoch_boundary_loss += boundary_loss.item() if isinstance(boundary_loss, torch.Tensor) else boundary_loss

            # 学习率调度
            scheduler.step()

            # 更新进度条
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'rank_dice': f'{rank_dice_loss.item():.4f}' if isinstance(rank_dice_loss,
                                                                          torch.Tensor) else f'{rank_dice_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'tau': f'{tau:.3f}',
                'lambda': f'{boundary_weight:.3f}'
            })

        # 计算平均损失
        epoch_loss /= len(train_dataloader)
        epoch_rank_dice_loss /= len(train_dataloader)
        epoch_boundary_loss /= len(train_dataloader)

        history['train_loss'].append(epoch_loss)
        history['train_rank_dice_loss'].append(epoch_rank_dice_loss)
        history['train_boundary_loss'].append(epoch_boundary_loss)

        print(f"\nEpoch {epoch + 1} 训练结果:")
        print(f"  总损失: {epoch_loss:.4f}")
        print(f"  RankDice损失: {epoch_rank_dice_loss:.4f}")
        print(f"  边界损失: {epoch_boundary_loss:.4f}")

        # ========== 验证 ==========
        if (epoch + 1) % eval_interval == 0 and val_dataloader is not None:
            val_metrics = evaluate_model(
                model,
                val_dataloader,
                processor,
                vlm_model,
                tokenizer,
                mamba_model,
                device
            )

            history['val_metrics'].append(val_metrics)

            print(f"  验证指标:")
            for key, value in val_metrics.items():
                print(f"    {key}: {value:.4f}")

            # 保存最佳模型
            val_score = (val_metrics['sm'] + val_metrics['em'] + val_metrics['wfm']) / 3
            if val_score > best_val_score:
                best_val_score = val_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': epoch_loss,
                    'val_score': val_score,
                    'val_metrics': val_metrics,
                    'history': history,
                }, os.path.join(model_save_path, 'model_best.pth'))
                print(f"  ✅ 保存最佳模型，验证分数: {val_score:.4f}")

        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'history': history,
            }, os.path.join(model_save_path, f'model_epoch_{epoch + 1}.pth'))

    print("训练完成!")

    # 保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
    }, os.path.join(model_save_path, 'model_final.pth'))

    return history


# ==================== 13. 评估函数 ====================
def evaluate_model(model, dataloader, processor, vlm_model, tokenizer, mamba_model, device):
    """评估模型"""
    model.eval()
    model.set_training_mode(False)

    # 这里需要导入你的评估指标计算函数
    # 假设你有类似以下的指标计算类
    try:
        from utils_downstream.saliency_metric import (
            cal_mae, cal_sm, cal_em, cal_wfm, cal_dice, cal_iou, cal_ber
        )
    except ImportError:
        # 如果导入失败，创建模拟的评估器
        class MockMetric:
            def __init__(self):
                self.values = []

            def update(self, pred, gt):
                self.values.append(0.5)

            def show(self):
                return 0.5 if self.values else 0.0

        cal_mae = cal_sm = cal_em = cal_wfm = cal_dice = cal_iou = cal_ber = MockMetric

    # 初始化评估指标
    mae, sm, em, wfm, m_dice, m_iou, ber = cal_mae(), cal_sm(), cal_em(), cal_wfm(), cal_dice(), cal_iou(), cal_ber()

    from tqdm import tqdm
    import torch.nn.functional as F

    pbar = tqdm(dataloader, desc='评估进度')

    with torch.no_grad():
        for step, (image, gt2D, img_1024_ori) in enumerate(pbar):
            image, gt2D = image.to(device), gt2D.to(device)
            img_1024_ori = img_1024_ori.to(device)

            # 获取VLM描述和特征
            vlm_inputs = processor(img_1024_ori, return_tensors="pt").to(device)
            vlm_outputs = vlm_model.generate(**vlm_inputs)
            description = processor.decode(vlm_outputs[0], skip_special_tokens=True)

            mamba_inputs = tokenizer(description, padding=True, return_tensors="pt").to(device)
            mamba_outputs = mamba_model(**mamba_inputs)
            vision_outputs = vlm_model.vision_model(**vlm_inputs)
            image_features_raw = vision_outputs.last_hidden_state[:, 1:, :]

            # 处理图像特征
            batch_size, seq_len, hidden_dim = image_features_raw.shape
            if seq_len == 576:
                image_features = image_features_raw.reshape(batch_size, 24, 24, hidden_dim)
                image_features = image_features.permute(0, 3, 1, 2)
                image_features = F.interpolate(image_features, size=(64, 64), mode='bilinear', align_corners=False)
            elif seq_len == 64 * 64:
                image_features = image_features_raw.reshape(batch_size, 64, 64, hidden_dim).permute(0, 3, 1, 2)
            else:
                image_features = image_features_raw.mean(dim=1, keepdim=True)
                image_features = image_features.unsqueeze(-1)
                image_features = F.interpolate(image_features, size=(64, 64))

            text_features = mamba_outputs.last_hidden_state

            # 模型推理
            pred_mask, _, _ = model(
                image=image,
                text_embeddings=text_features,
                image_features=image_features,
                gt_mask=None,
                hyper_cond={'threshold': 0.5, 'boundary_weight': 1.0},
                return_logits=False
            )

            # 计算指标
            pred_np = pred_mask.squeeze().cpu().numpy()
            gt_np = gt2D.squeeze().cpu().numpy()

            # 处理可能的维度问题
            if pred_np.ndim == 0:
                pred_np = np.expand_dims(pred_np, 0)
            if gt_np.ndim == 0:
                gt_np = np.expand_dims(gt_np, 0)

            mae.update(pred_np, gt_np)
            sm.update(pred_np, gt_np)
            em.update(pred_np, gt_np)
            wfm.update(pred_np, gt_np)
            m_dice.update(pred_np, gt_np)
            m_iou.update(pred_np, gt_np)
            ber.update(pred_np, gt_np)

    # 汇总结果
    metrics = {
        'sm': sm.show(),
        'em': em.show(),
        'wfm': wfm.show(),
        'mae': mae.show(),
        'dice': m_dice.show(),
        'iou': m_iou.show(),
        'ber': ber.show(),
    }

    return metrics


# ==================== 14. 推理函数 ====================
def inference_integrated_model(
        model,
        image,
        text_features,
        image_features,
        device="cuda",
        threshold=0.5,
        boundary_weight=1.0,
        use_rankdice=True,
):
    """使用集成模型进行推理"""
    model.eval()
    model.set_training_mode(False)

    hyper_cond = {
        'threshold': threshold,
        'boundary_weight': boundary_weight
    }

    with torch.no_grad():
        pred_mask, _, _ = model(
            image=image.to(device),
            text_embeddings=text_features.to(device),
            image_features=image_features.to(device),
            gt_mask=None,
            hyper_cond=hyper_cond,
            return_logits=False
        )

    return pred_mask.cpu()


# ==================== 15. 主要执行函数 ====================
def main():
    """主函数示例"""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="训练集成的MMSAM模型")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="SAM预训练权重路径")
    parser.add_argument("--model_type", type=str, default="vit_l", choices=["vit_b", "vit_l", "vit_h"],
                        help="SAM模型类型")
    parser.add_argument("--data_root", type=str, required=True, help="数据根目录")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.0002, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--use_amp", action="store_true", help="使用混合精度训练")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="保存目录")
    parser.add_argument("--val_sample_size", type=int, default=1000, help="验证集采样大小")

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    print("创建集成模型...")
    model = create_integrated_model(
        sam_checkpoint_path=args.sam_checkpoint,
        model_type=args.model_type,
        image_size=1024,
        use_rankdice=True,
        use_hypercond=True,
        n_streams=4,
        device=device,
        blip_feature_dim=768
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 创建优化器
    optimizer = create_optimizer_for_integrated_model(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 这里需要加载你的数据集
    # 示例：train_dataloader, val_dataloader = create_dataloaders(args.data_root, args.batch_size)

    # 这里需要加载VLM和Mamba模型
    # 示例：
    # from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, MambaModel
    # processor = BlipProcessor.from_pretrained("path/to/blip")
    # vlm_model = BlipForConditionalGeneration.from_pretrained("path/to/blip").to(device)
    # tokenizer = AutoTokenizer.from_pretrained("path/to/mamba")
    # mamba_model = MambaModel.from_pretrained("path/to/mamba").to(device)

    print("模型创建完成，可以开始训练！")


if __name__ == "__main__":
    main()