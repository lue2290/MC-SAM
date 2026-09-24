# Current implementation notes

The current MC-SAM configuration uses SAM ViT-H. The original image encoder and prompt encoder are frozen, while the SAM mask decoder and adaptation modules are trainable. BLIP and Mamba are loaded separately and frozen.

One MCA with a 128-dimensional bottleneck and four streams is reused at image encoder positions 8, 16, and 24 (zero-based). Its stream mixer uses 20 Sinkhorn row/column normalization pairs. The BLIP visual alignment has rank 48. CSPG forms two 256-dimensional sparse prompt streams using rank-48 text and visual projections, Gram affinity, temperature 1.0 and five Sinkhorn pairs.

HyperCond receives the threshold condition only. The sampled boundary coefficient scales the training boundary loss and is not an input to HyperCond. The dense-prompt head has two 256-to-256 depthwise-separable stages. RankDice-RMA is parameter-free and runs at inference only.

The integrated ViT-H segmentation model has 5,151,388 trainable and 637,032,268 frozen parameters, excluding the separately loaded frozen BLIP and Mamba models. Run `python count_model_parameters.py` to check the registered parameter count. Training and inference commands, data layout, and the limits of this source release are in the repository README.
