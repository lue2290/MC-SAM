# MC-SAM

Code for *MC-SAM: A Stability-Constrained Coupled Adaptation Framework for SAM in Camouflaged Scene Segmentation*.

MC-SAM builds on MM-SAM and uses a frozen SAM ViT-H image encoder, frozen BLIP and Mamba cue extractors, and a trainable SAM mask decoder. A shared bottleneck MCA is called at three image-encoder positions. CSPG generates sparse prompts. HyperCond supplies threshold conditioning to the dense prompt. RankDice-RMA is used only during inference and does not enter the training loss.

## Source layout

- `MCsam_train.py`: training, fixed train/validation split, checkpoint selection.
- `inference_mcsam.py`: loading a trained checkpoint and saving prediction masks.
- `segment_anything/modeling/mcsam_integrated.py`: MC-SAM model and losses.
- `config_mcsam.py`: default configuration.
- `count_model_parameters.py`: parameter accounting without loading pretrained checkpoints.
- `test_final_settings.py` and `test_fixes.py`: local component checks.

## Inputs and environment

Install PyTorch, torchvision, transformers, MONAI, NumPy, SciPy, Pillow, matplotlib and tqdm in a compatible Python environment. Obtain the SAM ViT-H checkpoint and the BLIP and Mamba models used by MM-SAM separately. The cue models correspond to `Salesforce/blip-image-captioning-large` and `state-spaces/mamba-130m-hf`. Pretrained and trained weights, as well as COD images, are not included in this repository.

The training directory must contain paired files in `Imgs/` and `GT/`, for example:

```text
combined_train/
  Imgs/0001.jpg
  GT/0001.png
```

The loader sorts image and mask filenames separately and checks their counts. Confirm that paired stems match before training. The expected pool combines 3,040 COD10K and 1,000 CAMO training images. With the default fixed split, 3,636 images update the model and 404 images select the checkpoint. Official test sets are excluded from this split.

## Training

```bash
python MCsam_train.py \
  --train_data /path/to/combined_train \
  --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
  --blip_path /path/to/Blip \
  --mamba_path /path/to/mamba \
  --model_type vit_h \
  --num_epochs 20 --batch_size 1 --lr 0.00005 \
  --val_sample_size 404 --split_seed 42 \
  --mca_bottleneck_dim 128 --projection_rank 48 \
  --cspg_temperature 1.0 --cspg_iters 5 \
  --work_dir /path/to/output
```

The default optimizer is AdamW with weight decay 0.01 and cosine learning-rate scheduling. Images are resized to 1024×1024 and normalized with ImageNet channel statistics; masks use nearest-neighbor resizing. There is no stochastic image augmentation or patience-based early stopping. Every epoch is evaluated on the fixed validation subset. The selected checkpoint maximizes the mean of validation S-measure, adaptive E-measure and weighted F-measure. A separate `--seed` controls a training run; `--split_seed` holds the validation assignment fixed.

BLIP caption generation uses the pretrained model's default generation configuration. HyperCond samples the threshold condition in [0.4, 0.6]; the sampled boundary weight in [0.3, 1.0] scales only the training boundary loss. RankDice-RMA adds no training loss.

## Inference

```bash
python inference_mcsam.py \
  --model_path /path/to/model_best.pth \
  --data_path /path/to/test_set \
  --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
  --blip_path /path/to/Blip \
  --mamba_path /path/to/mamba \
  --model_type vit_h --save_dir /path/to/predictions
```

This script expects paired images and masks because it uses the same dataset reader as the validation code. It saves a prediction for each sample. The model checkpoint records the training arguments; loading is strict so an incompatible architecture raises an error.

By default, inference saves the probability maps used for the fixed-threshold comparison in manuscript Table 2. Add `--use_rankdice` to apply the inference-only RankDice-RMA selector, as in the final row of manuscript Table 3. Run the two modes with separate `--save_dir` values to avoid mixing their predictions. The `--threshold` argument supplies the HyperCond condition; it does not turn RankDice-RMA on or off.

## Parameter accounting

Run `python count_model_parameters.py` from the repository root. For ViT-H, 128-dimensional shared MCA and rank-48 alignment, the integrated segmentation model contains **642,183,656** parameters: **5,151,388 trainable** and **637,032,268 frozen**. The count includes the trainable SAM mask decoder and counts the shared MCA once. Separately loaded frozen BLIP and Mamba models are outside this total. The count script uses a meta device and does not load a checkpoint or run training.

## Validation scope

The component checks can be run with `python -m unittest test_final_settings test_fixes`. Full benchmark reproduction requires the external datasets, pretrained cue models and a trained checkpoint. The repository does not contain model weights or a complete run artifact.
