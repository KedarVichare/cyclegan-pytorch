# CycleGAN PyTorch

Unpaired image-to-image translation using CycleGAN for Monet style transfer.

## Setup

### Requirements

- Python 3.7+
- PyTorch with CUDA support (recommended for GPU training)
- CUDA 11.0+ (optional, for GPU acceleration)

### Installation

```bash
pip install torch torchvision
pip install scipy pillow tqdm matplotlib numpy
```

### Dataset Structure

```
dataset/
├── monet_jpg/       (300 Monet paintings)
└── photo_jpg/       (7,038 real photographs)
```

Download the Kaggle dataset and extract to the `dataset/` directory.

## Training

### Configuration

Edit the following parameters in the notebook:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | 256 | Image resolution |
| `BATCH_SIZE` | 1 | Batch size |
| `N_EPOCHS` | 50 | Training epochs |
| `N_EPOCHS_DECAY` | 50 | Learning rate decay epochs |
| `LR` | 2e-4 | Generator learning rate |
| `D_LR` | 1e-4 | Discriminator learning rate |
| `LAMBDA_CYCLE` | 10.0 | Cycle consistency loss weight |
| `LAMBDA_IDENTITY` | 0.5 | Identity loss weight |

### Training Script

```bash
# Run the Jupyter notebook
jupyter notebook Part3_CycleGAN_Training_Kedar.ipynb
```

Or execute directly in Python environment.

### Output

- Checkpoints saved as `epoch_XXX.pth`
- Generated predictions:
  - `pred_A2B/` - Monet to Photo translations
  - `pred_B2A/` - Photo to Monet translations

## Inference

Load a checkpoint and generate translations:

```python
import torch
from torch.utils.data import DataLoader

checkpoint = torch.load("epoch_070.pth", map_location=device)
G_B2A.load_state_dict(checkpoint['G_B2A'])
G_B2A.eval()

# Generate predictions
generate_images(G_B2A, 'dataset/photo_jpg/', 'pred_B2A/')
```

## Model Architecture

### Generators (ResNet-based)

- 9 residual blocks
- Instance normalization
- Reflection padding
- ~11.4M parameters each

### Discriminators (PatchGAN)

- 70×70 receptive field
- ~2.8M parameters each

## Loss Functions

| Loss | Purpose | Weight |
|------|---------|--------|
| Adversarial (MSE) | Realistic output | 1.0 |
| Cycle Consistency (L1) | Content preservation | 10.0 |
| Identity (L1) | Color/tone preservation | 0.5 |

## Comparison Results

### Architecture Comparison

| Component | Photo→Monet | Monet→Photo |
|-----------|------------|------------|
| Generator | G_B2A | G_A2B |
| Role | Style transfer | Reverse translation |

### Loss Components During Training

- Generator loss decreases as discriminators learn
- Cycle consistency loss ensures reversibility
- Identity loss prevents unnecessary color shifts
- Adversarial loss drives stylistic quality

## Reference

Zhu et al. (2017). "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"

See `CycleGAN_Learning_Report.md` for detailed paper analysis.

## License

Educational project for DATA 266 Lab Q3.
