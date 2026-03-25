# CycleGAN: Monet Style Transfer — Learning Report

**Course:** DATA 266 — Lab Q3  
**Reference Paper:** *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks* (Zhu et al., 2017)

---

## 1. The Problem: What Are We Doing?

We want to teach a computer to take a real photograph and make it look like a Monet painting, and vice versa. The challenge is that we don't have matching pairs — we don't have a photo and its corresponding Monet painting side by side. We only have two separate collections:

- **Domain A:** 300 Monet paintings (`monet_jpg/`)
- **Domain B:** 7,038 real photographs (`photo_jpg/`)

This is called **unpaired image-to-image translation**. CycleGAN solves this by training two generators simultaneously and enforcing that translations can be reversed (cycle consistency).

---

## 2. The Paper: Key Ideas from Zhu et al.

### Core Insight: Cycle Consistency

If you translate a sentence from English to French and back to English, you should get the original sentence. CycleGAN applies this same idea to images. If we convert a photo to a Monet painting and then convert it back to a photo, we should get the original photo back. This constraint forces the generators to preserve the actual content of the image and only change the style.

### The Architecture (4 Networks)

| Network | Role | Parameters |
|---|---|---|
| **Generator G_A2B** | Translates Monet → Photo (ResNet, 9 residual blocks) | ~11.4M |
| **Generator G_B2A** | Translates Photo → Monet **(the one we care about)** | ~11.4M |
| **Discriminator D_A** | Judges whether a Monet painting is real or fake | ~2.8M |
| **Discriminator D_B** | Judges whether a photograph is real or fake | ~2.8M |

### The Three Losses

#### 1) Adversarial Loss — Makes output look real

The generators try to fool the discriminators. The discriminators try to catch fakes. This competition drives the generators to produce increasingly realistic outputs. The paper uses **LSGAN** (least-squares loss) instead of the original GAN log-likelihood, which is more stable and produces higher quality results.

#### 2) Cycle-Consistency Loss — Preserves content

```
Photo → G_B2A → Fake Monet → G_A2B → Reconstructed Photo ≈ Original Photo
Monet → G_A2B → Fake Photo → G_B2A → Reconstructed Monet ≈ Original Monet
```

Uses L1 loss. Controlled by `lambda_cycle` (set to 10 in the paper). Without this, the generator could turn a landscape into a portrait — still looks like Monet, but loses the original scene entirely.

#### 3) Identity Loss — Preserves color/tone

- If `G_B2A` receives a **real Monet**, it should leave it unchanged (already in target domain).
- If `G_A2B` receives a **real Photo**, it should leave it unchanged.

This prevents unnecessary color shifts. The paper shows (Figure 9) that without identity loss, daytime paintings get mapped to sunset photos. Not part of the original Equation 3 in the paper, but **recommended for style transfer tasks**.

### Full Objective (Paper Equation 3)

```
L(G, F, D_X, D_Y) = L_GAN(G, D_Y, X, Y) + L_GAN(F, D_X, Y, X) + λ * L_cyc(G, F)
```

The paper sets λ = 10. Identity loss is added on top for style transfer tasks, weighted as `lambda_cycle * lambda_identity`.

### Key Paper Finding: Ablation Study (Tables 4 & 5)

The paper tested removing each loss component:

| Variant | Result |
|---|---|
| Cycle loss alone | Very poor (no realism) |
| GAN loss alone | Mode collapse (all outputs look the same) |
| GAN + only forward cycle | Unstable, some mode collapse |
| GAN + only backward cycle | Training collapses |
| **Full CycleGAN (both cycles + GAN)** | **Best results by far** |

**Conclusion:** Both the adversarial loss AND cycle-consistency loss are critical. You need both directions of cycle consistency for stable training.

---

## 3. Training Details from the Paper

- Batch size = 1 with Instance Normalization (not Batch Norm)
- Learning rate = 0.0002, Adam optimizer with beta1=0.5, beta2=0.999
- 100 epochs constant LR + 100 epochs linear decay to zero (200 total)
- Image buffer of 50 previously generated images to stabilize discriminator
- LSGAN (MSE loss) instead of original BCE loss
- PatchGAN discriminator (70x70 patches) — fewer parameters, works on any image size
- Generator: 9 residual blocks for 256x256 images

---

## 4. Our Implementation (Notebook Structure)

| Cell | What It Does |
|---|---|
| Cell 1-2 | Install dependencies (scipy, pillow, tqdm, matplotlib) |
| Cell 4 | Imports and device setup (GPU: RTX 4090) |
| Cell 5 | Configuration: all hyperparameters in one place |
| Cell 7 | Dataset: UnpairedDataset loads Monet and Photo images with augmentation |
| Cell 9 | Model: Generator (ResNet, 9 blocks, 11.4M params) + PatchGAN Discriminator (2.8M params) |
| Cell 11 | Losses (LSGAN + L1 cycle + L1 identity), Adam optimizers, LR schedulers, image pool |
| Cell 13 | Training utilities: save/load checkpoints, generate_images function |
| Cell 14 | Training loop: trains all 4 networks for 300 epochs |
| Cell 16 | Loss curve plots (Generator, Discriminator, Cycle, Identity) |
| Cell 18 | Final image generation: 300 images per direction for evaluation |
| Cell 20-28 | Evaluation: FID and MiFID using Inception v3 features |
| Cell 30-32 | Visualization: side-by-side translations + cycle-consistency check |

### How Evaluation Works

- **FID (Frechet Inception Distance):** Measures how similar the distribution of generated images is to the distribution of real images, using Inception v3 features. **Lower = better.**
- **MiFID (Mean cosine distance):** Measures per-image similarity between generated and real images in feature space. **Lower = better.**
- Both directions are evaluated: Photo→Monet (B2A) and Monet→Photo (A2B)

---

## 5. Hyperparameter Optimization: What We Changed

To achieve FID < 85 and MiFID < 0.5, we made **5 targeted parameter changes** while keeping the notebook structure and architecture identical.

| Parameter | Before | After | Reason |
|---|---|---|---|
| **Total Epochs** | 250 (125+125) | 300 (150+150) | More training for better convergence |
| **LAMBDA_IDENTITY** | 0.5 | 1.0 | Better color/structure preservation |
| **Discriminator LR** | 2e-4 (same as G) | 1e-4 (half of G) | Prevents D from dominating G |
| **ColorJitter** | On (0.05) | Removed | Color is the style — don't distort it |
| **Label Smoothing** | 1.0 (hard) | 0.9 (soft) | Stabilizes LSGAN, reduces mode collapse |
| **SAVE_EVERY** | 25 | 50 | Fewer checkpoint interruptions |

### Change 1: More Training (250 → 300 Epochs)

The loss curves at epoch 25 were still dropping steadily — the model had not converged. We added 50 more epochs (25 extra at constant LR + 25 extra during decay). The constant-LR phase is where the model learns broad patterns. The decay phase is where it fine-tunes details. More of both helps.

### Change 2: Stronger Identity Loss (0.5 → 1.0)

This is arguably the most important change for Monet style transfer. The effective identity loss weight = `LAMBDA_CYCLE x LAMBDA_IDENTITY`.

- **Before:** 10 x 0.5 = 5.0
- **After:** 10 x 1.0 = 10.0 (doubled)

This forces each generator to not alter images that are already in the target domain. For Monet specifically, this preserves the warm color palette, brushstroke characteristics, and overall tone. The paper explicitly recommends identity loss for style transfer (Figure 9) and shows that without it, generators introduce unwanted color shifts.

### Change 3: Lower Discriminator Learning Rate (2e-4 → 1e-4)

When the discriminator learns too fast, it "wins" easily. It perfectly classifies real vs fake, and the gradients it sends back to the generator become uninformative (vanishing gradients). In the original run, D loss dropped to ~0.13 by epoch 7 — a sign of D being too strong. Halving D's learning rate keeps it in a useful range longer, providing richer gradient signals to the generators throughout training.

### Change 4: Removed ColorJitter

ColorJitter randomly changes brightness, contrast, and saturation of training images. In classification tasks, this helps generalization. But in style transfer, **color IS the style**. Monet paintings have a specific warm palette. Randomly distorting those colors during training teaches the model that "any color variation is fine," which directly contradicts what we want. Removing it lets the generators learn Monet's actual color distribution more precisely.

### Change 5: Label Smoothing (1.0 → 0.9)

With hard labels (real=1.0, fake=0.0), the discriminator can become overconfident — it outputs exactly 1.0 for real images and 0.0 for fakes. With soft labels (real=0.9), the discriminator can never be "perfectly right," so it always provides useful gradient information to the generator. This prevents mode collapse (where all generated images look the same) and leads to more diverse, realistic outputs.

---

## 6. Key Concepts Explained

### Lambda (the Balancing Knob)

Lambda controls how much weight each loss gets in the total loss. It is like a volume knob. `LAMBDA_CYCLE = 10` means "content preservation is 10x more important than fooling the discriminator." `LAMBDA_IDENTITY = 1.0` means "color preservation is equally weighted to cycle consistency." If lambda is too low, that loss gets ignored. If too high, it dominates and the model focuses only on that aspect.

### Adam Optimizer (the Smart Learner)

Adam adapts the learning rate for each parameter individually. If a parameter has been getting large gradients, Adam slows it down. If gradients have been small, Adam speeds it up. The `beta1=0.5` (instead of default 0.9) makes Adam less "memory-dependent" on past gradients, which is important for GANs where the loss landscape shifts rapidly. `beta2=0.999` keeps the second moment (gradient variance) stable.

### Learning Rate Schedule (the Speed Controller)

- **Phase 1 (epochs 1-150):** Constant LR = 0.0002. The model explores freely and learns broad patterns — overall composition, color mapping, brushstroke style.
- **Phase 2 (epochs 151-300):** LR decays linearly to 0. The model makes increasingly smaller adjustments, fine-tuning subtle details like texture quality and edge sharpness.

Without decay, the model keeps making large jumps and never settles on a good solution.

### Why Batch Size = 1?

CycleGAN uses Instance Normalization (not Batch Normalization). Instance Norm normalizes each image independently, so batch size doesn't affect normalization statistics. Batch size 1 means each image gets individual attention, leading to better per-image quality. This is the standard recommendation from the paper.

### PatchGAN Discriminator (the Detail Checker)

Instead of giving a single real/fake verdict for the whole image, PatchGAN classifies every 70x70 overlapping patch independently. This forces the generator to produce realistic textures everywhere in the image, not just get the overall composition right. It also has fewer parameters than a full-image discriminator, making training faster and more stable.

---

## 7. Known Limitations (from the Paper)

- **Geometric changes:** CycleGAN struggles with shape transformations (e.g., dog to cat). It works best for texture/color/style changes.
- **Mode collapse risk:** Despite safeguards, the generator may learn a limited range of styles rather than the full diversity of the target domain.
- **Color shifts:** Even with identity loss, subtle unwanted color shifts can occur in extreme lighting.
- **Training data bias:** If the training set doesn't include certain scenarios (e.g., person riding a horse), the model fails on those cases.
- **Dataset imbalance:** We have 300 Monet vs 7,038 photos. The model sees each Monet ~23x more per epoch, which could lead to overfitting on the Monet domain.
- **Resolution:** Training at 256x256 limits fine-grained texture fidelity at higher resolutions.

---

## 8. Summary: What We Did

1. Implemented the CycleGAN architecture from the Zhu et al. paper for Monet to Photo style transfer.
2. Trained two generators (G_A2B, G_B2A) and two discriminators (D_A, D_B) with three losses: adversarial (LSGAN), cycle-consistency (L1), and identity (L1).
3. Optimized 5 hyperparameters based on insights from the paper and established GAN training practices: increased training epochs (300), doubled identity loss (1.0), halved discriminator LR (1e-4), removed ColorJitter, and added label smoothing (0.9).
4. Generated 300 images per direction (`pred_A2B/` and `pred_B2A/`) for evaluation.
5. Evaluated using FID and MiFID metrics with Inception v3 features. **Target: FID < 85, MiFID < 0.5.**

---

*Reference: Zhu, Park, Isola, Efros. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." arXiv:1703.10593, 2017.*
