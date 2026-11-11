# GastroVisionGAN

**A Denoising Diffusion Guided Generative Adversarial Network with Adaptive Feedback Mechanism for Synthetic Data Generation and Improved Classification of Gastrointestinal Diseases.**

Doctors reviewing capsule endoscopy footage have to go through thousands of frames, and the datasets we have to train AI on are small, varied, and expensive to label. GastroVisionGAN tackles this by generating realistic synthetic GI images that keep the lesion and the surrounding tissue structure intact.

Trained and tested on the public **Kvasir-v2** dataset (8 classes, 8,000 images).


---

## 1. Overview of the Proposed GastroVisionGAN Architecture

- **Denoising-diffusion GAN.** Based on DDGAN, with an 8-step VP-SDE noise schedule (`T = 8`, `β_min = 0.1`, `β_max = 4.0`). Few steps means fast generation.
- **DenseNet-121 generator.** Pretrained DenseNet-121 encoder, residual-upsampling decoder, and FPN-style skip connections that bring encoder detail back into the decoder. All BatchNorm layers are swapped for GroupNorm so training stays stable with small batches.
- **CBAM attention.** Channel + spatial attention at the 64×64 and 128×128 decoder stages, where texture and structure both matter.
- **Two-headed discriminator.** A patch head looks at local texture, a global head judges the whole image, and a small layer combines the two. Spectral norm and timestep embeddings keep it steady.
- **Adaptive feedback.** Every epoch, the model measures SSIM and PSNR on 96 validation images, smooths them with an EMA, and compares them to targets (SSIM 0.85, PSNR 34 dB). That gap is fed back into the generator as a conditioning signal, so it knows how far it still has to go.
- **Hybrid loss.** Adversarial + L1 reconstruction + VGG-19 perceptual + Jensen–Shannon divergence (weights 2.0 / 2.0 / 2.5 / 0.05).

---

## 2. How it works

```
 real endoscopy image (256 x 256)
   │
   ▼
 add noise (forward diffusion, about 62.5% of the way through T = 8)
   │
   ▼
 Generator  (DenseNet-121 encoder → residual decoder + FPN skips + CBAM)
   │   conditioned on: timestep, random vector z, and the quality gap (SSIM, PSNR)
   ▼
 denoise step by step  (t = 5 → 1)
   │
   ▼
 synthetic image   ← same anatomy and lesion position, new appearance
   │
   ▼
 Discriminator  (patch head + global head)  →  real / fake
```


---

## 3. Getting started

### Install

```bash
git clone https://github.com/yumnawaheedyw/GASTROVISIONGAN-GI-Medical-Image-Generation.git
cd GASTROVISIONGAN-GI-Medical-Image-Generation
pip install -r requirements.txt
```

### Get the data

Download **Kvasir-v2** from [Simula](https://datasets.simula.no/kvasir/) (it is also on Kaggle). Make sure the folder has one sub-folder per class, like this:

```
kvasir-v2/
├── dyed-lifted-polyps/
├── dyed-resection-margins/
├── esophagitis/
├── normal-cecum/
├── normal-pylorus/
├── normal-z-line/
├── polyps/
└── ulcerative-colitis/
```

The data is split 70 / 15 / 15 (train / validation / test), stratified, with seed 42.


## 4. Training settings at a glance

| Setting | Value |
|---|---|
| Image size | 256 × 256 |
| Batch size | 16 |
| Epochs | 500 |
| Generator / discriminator learning rate | 2e-6 / 1.5e-6 |
| Diffusion steps | 8 |
| Adaptive warm-up | 12 epochs |
| Synthetic images generated | 10,000 |

---

## 5. Acknowledgements

- [Kvasir-v2 dataset](https://datasets.simula.no/kvasir/) by Simula Research Laboratory
- [Denoising Diffusion GANs](https://arxiv.org/abs/2112.07804) (Xiao et al.)
- [CBAM](https://arxiv.org/abs/1807.06521) (Woo et al.)

## License

No license has been specified yet, so all rights are reserved by default until one is added. The Kvasir-v2 images belong to their original authors and are subject to the dataset's own terms.
