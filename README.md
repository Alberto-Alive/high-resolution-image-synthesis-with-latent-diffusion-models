This will be a summary of the research paper:
- [**High-Resolution Image Synthesis with Latent Diffusion Models** (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752)

- [PDF](https://arxiv.org/pdf/2112.10752.pdf)


**Abstract**
So diffusion models start with random noise (pixels are randomly arranged/colors). But this noise has a certain distribution - Gaussian noise aka normal distribution aka bell shaped which means the randomness tends to cluster around the mean.

Gaussian randomness/distribution is kind of needed because out of all distribtuions has the following properties:

- has max entropy (maximally random given variance)

- noising and denoising gaussian results in gaussian so it is predictable as it maintains distribution

Apparently at each denoising step if the model works with text prompts the text embedding is reused every denoising step.

The problem is when the model works on full-resolution pixels and the training takes a lot of GPU resources.

The solution... obviously don't run diffusion on pixels. The actual problem is that ml algorithms compress and learn the information at the same time but we need better compression so:

- first use a pretrained autoencoder to compress images into smaller "latent" representations that still keep important information

- train the diffusion model to denoise in this latent space


**Introduction**

Nothing worth your time except the part about the autoencoder and how that can be trained once and be reused in training different diffusion models in latent space - it learns a good map from image to latent code (encode) and from latent code to image (decode) and this is universal and independent from the diffusion model.


**Related Work**

The general idea... generating images is hard because images are huge.


1) The trade-offs of othe models

- **GANs** make sharp, high-res images fast but are hard to train and often miss parts of the data
- **Likelihood-based model (more stable traning):**
    - VAEs / flow models: easier to optimize, but images often look less sharp / lower quality than GANs
    - Autoregressive models (AR): great at "probability modeling" but slow to sample (pixel-by-pixel / token-by-token) and their architectures can be very expensive - so they often get stuck at lower resolution.

2) Two stage approaches (compress first then model)

To scale to higher resolutions, many methods do:
1. compress the image into a smaller latent representation
2. train a generative model in that latent space

Examples:
- VQ-VAE + AR prior: compress into discrete codes then AR model generates the codes.
- VQGAN / DALL-E-style: use perceptual + adversarial losses to get better latents, then an AR transformer generates them (works, but often needs very high compression to keep AR training feasible -> can hurt details or requires giant models)

What they complained of... AR priors often force a nasty trade-off:
- compress a lot -> lose detail
- compress less -> AR prior becomes too expensive (often billions of parameters)

3) Diffusion models (great quality but expensive in pixel space)

Diffusion models are currently top-tier for quality and density modeling, especially with a U-Net backbone and some training tricks.
But: training and sampling in pixel space is very costly because the model processes full res images for many steps and compute heavy gradients.