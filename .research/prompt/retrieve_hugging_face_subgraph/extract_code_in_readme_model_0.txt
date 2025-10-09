
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
license: cc-by-nc-4.0
---

# Scalable Diffusion Models with Transformers (DiT)

## Abstract

We train latent diffusion models, replacing the commonly-used U-Net backbone with a transformer that operates on latent patches. We analyze the scalability of our Diffusion Transformers (DiTs) through the lens of forward pass complexity as measured by Gflops. We find that DiTs with higher Gflops---through increased transformer depth/width or increased number of input tokens---consistently have lower FID. In addition to good scalability properties, our DiT-XL/2 models outperform all prior diffusion models on the class-conditional ImageNet 512×512 and 256×256 benchmarks, achieving a state-of-the-art FID of 2.27 on the latter.


Output:
{
    "extracted_code": ""
}
