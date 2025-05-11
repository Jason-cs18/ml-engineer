==========
AIGC model
==========

Modeling techniques for AIGC
-----------------------------



Taking a face generation as an example, we will introduce representative generative models and highlight their differences. The generation process aims to estimate the probability distribution of the face data :math:`p(x)`. With bayes rule, we can estimate the probability of the data given the model :math:`p(x|z)` by marginalizing over the latent variable :math:`z`:

.. math::

  p(x) = \int p(x|z)p(z) dz

**Autoencoder (AE)** models :math:`p(x)` where :math:`x` is a face. We cannot control the generated face. 

**Variational Autoencoder (VAE)** models :math:`p(x|z)` where :math:`z` is a latent continuous variable (e.g., expression).

**Vector Quantized Variational Autoencoder (VQ-VAE)** models :math:`p(x|z)` where :math:`z` is a discrete latent variable (e.g., gender).

**Autoregressive** models a joint distribution :math:`p(x_1, x_2, ..., x_T)` where :math:`x_1, x_2, ..., x_T` are the pixels of the face.

**Generative Adversarial Networks (GANs)** employs a discriminator :math:`D(x)` to distinguish the real data :math:`x` from the generated data :math:`G(z)`. The generator :math:`G(z)` tries to fool the discriminator.

**Diffusion Models** WIP

**Neural Radience Fields (NeRF)** WIP

**3D Gaussian Splatting (3DGS)** WIP

**Generative Video Models** WIP

Foundation model development
----------------------------
Developing a large-scale foundation model is a challenging task due to the high computational cost and the need for large-scale data. In this section, we will introduce the key components of a foundation model and accelerate the development process with `NeMo <https://github.com/NVIDIA/NeMo>`_.

`NeMo <https://github.com/NVIDIA/NeMo>`_ is a scalable and high-performant generative AI framwork developed by NVIDIA that provides a set of tools for building large-scale foundation models (LLM, MLLM, and TTS).

.. figure:: https://docs.nvidia.com/nemo-framework/user-guide/latest/_images/nemo-llm-mm-stack.png
   :align: center

   NeMo Overview

As shown in the figure, the lifecycle of a foundation model development includes the following steps: 

- `data curation <https://github.com/NVIDIA/NeMo-Curator>`_: extract/synthetic high-quality data
- `training and customization <https://github.com/NVIDIA/NeMo-Run>`_: supervised fine-tuning and parameter-efficient fine-tuning
- `alignment <https://github.com/NVIDIA/NeMo-Aligner>`_: align the model with human values (DPO, SteerLM, RLHF)
- `deployment and inference <https://docs.nvidia.com/nemo-framework/user-guide/latest/deployment/llm/index.html#deploy-nemo-framework-models-llm>`_: TensorRT-LLM/vLLM on NVIDIA Triton inference server
- `multimodal models development <https://docs.nvidia.com/nemo-framework/user-guide/latest/multimodalmodels/index.html#>`_: multimodal llms, vision-language models, text2image and NeRF

Tutorial notebooks are listed `here <https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html>`_.


References
-----------
1. Kaiming He. "`6.S978 Deep Generative Models (MIT EECS, 2024 Fall). <https://mit-6s978.github.io/schedule.html>`_"