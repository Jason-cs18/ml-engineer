==========
AIGC model
==========

Scaling law for training
------------------------
Scaling law is an empirical scaling law that describes how neural network performance changes (e.g., loss :math:`L`) as key factors are scaled up or down. These factors typically include the number of parameters :math:`N`, training dataset size :math:`D`, and training cost :math:`C`. The scaling law is usually expressed as a power-law relationship between these factors and the performance metric.

.. math::

   N*(C) = A_{1} \cdot C^{\alpha_{1}}, D*(C) = A_{2} \cdot C^{\alpha_{2}}

In practice, we often conduct many experiments in a limited budget and find the compute-optimal scaling law via fitting. With the scaling law, we can predict the optimal model size :math:`N*(C)` and dataset size :math:`D*(C)` for a given compute budget :math:`C`.

.. figure:: ./images/llama3_scaling.png
   :align: center
   :alt: Ray Cluster Architecture

   Llama3 scaling law

Scaling law for inference
---------------------------
Unlike training, scaling law for inference is dependent on the model size and the inference strategy.

Large language model (LLM)
--------------------------

.. list-table:: 
   :header-rows: 1

   * - Name
     - Affiliation
     - Date
     - Model size
     - Model architecture
     - Pre-training
     - Post-training
     - Inference
     - Highlight
   * - `Llama3 <https://arxiv.org/pdf/2407.21783>`_
     - Llama Team, AI @ Meta
     - 2024.07
     - 8B, 70B, 405B
     - Encoder-decoder, group query attention (GQA), attention mask
     - PT (8k-2.87T token -> 128k-800B toke -> 40M token using smaller lr)
     - SFT + DPO 
     - Pipeline parallelism, FP8 quantization
     - Multilingual support, multimodal support, safety migations with Llama Guard 3
    
Multimodal models
-------------------------------

.. list-table:: 
   :header-rows: 1

   * - Name
     - Affiliation
     - Date
     - Model architecture
     - Training strategy
     - Highlight
   * - `CLIP <https://openai.com/blog/clip/>`_
     - OpenAI
     - ICML 2021
     - Text encoder + Image encoder
     - Contrastive learning
     - The first work to reveal zero-shot capability with multimodal constrastive learning
   * - `LLaVa <https://llava-vl.github.io/>`_
     - University of Wisconsin-Madison, Microsoft Research, and Columbia University
     - NeurIPS 2023
     - Image encoder + projector + language model
     - Instruction tuning
     - The first work to show visual capability of LLM
   * - `Stable Diffusion (SD) <https://arxiv.org/abs/2112.10752>`_
     - Stability AI
     - CVPR 2022
     - VAE: text encoder + diffusion (U-Net) + image decoder 
     - Reverse diffusion process
     - Text to image generation, compute-intensive (less parameters but slow)
   * - `Diffusion Transformer (DiT) <https://arxiv.org/abs/2212.09748>`_
     - UC Berkeley and New York University
     - ICCV 2023
     - VAE: text encoder + diffusion (transformer) + image decoder
     - Reverse diffusion process
     - Compute-optimal, better scaling