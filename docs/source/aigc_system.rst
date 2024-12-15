===========
AIGC system
===========

Fast LLM serving
--------------------------------

Memory optimization for LLM serving
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
LLM inference demands a lot of memory for KV caches. Especially for long-context requests and high concurrent requests, the large memory usage limits the scalability of LLM serving.

To improve this, three techniques are widely used: continuous batching, paged attention, and vAttention.

**Orca** (`Yu et al. <https://www.usenix.org/conference/osdi22/presentation/yu>`_) proposed continuous batching 

**PagedAttention** (`Kwon et al. <https://arxiv.org/pdf/2309.06180>`_) aims to xxx

**vAttention** (`Prabhu et al. <https://arxiv.org/pdf/2405.04437>`_) targets to 

**RadixAttention** (`Zhang et al. <https://arxiv.org/pdf/2312.07104>`_) aims to xxx

Fast diffusion serving
-----------------------
Unlike LLM, diffusion models are compute-intensive and lead to slow inference. To improve the performance, some advanced techniques are employed. 

**Consistency Models** (`Song et al. <https://proceedings.mlr.press/v202/song23a>`_) proposes an effective distillation technique xxx

**Latent Consistency Models** (`Luo et al. <https://arxiv.org/abs/2310.04378>`_) leverages xxx to accelerate latent diffusion in further.

**DistriFusion** (`Li et al. <https://hanlab.mit.edu/projects/distrifusion>`_) aims to accelerate diffusion inference within a distributed manner. xxx

**Approximate Caching** (`Agarwal et al. <https://www.usenix.org/conference/nsdi24/presentation/agarwal-shubham>`_) designs zzz

References
-----------
1. Yu et al. `"Orca: A Distributed Serving System for Transformer-Based Generative Models" <https://www.usenix.org/conference/osdi22/presentation/yu>`_ OSDI 2022.
2. Kwon et al. `"Efficient Memory Management for Large Language Model Serving with PagedAttention" <https://arxiv.org/pdf/2309.06180>`_ SOSP 2023.
3. Prabhu et al. `"vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention" <https://arxiv.org/pdf/2405.04437>`_ OSDI 2024.
4. Zheng et al. `"SGLang: Efficient Execution of Structured Language Model Programs" <https://arxiv.org/pdf/2312.07104>`_ arXiv preprint arXiv:2312.07104 (2023).
5. Song et al. `"Consistency Models" <https://proceedings.mlr.press/v202/song23a>`_ ICML 2023.
6. Luo et al. `"Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference" <https://arxiv.org/abs/2310.04378>`_ arXiv preprint arXiv:2310.04378 (2023).
7. Li et al. `"DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models" <https://hanlab.mit.edu/projects/distrifusion>`_ CVPR 2024 Highlight.
8. Agarwal et al. `"Approximate Caching for Efficiently Serving Text-to-Image Diffusion Models" <https://www.usenix.org/conference/nsdi24/presentation/agarwal-shubham>`_ NDSI 2024.