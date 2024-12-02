==================
CUDA
==================

CUDA is a parallel computing platform and application programming interface (API) model created by Nvidia. It allows developers to use a CUDA-enabled graphics processing unit (GPU) for general purpose processing – an approach known as GPGPU (General-Purpose computing on Graphics Processing Units).

Why do we need to learn CUDA?
------------------------------

- Existing foundation models are not hardware-optimized. We need to design hardware-aware models (e.g., `FlashAttention series <https://arxiv.org/abs/2205.14135>`_).
- Not all operators for new emerging models are supported by PyTorch. We have to implement them by ourselves.

CUDA extension for PyTorch
----------------------------
xxx

How to profile a PyTorch model?
-------------------------------

Profile a PyTorch model using `torch.profiler <https://pytorch.org/docs/stable/profiler.html>`_.


References
-----------
1. MIT 6.S096, Jan 2014. `"Effective Programming In C And C++" <https://ocw.mit.edu/courses/6-s096-effective-programming-in-c-and-c-january-iap-2014/pages/syllabus/>`_.
2. Stanford CS149, Fall 2023. `"Parallel Computing" <https://gfxcourses.stanford.edu/cs149/fall23/>`_.
3. MIT 6.5940, Fall 2023. `TinyEngine and Parallel Processing <https://www.youtube.com/watch?v=HGsvWHqU29Y>`_.
4. MIT Han Lab. `"Parallel Computing Tutorial" <https://github.com/mit-han-lab/parallel-computing-tutorial>`_.
5. MIT 6.5940, Fall 2024. `"Lab 5: Optimize LLM on Edge Devices" <https://drive.google.com/drive/folders/1MhMvxvLsyYrN-4C6eQG8Zj2JeSuyAOf0>`_.
6. Tinkerd. `"Writing CUDA Kernels for PyTorch" <https://tinkerd.net/blog/machine-learning/cuda-basics/#writing-custom-pytorch-kernels>`_ Tech Blog.
7. Peter Goldsborough. `"Custom C++ and CUDA Extensions" <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_ PyTorch Tutorial.
8. Georgi Gerganov. `"llama.cpp: LLM inference in C/C++" <https://github.com/ggerganov/llama.cpp>`_ Github repo.
9. Christopher Ré. `"Systems for Foundation Models, and Foundation Models for Systems" <https://neurips.cc/virtual/2023/invited-talk/73990>`_ NeurIPS 2023 invited talk.