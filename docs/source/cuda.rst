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
.. code-block:: python

    cuda_kernel = """
    extern "C" __global__
    void square_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            output[index] = input[index] * input[index];
        }
    }
    """

    import torch
    import torch.utils.cpp_extension

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module = torch.utils.cpp_extension.load_inline(
        name='square',
        cpp_sources='',
        cuda_sources=cuda_kernel,
        functions=['square_kernel']
    )

    def square(input):
        output = torch.empty_like(input)
        threads_per_block = 1024
        blocks_per_grid = (input.numel() + (threads_per_block - 1)) // threads_per_block
        module.square_kernel(blocks_per_grid, threads_per_block, input, output, input.numel())
        return output

    # Example usage
    input_tensor = torch.randn(100, device=device)
    output_tensor = square(input_tensor)

References
-----------
1. MIT 6.S096, Jan 2014. `"Effective Programming In C And C++" <https://ocw.mit.edu/courses/6-s096-effective-programming-in-c-and-c-january-iap-2014/pages/syllabus/>`_.
2. Stanford CS149, Fall 2023. `"Parallel Computing" <https://gfxcourses.stanford.edu/cs149/fall23/>`_.
3. MIT 6.5940, Fall 2023. `TinyEngine and Parallel Processing <https://www.youtube.com/watch?v=HGsvWHqU29Y>`_.
4. MIT Han Lab. `"Parallel Computing Tutorial" <https://github.com/mit-han-lab/parallel-computing-tutorial>`_.
5. MIT 6.5940, Fall 2024. `"Lab 5: Optimize LLM on Edge Devices" <https://drive.google.com/drive/folders/1MhMvxvLsyYrN-4C6eQG8Zj2JeSuyAOf0>`_.
6. Tinkerd. `"Writing CUDA Kernels for PyTorch" <https://tinkerd.net/blog/machine-learning/cuda-basics/#writing-custom-pytorch-kernels>`_ Tech Blog.
7. Richard Zou. `"Custom C++ and CUDA Extensions" <https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#testing-an-operatorl>`_ PyTorch Tutorial.
8. Georgi Gerganov. `"llama.cpp: LLM inference in C/C++" <https://github.com/ggerganov/llama.cpp>`_ Github repo.
9. Christopher Ré. `"Systems for Foundation Models, and Foundation Models for Systems" <https://neurips.cc/virtual/2023/invited-talk/73990>`_ NeurIPS 2023 invited talk.
10. NVIDIA. `"CUDA C++ Programming Guide" <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>`_ NVIDIA Documentation.