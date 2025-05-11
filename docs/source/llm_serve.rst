===============
LLM serving
===============

Large language models (LLMs) have demonstrated remarkable capabilieties in various natural language processing tasks. However, deploying these models for real-world applications can be challenging due to their high computational requirements and the need for efficient serving infrastructure.

On this note, we summarize two popular inference engines for LLMs: `vLLM <https://github.com/vllm-project/vllm>`_ and `SGLang <https://github.com/sgl-project/sglang>`_.

Milestone models
----------------
xxx

Main challenges
----------------
xxx

Popular optimizations
----------------------

Inference engine
-----------------

vLLM
^^^^^
`vLLM <https://github.com/vllm-project/vllm>`_ is a high-performance inference engine for LLMs. It is optimized for throughput and employs a lot of advanced features (e.g., PagedAttention, Continuous batching, Speculative decoding).


SGLang
^^^^^^^
`SGLang <https://github.com/sgl-project/sglang>`_ is throughput-optimized inference engine for LLMs. Especially on long context tasks, SGLang can achieve 5x throughput improvement over vLLM.

Best practices
---------------
xxx