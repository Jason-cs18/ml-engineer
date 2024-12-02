===========
AIGC System
===========

Memory managment for LLM serving
--------------------------------
LLM inference demands a lot of memory for KV caches. Especially for long-context requests and high concurrent requests, the large memory usage limits the scalability of LLM serving.

To improve this, three techniques are widely used: continuous batching, paged attention, and vAttention.

**Orca** (`Yu et al. <https://www.usenix.org/conference/osdi22/presentation/yu>`_) proposed continuous batching 

**PagedAttention** (`Kwon et al. <https://arxiv.org/pdf/2309.06180>`_) aims to xxx

**vAttention** (`Prabhu et al. <https://arxiv.org/pdf/2405.04437>`_) targets to 


LLM based agents development
--------------------------------
What are LLM based agents?

How to build LLM based agents?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
workflow and tools

How to optimize LLm based agents?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
challenges and solutions

Best practices
^^^^^^^^^^^^^^
tips for building and optimizing compound AI systems

Case studies on researchers
^^^^^^^^^^^^^^^^^^^^^^^^^^^
an autonmous AI agent can draft a research proposal and write a paper.

Case studies on expert-level assistants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
an AI agent can assist a people to analysis data and write a report.

References
-----------
1. Yu et al. `"Orca: A Distributed Serving System for Transformer-Based Generative Models" <https://www.usenix.org/conference/osdi22/presentation/yu>`_ OSDI 2022.
2. Kwon et al. `"Efficient Memory Management for Large Language Model Serving with PagedAttention" <https://arxiv.org/pdf/2309.06180>`_ SOSP 2023.
3. Prabhu et al. `"vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention" <https://arxiv.org/pdf/2405.04437>`_ OSDI 2024.