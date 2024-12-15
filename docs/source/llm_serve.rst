===============
LLM serving
===============

Large language models (LLMs) have demonstrated remarkable capabilieties in various natural language processing tasks. However, deploying these models for real-world applications can be challenging due to their high computational requirements and the need for efficient serving infrastructure.

On this note, we summarize two popular inference engines for LLMs: `vLLM <https://github.com/vllm-project/vllm>`_ and `SGLang <https://github.com/sgl-project/sglang>`_.

vLLM
----
`vLLM <https://github.com/vllm-project/vllm>`_ is a high-performance inference engine for LLMs. It is optimized for throughput and employs a lot of advanced features (e.g., PagedAttention, Continuous batching, Speculative decoding).

Installation
^^^^^^^^^^^^^^^^^^
.. code-block:: bash
    
    pip install vllm

use vLLM to serve LLMs.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
star a vLLM server

.. code-block:: bash

    vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype auto --api-key token-abc123

More advanced configurations can refer to `OpenAI Compatible Server <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>`_.

send a request

.. code-block:: python

    from openai import OpenAI
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ]
    )
    print("Chat response:", chat_response)

SGLang
-------
`SGLang <https://github.com/sgl-project/sglang>`_ is throughput-optimized inference engine for LLMs. Especially on long context tasks, SGLang can achieve 5x throughput improvement over vLLM.

Installation
^^^^^^^^^^^^^^^^^^
.. code-block:: bash

    pip install --upgrade pip
    # please check https://docs.flashinfer.ai/installation.html for the proper versions of torch and cuda
    pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/

use SGLang to serve LLMs
^^^^^^^^^^^^^^^^^^^^^^^^^^
Use SGLang to serve OpenAI models
""""""""""""""""""""""""""""""""""""""
.. code-block:: bash

    export OPENAI_API_KEY=sk-******

.. code-block:: python

    from sglang import function, system, user, assistant, gen, set_default_backend, OpenAI

    @function
    def multi_turn_question(s, question_1, question_2):
        s += system("You are a helpful assistant.")
        s += user(question_1)
        s += assistant(gen("answer_1", max_tokens=256))
        s += user(question_2)
        s += assistant(gen("answer_2", max_tokens=256))

    set_default_backend(OpenAI("gpt-3.5-turbo"))

    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
    )

    for m in state.messages():
        print(m["role"], ":", m["content"])

    print(state["answer_1"])

More examples are `here <https://github.com/sgl-project/sglang/tree/main/examples/frontend_language/quick_start>`_. We can customized the `backend <https://github.com/sgl-project/sglang/tree/5f2595be430239ba13c5adbe559e21333f5adf9e/python/sglang/lang/backend>`_ manually to support more LLMs.

Use SGLang to serve local embedding models
"""""""""""""""""""""""""""""""""""""""""""
.. code-block:: bash

    # start a sglang server
    python -m sglang.launch_server --model-path Alibaba-NLP/gte-Qwen2-7B-instruct \
    --port 30000 --host 0.0.0.0 --is-embedding

.. code-block:: python

    # send a request
    import requests

    text = "Once upon a time"

    response = requests.post(
        "http://localhost:30000/v1/embeddings",
        json={"model": "Alibaba-NLP/gte-Qwen2-7B-instruct", "input": text},
    )

    text_embedding = response.json()["data"][0]["embedding"]

    print_highlight(f"Text embedding (first 10): {text_embedding[:10]}")

Use SGLang to serve local LLMs
""""""""""""""""""""""""""""""""""""""
.. code-block:: bash

    # start a sglang server
    python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 30000 --host 0.0.0.0

.. code-block:: python
    
    # send a request
    import requests

    url = "http://localhost:30000/v1/chat/completions"

    data = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    }

    response = requests.post(url, json=data)
    print_highlight(response.json())
