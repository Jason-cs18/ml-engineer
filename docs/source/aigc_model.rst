==========
AIGC model
==========

Scaling law for training
------------------------
Scaling law is an empirical scaling law that describes how neural network performance changes (e.g., loss :math:`L`) as key factors are scaled up or down. These factors typically include the number of parameters :math:`N`, training dataset size :math:`D`, and training cost :math:`C`. With scaling law, we can estimate optimal model size and training dataset size for a given training cost. 

Scaling law for inference
---------------------------
Unlike training, scaling law for inference is dependent on the model size and the inference strategy.

Large language model (LLM)
--------------------------

.. list-table:: Milestones
   :header-rows: 1

   * - Name
     - Affiliation
     - Date
     - Model size
     - Pre-training
     - Post-training
     - Highlight
   * - `Llama3 <https://arxiv.org/pdf/2407.21783>`_
     - Llama Team, AI @ Meta
     - 2024.07
     - 8B, 70B, 405B
     - xxx
     - xxx
     - Multilingual support, multimodal support, safety migations with Llama Guard 3