=================
Vision perception
=================
Vision perception aims to understand the environment through images and videos. Driven by cross-modal learning and multimodal models, vision perception models have zero-shot capabilities and thus can be applied to many different tasks. In this section, we will introduce some popular vision perception models and their optimization methods.

Vision foundation models
-------------------------
.. list-table:: 
   :header-rows: 1

   * - Name
     - Affiliation
     - Publication
     - Model architecture
     - Training strategy
     - Highlight
   * - CLIP
     - OpenAI
     - ICML 2021
     - Text & Vison encoder
     - Image-text contrastive pre-training, 400M image-text pairs
     - Zero-shot on image classification
   * - YOLO-World
     - Tencent AI Lab, Tencent PCG, and Huazhong University of Science & Technology
     - CVPR 2024
     - Text & vision encoder & cross-modal fusion module
     - Constrastive learning + object detection, 1.6M images
     - 52 FPS on V100

**CLIP** (`Radford et al. <https://arxiv.org/pdf/2103.00020>`_) is the first work to present zero-shot image classification through image-vision contrastive pre-training. The model employs two encoders, one for images and the other for texts, and learns to map images and texts into a shared space where the similarity between them can be measured. Instead of using existing small datasets, authors have created a large-scale dataset containing 400M image-text pairs for pre-training.

**YOLO-World** (`Cheng et al. <https://arxiv.org/abs/2401.17270>`_) is the first work to explore zero-shot object detetcion with a lightweight backbone (YOLOv8). The model contains a text encoder, a vision encoder, and a cross-modal fusion module. Cross-modal fusion module is used to enhance the features from text and vision encoders and generate the final object detection results. It achieves 52 FPS on V100.

Optimization methods
----------------------

Metrics and evaluation
----------------------

References
----------
1. Radford et al. `"Learning Transferable Visual Models From Natural Language Supervision" <https://arxiv.org/pdf/2103.00020>`_ ICML 2021.
2. Cheng et al. `"YOLO-World: Zero-shot Object Detection with Cross-modal Learning" <https://arxiv.org/abs/2401.17270>`_ CVPR 2024.