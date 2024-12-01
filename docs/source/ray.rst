=========
Ray
=========

Ray is a unified distributed framework for AI workloads. With Ray, you can build a scalable AI platform for training, inference, and serving machine learning models.

Ray Core
---------
xxx

Ray Data
---------
xxx


Ray Train
---------
xxx

Ray Serve
---------
xxx

Ray Tune
---------
xxx

Ray Cluster
-----------

Ray Cluster provides a scalable and flexible way to deploy and manage AI applications on Kubernetes. It consists of the following components:

- Ray head node: The master node of the Ray cluster, responsible for managing the cluster and scheduling tasks.
- Ray worker nodes: The worker nodes of the Ray cluster, responsible for executing tasks and storing data.

.. raw:: html
    
    <object data="./pinned-octocat.svg" type="image/svg+xml">
    </object>

    
.. figure:: https://docs.ray.io/en/latest/_images/ray-cluster.svg
   :align: center
   :alt: Ray Cluster Architecture

With Ray Cluster, users can easily deploy and manage their AI applications via ray job SDK.

.. image:: https://mmbiz.qpic.cn/mmbiz_png/x1nibL49E8dNHCUNAImfqFYlWdyjE75Uc53G94uQXQBqgvk2bgUatWYhKDwbkHLyibvibGibeHUicRRgFNwAowpMAHw/640?wx_fmt=png&wxfrom=13&tp=wxpic
   :align: center
   :alt: Ray Job Workflow

Install Ray Cluster
^^^^^^^^^^^^^^^^^^^^
.. code:: bash

    pip install 'ray[default]'

Use kuberay to deploy Ray on Kubernetes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Kuberay is a Kubernetes-native operator for managing Ray clusters. Each Ray cluster consists of a head node pod and a collection of worker node pods.

.. image:: https://docs.ray.io/en/latest/_images/ray_on_kubernetes.png
   :align: center
   :alt: KubeRay Workflow

Submit a training job to Ray Cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

xxx

Submit a inference job to Ray Cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

xxx

Submit a serving job to Ray Cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

xxx

Monitor running jobs on Ray Cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

xxx