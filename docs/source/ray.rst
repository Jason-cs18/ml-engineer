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
- Kubernetes cluster: The underlying infrastructure for the Ray cluster, providing scalability and high availability.

.. image:: https://mmbiz.qpic.cn/mmbiz_png/x1nibL49E8dNHCUNAImfqFYlWdyjE75UclqPVqzUjicbd4f6144LDKlNCZujc6RaTa5N8rKdBJzribXiaIeTLttHwg/640?wx_fmt=png&wxfrom=13&tp=wxpic
   :align: center
   :alt: KubeRay Architecture

Install Ray Cluster
^^^^^^^^^^^^^^^^^^^^
.. code:: bash

    pip install 'ray[default]'

Use kuberay to deploy Ray on Kubernetes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
xxx

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