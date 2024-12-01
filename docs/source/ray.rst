=========
Ray
=========

Ray is a unified distributed framework for AI workloads. With Ray, you can build a scalable AI platform for training, inference, and serving machine learning models.

Ray Core (simple applications)
--------------------------------------------
xxx

Ray Data (data engineering)
----------------------------
xxx


Ray Train (distributed training)
--------------------------------
xxx

Ray Serve (inference server)
----------------------------
xxx

Ray Tune (hyperparameter tuning)
--------------------------------
xxx

Ray Cluster (AI platform)
-------------------------

`Ray Cluster <https://docs.ray.io/en/latest/cluster/getting-started.html>`_ provides a scalable and flexible way to deploy and manage AI applications on Kubernetes. It consists of the following components:

- Ray head node: The master node of the Ray cluster, responsible for managing the cluster and scheduling tasks.
- Ray worker nodes: The worker nodes of the Ray cluster, responsible for executing tasks and storing data.

.. figure:: ./images/ray-cluster.png
   :align: center
   :alt: Ray Cluster Architecture

With Ray Cluster, users can easily deploy and manage their AI applications via ray job SDK.

.. image:: ./images/ray-job-diagram.png
   :align: center
   :alt: Ray Job Workflow

Install Ray Cluster
^^^^^^^^^^^^^^^^^^^^
.. code:: bash

    pip install 'ray[default]'

Use kuberay to deploy Ray on Kubernetes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`Kuberay <https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html>`_ is a Kubernetes-native operator for managing Ray clusters. Each Ray cluster consists of a head node pod and a collection of worker node pods.

.. image:: ./images/kuberay.png
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


Known Issues
-------------
xxx