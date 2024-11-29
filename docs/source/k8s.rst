==================
Kubenetes
==================


Kubenetes (k8s) is a container orchestration platform that automates the deployment, scaling, and management of containerized applications. It provides a framework for deploying and managing containerized applications at scale, and is widely used in production environments.

Kind
----
`kind <https://kind.sigs.k8s.io/    >`_ is a tool for running local Kubernetes clusters using Docker container “nodes”.
kind was primarily designed for testing Kubernetes itself, but may be used for local development or CI.

Troubleshooting kind
^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash 

  kind create cluster --retain --image=kindest/node:v1.26.0
  kind export logs
  cat /tmp/xxx/kind-control-plane/journal.log # xxx is the name of log directory

Use kind to create a k8s cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash 

  kind create cluster --image=kindest/node:v1.26.0

.. note::

   ``cgroup`` driver must be ``systemd`` for ``kind`` to work properly.
