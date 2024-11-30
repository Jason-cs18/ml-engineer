==================
Kubenetes
==================


Kubenetes (k8s) is a container orchestration platform that automates the deployment, scaling, and management of containerized applications. It provides a framework for deploying and managing containerized applications at scale, and is widely used in production environments.

Known issues
------------

Use new configs to setup docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After modifying the docker config file (``/etc/docker/deamon.json``), you need to restart the docker service to make the changes take effect. 

.. code-block:: bash

  sudo systemctl daemon-reload
  sudo systemctl restart docker

.. note::

  If the commands above don't work, you can try to restart the docker service by using following commands

  .. code-block:: bash
  
    sudo systemctl daemon-reload
    sudo systemctl stop docker.service
    sudo systemctl stop docker.socket
    sudo systemctl start docker.service
    sudo systemctl start docker.socket

Kind
----
`kind <https://kind.sigs.k8s.io/>`_ is a tool for running local Kubernetes clusters using Docker container “nodes”.
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

  ``kind`` need ``Cgroup Version: 2`` to work properly. You can use following commands to check and enable it.

  .. code-block:: bash

    # 1. check if cgroup2 is enabled
    cat /sys/fs/cgroup/cgroup.controllers 
    # 2. if not, enable cgroup2
    sudo vim /etc/default/grub
    # 3. add systemd.unified_cgroup_hierarchy=1 to GRUB_CMDLINE_LINUX
    # 4. update grub
    sudo update-grub
    # 5. reboot
    sudo reboot
