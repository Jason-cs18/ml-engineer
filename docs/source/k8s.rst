==================
Kubenetes
==================


Kubenetes (k8s) is a container orchestration platform that automates the deployment, scaling, and management of containerized applications. It provides a framework for deploying and managing containerized applications at scale, and is widely used in production environments.

Docker
------

Docker is a platform for developing, shipping, and running applications inside containers. It provides a lightweight and portable way to package and run applications, and is widely used in containerized environments such as Kubernetes.

Install nvidia-docker2
^^^^^^^^^^^^^^^^^^^^^^

To use nvidia gpus in docker, we need to install nvidia-docker2. You can use following commands to install it.

.. code-block:: bash

  distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

  apt-get update
  apt-get install -y nvidia-docker2
  sudo systemctl restart docker

.. note::
  After installation, you need to check whether your nvidia driver is available on the host machine via ``nvidia-smi``. 

Setup proxy for docker
^^^^^^^^^^^^^^^^^^^^^^

In China, we need to setup proxy for docker to access the internet (e.g., dockerhub, github, etc.). You can use following commands to setup proxy for docker.

.. code-block:: bash

  sudo mkdir -p /etc/systemd/system/docker.service.d
  sudo vim /etc/systemd/system/docker.service.d/http-proxy.conf
  # add following lines to the file
  [Service]
  Environment="HTTP_PROXY=http://127.0.0.1:7890"
  Environment="HTTPS_PROXY=http://127.0.0.1:7890"
  Environment="NO_PROXY=localhost,127.0.0.1"
  
  mkdir -p /etc/systemd/system/containerd.service.d
  sudo vim /etc/systemd/system/containerd.service.d/http-proxy.conf
  # add following lines to the file
  [Service]
  Environment="HTTP_PROXY=http://127.0.0.1:7890"
  Environment="HTTPS_PROXY=http://127.0.0.1:7890"
  Environment="NO_PROXY=localhost,127.0.0.1"
  
  sudo systemctl daemon-reload
  sudo systemctl restart docker

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

Load local images to kind
^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes, we can not pull images via kind and need to load local images instead. You can use following commands to do it.

.. code-block:: bash

  kind load docker-image my-custom-image --name cluster-name

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