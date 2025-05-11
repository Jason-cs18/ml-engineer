=========
Makefile
=========

``Makefile`` automates the process of building and installing the package. Usually, we use it to help us manage ML experiments.

Manage ML experiments
--------------------------------------

1. Create a ``Makefile`` in the root directory of your project.

.. code-block:: bash

    SHELL=/bin/bash

    CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

    .PHONY: help generate copy

    help:

    @echo "make help: print all available commands"

    @echo "make generate: generate a new template video (params: img_path)"

    @echo "make move: copy generated video to output folder (params: vid_path, target_path)"

    generate:

    ($(CONDA_ACTIVATE) /mnt/data/envs/LivePortrait; python -W ignore e2e_video.py --image_path ${img_path})

    move:

    cp /mnt/data/model_zoo/LivePortrait/animations/${vid_path} /mnt/data1/production/docker/avatar_2d/${target_path}

2. Run ``make`` command with the desired target and parameters. For example, to generate a new template video, run:

.. code-block:: bash
    
    make generate img_path=/path/to/image