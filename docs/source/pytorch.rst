==========
PyTorch
==========
PyTorch is an open-source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab. It is free and open-source software released under the Modified BSD license.

``torch.compile``
-----------------

Accelerate inference
^^^^^^^^^^^^^^^^^^^^^^
xxx

Accelerate training
^^^^^^^^^^^^^^^^^^^^^^
xxx

Profiling
----------------------
xxx

Use PyTorch Lightning
----------------------

`PyTorch Lightning <https://github.com/Lightning-AI/pytorch-lightning>`_ provides a lightweight PyTorch wrapper to help researchers and practitioners streamline their code and make it more readable and maintainable.

Define the training workflow. Here's a toy example:

.. code-block:: python

    # main.py
    # ! pip install torchvision
    import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
    import lightning as L
    from lightning import loggers

    # --------------------------------
    # Step 1: Define a LightningModule
    # --------------------------------
    # A LightningModule (nn.Module subclass) defines a full *system*
    # (ie: an LLM, diffusion model, autoencoder, or simple image classifier).


    class LitAutoEncoder(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
            self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

        def forward(self, x):
            # in lightning, forward defines the prediction/inference actions
            embedding = self.encoder(x)
            return embedding

        def training_step(self, batch, batch_idx):
            # training_step defines the train loop. It is independent of forward
            x, _ = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = F.mse_loss(x_hat, x)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            # this is the validation loop
            x, _ = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            val_loss = F.mse_loss(x_hat, x)
            self.log("val_loss", val_loss)

        def test_step(self, batch, batch_idx):
            # this is the test loop
            x, _ = batch
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            test_loss = F.mse_loss(x_hat, x)
            self.log("test_loss", test_loss)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer


    # -------------------
    # Step 2: Define data
    # -------------------
    dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
    train, val = data.random_split(dataset, [55000, 5000])

    # -------------------
    # Step 3: Train
    # -------------------
    autoencoder = LitAutoEncoder()
    trainer = L.Trainer(accelerator="gpu", devices=8, logger=TensorBoardLogger("logs/"))
    # trainer.test(model, dataloaders=DataLoader(test_set))
    trainer.fit(autoencoder, data.DataLoader(train), data.DataLoader(val))

Run the model on your terminal

.. code-block:: bash

    pip install torchvision
    python main.py

Export to torchscript (JIT)

.. code-block:: python

    # torchscript
    autoencoder = LitAutoEncoder()
    torch.jit.save(autoencoder.to_torchscript(), "model.pt")

Export to ONNX

.. code-block:: python

    # onnx
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmpfile:
        autoencoder = LitAutoEncoder()
        input_sample = torch.randn((1, 64))
        autoencoder.to_onnx(tmpfile.name, input_sample, export_params=True)
        os.path.isfile(tmpfile.name)

Develop a reusable datamodule

.. code-block:: python
    
    import lightning as L
    from torch.utils.data import random_split, DataLoader

    # Note - you must have torchvision installed for this example
    from torchvision.datasets import MNIST
    from torchvision import transforms


    class MNISTDataModule(L.LightningDataModule):
        def __init__(self, data_dir: str = "./"):
            super().__init__()
            self.data_dir = data_dir
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        def prepare_data(self):
            # download
            MNIST(self.data_dir, train=True, download=True)
            MNIST(self.data_dir, train=False, download=True)

        def setup(self, stage: str):
            # Assign train/val datasets for use in dataloaders
            if stage == "fit":
                mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
                self.mnist_train, self.mnist_val = random_split(
                    mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
                )

            # Assign test dataset for use in dataloader(s)
            if stage == "test":
                self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

            if stage == "predict":
                self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

        def train_dataloader(self):
            return DataLoader(self.mnist_train, batch_size=32)

        def val_dataloader(self):
            return DataLoader(self.mnist_val, batch_size=32)

        def test_dataloader(self):
            return DataLoader(self.mnist_test, batch_size=32)

        def predict_dataloader(self):
            return DataLoader(self.mnist_predict, batch_size=32)

Use the datamodule

.. code-block:: python

    dm = MNISTDataModule()
    model = Model()
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)
    trainer.validate(datamodule=dm)
    trainer.predict(datamodule=dm)

Find training loop bottlenecks

.. code-block:: python

    trainer = Trainer(profiler="simple")

.. code-block:: bash

    FIT Profiler Report

    -------------------------------------------------------------------------------------------
    |  Action                                          |  Mean duration (s) |  Total time (s) |
    -------------------------------------------------------------------------------------------
    |  [LightningModule]BoringModel.prepare_data       |  10.0001           |  20.00          |
    |  run_training_epoch                              |  6.1558            |  6.1558         |
    |  run_training_batch                              |  0.0022506         |  0.015754       |
    |  [LightningModule]BoringModel.optimizer_step     |  0.0017477         |  0.012234       |
    |  [LightningModule]BoringModel.val_dataloader     |  0.00024388        |  0.00024388     |
    |  on_train_batch_start                            |  0.00014637        |  0.0010246      |
    |  [LightningModule]BoringModel.teardown           |  2.15e-06          |  2.15e-06       |
    |  [LightningModule]BoringModel.on_train_start     |  1.644e-06         |  1.644e-06      |
    |  [LightningModule]BoringModel.on_train_end       |  1.516e-06         |  1.516e-06      |
    |  [LightningModule]BoringModel.on_fit_end         |  1.426e-06         |  1.426e-06      |
    |  [LightningModule]BoringModel.setup              |  1.403e-06         |  1.403e-06      |
    |  [LightningModule]BoringModel.on_fit_start       |  1.226e-06         |  1.226e-06      |
    -------------------------------------------------------------------------------------------