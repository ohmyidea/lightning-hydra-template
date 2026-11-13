from pathlib import Path

import pytest
import torch

from src.data.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "database/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_mnist_preserves_official_test_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Official test samples must never enter the train or validation subsets."""

    class FakeMNIST:
        def __init__(self, root, train, transform=None):
            self.train = train

        def __len__(self):
            return 60_000 if self.train else 10_000

    monkeypatch.setattr("src.data.mnist_datamodule.MNIST", FakeMNIST)
    dm = MNISTDataModule()
    dm.setup()
    assert dm.data_train.dataset.train
    assert dm.data_val.dataset is dm.data_train.dataset
    assert not dm.data_test.train
    assert set(dm.data_train.indices).isdisjoint(dm.data_val.indices)
    assert len(dm.data_train) == 55_000
    assert len(dm.data_val) == 5_000
    assert len(dm.data_test) == 10_000
    train_subset = dm.data_train
    dm.setup()
    assert dm.data_train is train_subset
