
import pytest
import torch

from tinygpt.dataset import TextDataset


def make_dataset():
    tokens = list(range(200))
    return TextDataset(
        tokens,
        block_size=8,
        train_ratio=0.8,
    )


def test_dataset_split():
    dataset = make_dataset()

    assert len(dataset.train_data) == 160
    assert len(dataset.val_data) == 40


def test_batch_shapes():
    dataset = make_dataset()

    x, y = dataset.get_batch(
        split="train",
        batch_size=4,
    )

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert x.dtype == torch.long


def test_targets_are_shifted():
    dataset = make_dataset()

    x, y = dataset.get_batch(
        split="train",
        batch_size=4,
    )

    assert torch.equal(x[:, 1:], y[:, :-1])


def test_validation_batch():
    dataset = make_dataset()

    x, y = dataset.get_batch(
        split="val",
        batch_size=2,
    )

    assert x.shape == (2, 8)
    assert y.shape == (2, 8)


def test_invalid_block_size():
    with pytest.raises(ValueError):
        TextDataset(
            list(range(100)),
            block_size=0,
        )


def test_invalid_split():
    dataset = make_dataset()

    with pytest.raises(ValueError):
        dataset.get_batch(split="test")


def test_too_small_dataset():
    with pytest.raises(ValueError):
        TextDataset(
            list(range(10)),
            block_size=8,
            train_ratio=0.9,
        )
