import torch


class TextDataset:
    """Prepare tokenized text for next-token prediction."""

    def __init__(
        self, token_ids: list[int], block_size: int = 64, train_ratio: float = 0.9
    ):
        if block_size <= 0:
            raise ValueError("block_size must be a positive integer")

        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")

        self.block_size = block_size

        self.data = torch.tensor(token_ids, dtype=torch.long)

        # Split the data into training and validation sets
        split_idx = int(len(self.data) * train_ratio)

        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]

        for name, split in [("train", self.train_data), ("validation", self.val_data)]:
            if len(split) <= block_size:
                raise ValueError(
                    f"{name} split needs at least" f"{block_size +1} tokens"
                )

    def get_batch(self, split: str = "train", batch_size: int = 8):
        if split not in ("train", "val"):
            raise ValueError("split must be either 'train' or 'val'")

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        data = self.train_data if split == "train" else self.val_data

        max_start = len(data) - self.block_size

        starts = torch.randint(
            low=0, high=max_start, size=(batch_size,)
        )

        x = torch.stack([data[i : i + self.block_size] for i in starts])

        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in starts])

        return x, y