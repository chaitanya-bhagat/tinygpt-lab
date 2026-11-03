
from pathlib import Path

from tinygpt.tokenizer import CharTokenizer
from tinygpt.dataset import TextDataset


def main():
    text = Path("data/input.txt").read_text(
        encoding="utf-8"
    )

    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)

    dataset = TextDataset(
        token_ids,
        block_size=8,
        train_ratio=0.9,
    )

    x, y = dataset.get_batch(
        split="train",
        batch_size=4,
    )

    print("Vocabulary size:", tokenizer.vocab_size)
    print("Training tokens:", len(dataset.train_data))
    print("Validation tokens:", len(dataset.val_data))
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)

    print("\nFirst training example:")
    print("Input:", tokenizer.decode(x[0].tolist()))
    print("Target:", tokenizer.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
