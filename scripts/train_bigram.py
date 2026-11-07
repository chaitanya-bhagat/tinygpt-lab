from pathlib import Path

from tinygpt.tokenizer import CharTokenizer
from tinygpt.dataset import TextDataset
from tinygpt.bigram import BigramLanguageModel
import torch


def main():

    torch.manual_seed(42)  # For reproducibility

    # Load the text data
    text = Path("data/input.txt").read_text(encoding="utf-8")

    # Initialize the tokenizer and encode the text
    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)

    # Create the dataset
    block_size = 8
    train_ratio = 0.9
    dataset = TextDataset(token_ids, block_size=block_size, train_ratio=train_ratio)

    # Initialize the model
    vocab_size = tokenizer.vocab_size

    model = BigramLanguageModel(vocab_size=vocab_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    max_steps = 1000
    batch_size = 32

    for step in range(max_steps):
        # Get a batch of data
        x, y = dataset.get_batch(split="train", batch_size=batch_size)

        # Forward pass
        logits, loss = model(x, targets=y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: " f"training loss = {loss.item():.4f}")

    context = torch.zeros((1, 1), dtype=torch.long)
    generated_indices = model.generate(context, max_new_tokens=200)
    print("\nGenerated text:\n")
    print(tokenizer.decode(generated_indices[0].tolist()))


if __name__ == "__main__":
    main()
