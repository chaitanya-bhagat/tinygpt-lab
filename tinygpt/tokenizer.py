class CharTokenizer:
    """Character-level tokenizer for TinyGPT."""
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        self.stoi = {char: idx for idx, char in enumerate(self.chars)}
        self.itos = {idx: char for char, idx in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        """Encodes a string into a list of integers."""
        return [self.stoi[char] for char in text]

    def decode(self, indices: list[int]) -> str:
        """Decodes a list of integers back into a string."""
        return "".join(self.itos[idx] for idx in indices)


if __name__ == "__main__":
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    sample = "Hello"

    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Original:", sample)
    print("Encoded:", encoded)
    print("Decoded:", decoded)