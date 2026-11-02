from tinygpt.tokenizer import CharTokenizer

def test_encode_decode():
    text = "Hello, World!"
    tokenizer = CharTokenizer(text)

    sample = "Hello"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    assert decoded == sample, f"Decoded text '{decoded}' does not match original '{sample}'"

def test_vocab_size():
    tokenizer = CharTokenizer("aabbcc")

    assert tokenizer.vocab_size == 3

def test_repeated_characters():
    tokenizer = CharTokenizer("hello")

    ids = tokenizer.encode("ll")

    assert ids[0] == ids[1]