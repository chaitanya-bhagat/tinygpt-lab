
import torch

from tinygpt.bigram import BigramLanguageModel


def test_forward_shapes():
    model = BigramLanguageModel(vocab_size=10)

    x = torch.randint(0, 10, (2, 4))

    logits, loss = model(x)

    assert logits.shape == (2, 4, 10)
    assert loss is None


def test_forward_with_targets():
    model = BigramLanguageModel(vocab_size=10)

    x = torch.randint(0, 10, (2, 4))
    y = torch.randint(0, 10, (2, 4))

    logits, loss = model(x, y)

    assert logits.shape == (2, 4, 10)
    assert loss is not None
    assert torch.isfinite(loss)


def test_backward():
    model = BigramLanguageModel(vocab_size=10)

    x = torch.randint(0, 10, (2, 4))
    y = torch.randint(0, 10, (2, 4))

    _, loss = model(x, y)

    loss.backward()

    assert model.token_embedding_table.weight.grad is not None


def test_generate():
    model = BigramLanguageModel(vocab_size=10)

    context = torch.zeros((1, 1), dtype=torch.long)

    generated = model.generate(
        context,
        max_new_tokens=5,
    )

    assert generated.shape == (1, 6)
    assert torch.all(generated >= 0)
    assert torch.all(generated < 10)
