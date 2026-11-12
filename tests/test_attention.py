import torch
from tinygpt.attention import Head


def test_attention_output_shape():
    head = Head(n_embd=4, head_size=2, block_size=8)
    x = torch.randn(2, 3, 4)

    out = head(x)

    assert out.shape == (
        2,
        3,
        2,
    ), f"Expected output shape (2, 3, 2), but got {out.shape}"


def test_attention_gradients():
    head = Head(n_embd=4, head_size=2, block_size=8)
    x = torch.randn(2, 3, 4)
    out = head(x)

    out.sum().backward()

    assert head.query.weight.grad is not None
    assert head.key.weight.grad is not None
    assert head.value.weight.grad is not None


def test_attention_is_casual():
    torch.manual_seed(42)

    head = Head(n_embd=4, head_size=2, block_size=8)
    x = torch.randn(1, 3, 4)
    x_modified = x.clone()

    x_modified[:, 2, :] += 100

    out1 = head(x)
    out2 = head(x_modified)

    torch.testing.assert_close(out1[:, :2, :], out2[:, :2,:])