import torch 
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size:int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=vocab_size,
            )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        logits = self.token_embedding_table(idx)
        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
