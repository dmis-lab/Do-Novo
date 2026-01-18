import torch
import torch.nn as nn

class CrossAttnTopKTheoPeakSampler(nn.Module):
    def __init__(self, d_model, n_bins, hidden_dim, k):
        super().__init__()
        self.k = k
        self.base_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_bins),
        )
        self.prior_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, emb, emb_mask, attn_prior=None):

        logits_base = self.base_mlp(emb[:, 0, :])

        if attn_prior is not None:
            # attn_prior: (B, n_bins), sum=1
            prior_clamped = attn_prior.clamp(min=1e-6, max=1.0-1e-6)
            prior_logit = torch.log(prior_clamped / (1 - prior_clamped))
            logits = logits_base + torch.sigmoid(self.prior_gate) * prior_logit
        else:
            logits = logits_base

        probs = torch.sigmoid(logits)  # (B, n_bins)

        # Top-K straight-through mask
        k = min(self.k, probs.size(1))
        topk_vals, topk_idx = torch.topk(probs, k, dim=1)
        samples_hard = torch.zeros_like(probs)
        samples_hard.scatter_(1, topk_idx, 1.0)
        samples = samples_hard + (probs - probs.detach())
        # samples = torch.zeros_like(probs)
        # samples.scatter_(1, topk_idx, 1.0)

        extras = {"probs": probs}

        return samples, probs, logits, extras, torch.sigmoid(self.prior_gate.detach())