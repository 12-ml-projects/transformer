import torch


def gpt3_adamw(
    model: torch.nn.Module, lr: float, weight_decay: float = 0.1
) -> torch.optim.AdamW:
    """AdamW with the GPT-3 settings (betas 0.9 / 0.95, eps 1e-8, weight decay 0.1).

    Weight decay applies only to weight matrices and embeddings: shrinking biases
    and LayerNorm parameters towards zero does not regularize anything useful.
    """
    parameters = [p for p in model.parameters() if p.requires_grad]

    groups = [
        {
            "params": [p for p in parameters if _is_matrix(p)],
            "weight_decay": weight_decay,
        },
        {"params": [p for p in parameters if not _is_matrix(p)], "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def _is_matrix(parameter: torch.nn.Parameter) -> bool:
    """Weight matrices and embeddings are decayed; vectors (biases, norms) are not."""
    return parameter.dim() >= 2
