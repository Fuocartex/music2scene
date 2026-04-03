import torch
import torch.nn as nn

class AudioToPromptAdapter(nn.Module):
    """
    Adapter audio→prompt più profondo e regolarizzato.
    - Multi-layer feedforward
    - Residual connections
    - LayerNorm per stabilizzare
    - Dropout leggero
    """
    def __init__(
        self,
        audio_dim=512,
        n_tokens=77,
        hidden_dim=768,
        hidden=1024,
        n_layers=8,
        dropout=0.15,
        use_residual=False,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        layers = []
        in_dim = audio_dim
        for i in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
            ])
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden, n_tokens * hidden_dim)

        # opzionale: mappa anche direttamente l’audio → feature per residuo
        if use_residual:
            self.residual_proj = nn.Linear(audio_dim, n_tokens * hidden_dim)

    def forward(self, x):
        """
        x: (B, 512)
        ritorna (B, n_tokens, hidden_dim)
        """
        y = self.backbone(x)              # (B, hidden)
        out = self.proj(y)                # (B, n_tokens * hidden_dim)
        if self.use_residual:
            out = out + self.residual_proj(x)
        out = out.view(-1, self.n_tokens, self.hidden_dim)
        return out
