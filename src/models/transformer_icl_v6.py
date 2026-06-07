"""
ICL Transformer v6: set-based, bidirectional, per-context normalization.

Key changes vs v5 (InContextTransformerGPT):
  1. Bidirectional attention (nn.TransformerEncoder, no causal mask)
  2. Context tokens have no positional encoding (set-based; order doesn't matter)
  3. label_emb expanded to 3 slots: 0=negative, 1=positive, 2=unknown (query)
     — fixes the "query biased toward negative" confound
  4. Per-context normalization: each context window normalized by its own mean/std
     — normalizes over the K context examples per feature before projection
  5. No GPT-2 wpe (not using GPT-2 at all)
  6. Query is distinguishable via (a) the unknown label and (b) a learned query_pos token
  7. Train and eval context size can differ without PE extrapolation issues
"""

import torch
import torch.nn as nn
import math


class ICLTransformerV6(nn.Module):
    def __init__(
        self,
        protein_dim: int = 2941,
        hidden_dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        dropout: float = 0.2,
        dim_feedforward: int = 3072,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project protein features to hidden dim
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)

        # 3-slot label embedding: 0=negative, 1=positive, 2=unknown (query)
        self.label_emb = nn.Embedding(3, hidden_dim)

        # Learned query position marker (distinguishes query from context tokens)
        self.query_pos = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query_pos, std=0.02)

        # Bidirectional Transformer (no causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LayerNorm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Binary classification head
        self.output_proj = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.protein_proj.weight)
        nn.init.zeros_(self.protein_proj.bias)
        nn.init.normal_(self.label_emb.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        batch keys:
            context_proteins  (B, K, protein_dim)
            context_labels    (B, K)  int64, values in {0, 1}
            query_proteins    (B, protein_dim)
            target_label      (B,)    unused in forward

        Returns:
            logits (B,)
        """
        ctx_proteins = batch['context_proteins']   # (B, K, D)
        ctx_labels   = batch['context_labels']     # (B, K)
        qry_proteins = batch['query_proteins']     # (B, D)

        B, K, _ = ctx_proteins.shape

        # --- Per-context normalization ---
        # Normalize each context window by its own statistics (per feature)
        ctx_mean = ctx_proteins.mean(dim=1, keepdim=True)                  # (B, 1, D)
        ctx_std  = ctx_proteins.std(dim=1, keepdim=True).clamp(min=1e-6)   # (B, 1, D)
        ctx_norm = (ctx_proteins - ctx_mean) / ctx_std                     # (B, K, D)
        # Apply same normalization to query (using context's statistics)
        qry_norm = (qry_proteins.unsqueeze(1) - ctx_mean) / ctx_std        # (B, 1, D)

        # --- Embed context tokens (no positional encoding — set-based) ---
        ctx_emb = self.protein_proj(ctx_norm)                              # (B, K, H)
        ctx_emb = ctx_emb + self.label_emb(ctx_labels.long())             # (B, K, H)

        # --- Embed query token with "unknown" label (index 2) + query position marker ---
        qry_emb = self.protein_proj(qry_norm)                             # (B, 1, H)
        qry_emb = qry_emb + self.label_emb(
            torch.full((B, 1), 2, dtype=torch.long, device=qry_proteins.device)
        )                                                                   # (B, 1, H)
        qry_emb = qry_emb + self.query_pos                                # (B, 1, H)

        # --- Concatenate: context tokens then query token ---
        seq = torch.cat([ctx_emb, qry_emb], dim=1)                        # (B, K+1, H)

        # --- Bidirectional Transformer (all tokens attend to all tokens) ---
        hidden = self.transformer(seq)                                     # (B, K+1, H)

        # --- Predict from query position (last token) ---
        query_hidden = hidden[:, -1, :]                                    # (B, H)
        logits = self.output_proj(query_hidden).squeeze(-1)                # (B,)

        return logits
