"""
ICL Transformer v7 "kitchen sink" — I-block specialist built on top of Exp B
(perprotattn), the current SOTA on I block (test AUROC 0.7342). Adds stacked
TabPFN-inspired tricks aimed at closing the gap to DNN baseline (~0.78).

Changes vs Exp B (`transformer_icl_v6_perprotattn.py`):
  1. Per-context QUANTILE normalization (uniform transform) instead of z-score.
     Matches TabPFN's `quantile_uni_coarse` preprocessing. Robust to outliers,
     avoids the mean/std collapse when the context has extreme cells.
  2. 3-layer per-protein cross-sample attention (was 1 layer). Matches TabPFN v3's
     `dist_embed_num_blocks=3`. Deeper feature-level context.
  3. Multi-CLS aggregation for output: 4 learnable CLS tokens prepended to the
     sequence, aggregate via row transformer, mean-pool as output. Replaces
     "take query position" pooling. Matches TabPFN v3's `feat_agg_num_cls_tokens=4`.

Does NOT include diseaseemb (Exp A) — it was empirically negative on I.
"""

import torch
import torch.nn as nn

from models.transformer_icl_v6_perprotattn import PerProteinAttentionBlock


class ICLTransformerV7KitchenSink(nn.Module):
    def __init__(
        self,
        protein_dim: int = 2941,
        hidden_dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        dropout: float = 0.2,
        dim_feedforward: int = 3072,
        # Per-protein attention (Exp B) — now deeper
        per_prot_hidden: int = 32,
        per_prot_heads: int = 4,
        per_prot_layers: int = 3,     # NEW: was 1 in Exp B
        # Multi-CLS aggregation
        n_cls_tokens: int = 4,        # NEW: was 0 (used query position)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_cls_tokens = n_cls_tokens

        # Stack of per-protein attention blocks (3 layers)
        self.per_protein_attn_layers = nn.ModuleList([
            PerProteinAttentionBlock(local_hidden=per_prot_hidden,
                                     n_heads=per_prot_heads)
            for _ in range(per_prot_layers)
        ])

        # v6 stack
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.label_emb    = nn.Embedding(3, hidden_dim)
        self.query_pos    = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query_pos, std=0.02)

        # Multi-CLS tokens — learnable summary slots
        self.cls_tokens = nn.Parameter(torch.zeros(1, n_cls_tokens, hidden_dim))
        nn.init.normal_(self.cls_tokens, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.protein_proj.weight)
        nn.init.zeros_(self.protein_proj.bias)
        nn.init.normal_(self.label_emb.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _quantile_normalize(ctx: torch.Tensor, qry: torch.Tensor):
        """Per-context QUANTILE (uniform) transform. Robust replacement for z-score.

        ctx: (B, K, D)   qry: (B, D)  → returns (B, K, D), (B, 1, D) in [-0.5, 0.5]

        For each (b, d) independently:
          * rank each context value → uniform in (0, 1)
          * find query's insertion rank in sorted context → uniform
          * shift to [-0.5, 0.5]
        """
        B, K, D = ctx.shape
        # Ranks of ctx values within their (b, d) column (0..K-1)
        # argsort(argsort(x)) gives the rank of each element
        ranks = ctx.argsort(dim=1).argsort(dim=1).float()          # (B, K, D)
        ctx_uni = (ranks + 0.5) / K - 0.5                          # (B, K, D)

        # Query rank: how many ctx values are strictly less than qry
        ctx_sorted, _ = ctx.sort(dim=1)                            # (B, K, D)
        qry_expanded = qry.unsqueeze(1)                            # (B, 1, D)
        qry_ranks = (ctx_sorted < qry_expanded).float().sum(dim=1, keepdim=True)  # (B, 1, D)
        qry_uni = (qry_ranks + 0.5) / K - 0.5                      # (B, 1, D)

        return ctx_uni, qry_uni

    def forward(self, batch: dict) -> torch.Tensor:
        ctx_proteins = batch['context_proteins']
        ctx_labels   = batch['context_labels']
        qry_proteins = batch['query_proteins']

        B, K, D = ctx_proteins.shape

        # Trick 4: quantile normalization (per-context, per-protein)
        ctx_norm, qry_norm = self._quantile_normalize(ctx_proteins, qry_proteins)

        # Trick 2: 3 layers of per-protein cross-sample attention
        stacked = torch.cat([ctx_norm, qry_norm], dim=1)   # (B, K+1, D)
        for attn_layer in self.per_protein_attn_layers:
            stacked = attn_layer(stacked)
        ctx_norm, qry_norm = stacked[:, :K, :], stacked[:, K:, :]

        # Row embeddings (unchanged v6)
        ctx_emb = self.protein_proj(ctx_norm) + self.label_emb(ctx_labels.long())
        qry_emb = self.protein_proj(qry_norm) + self.label_emb(
            torch.full((B, 1), 2, dtype=torch.long, device=qry_proteins.device)
        ) + self.query_pos

        # Trick 3: prepend multi-CLS tokens
        cls = self.cls_tokens.expand(B, -1, -1)             # (B, n_cls, H)
        seq = torch.cat([cls, ctx_emb, qry_emb], dim=1)     # (B, n_cls+K+1, H)

        hidden = self.transformer(seq)                       # (B, n_cls+K+1, H)

        # Pool from CLS tokens (mean over the n_cls learned summary slots)
        cls_hidden = hidden[:, :self.n_cls_tokens, :]        # (B, n_cls, H)
        pooled = cls_hidden.mean(dim=1)                      # (B, H)

        return self.output_proj(pooled).squeeze(-1)
