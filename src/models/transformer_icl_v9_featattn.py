"""
ICL Transformer v9 = v8 + feature-level attention pooling.

Key change vs v8:
  - v7/v8 collapse the (K+1, 2941, 32) matrix back to (K+1, 2941) scalars via
    Linear(32, 1), then use Linear(2941, 768) to form the patient token.
    That downcast throws away the 31/32 dims per cell the per-protein attn
    just learned.

  - v9 keeps the matrix. It prepends N=24 learnable feature-CLS tokens (each
    32-d) and runs a small 3-layer transformer over (24 + 2941) rows.
    The 24 CLS output vectors are CONCATENATED (not mean-pooled) to form a
    24 * 32 = 768-d patient token directly. No further Linear(32, 768) is
    needed and Linear(2941, 768) is removed entirely.

Info flow: (2941, 32) matrix → feat attn → 24 CLS × 32 = 768-d token.
No 32-d bottleneck (unlike the 4-CLS + mean design which collapses to 32).

Downstream (label_emb, sample-CLS, 12-layer main transformer, readout)
unchanged.
"""
import torch
import torch.nn as nn


class PerProteinAttentionBlockV9(nn.Module):
    """Per-protein cross-sample attention — matrix in, matrix out (no downcast).

    Matches v7/v8's identity-at-init training dynamic: outer residual with
    zero-init gate (\texttt{out\_gate}) so \texttt{delta = 0} at start and
    the block is an identity map.
    """

    def __init__(self, local_hidden: int = 32, n_heads: int = 4):
        super().__init__()
        self.attn_layer = nn.TransformerEncoderLayer(
            d_model=local_hidden, nhead=n_heads,
            dim_feedforward=local_hidden * 2,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        # Zero-init gate for outer residual — matches v7/v8's out_proj trick
        self.out_gate = nn.Linear(local_hidden, local_hidden)
        nn.init.zeros_(self.out_gate.weight)
        nn.init.zeros_(self.out_gate.bias)

    def forward(self, x_BSDH: torch.Tensor) -> torch.Tensor:
        B, S, D, H = x_BSDH.shape
        # (B, S, D, H) → (B*D, S, H) so attention runs over S for each protein
        x = x_BSDH.permute(0, 2, 1, 3).reshape(B * D, S, H)
        h = self.attn_layer(x)
        delta = self.out_gate(h)                          # zero at init
        y = x + delta                                     # identity at init
        y = y.reshape(B, D, S, H).permute(0, 2, 1, 3)     # (B, S, D, H)
        return y


class FeatureAttentionPool(nn.Module):
    """Feature-level pool: (B, S, D, H) matrix per patient → (B, S, N*H) token.

    2026-09-07: replaced the CLS-attention version with a direct linear
    projection over the D=2941 protein axis. Empirically the CLS-attention pool
    could not train (val AUROC stuck at 0.60): CLS-CLS self-attention dominated
    → 24 CLS outputs were near-identical across patients → main transformer
    saw indistinguishable patient tokens → loss did not decrease.

    New design: learned `Linear(D, n_cls)` mixes the 2941 protein rows into
    `n_cls` latent rows, each still H-dim. Concat → hidden_dim patient token.
    Xavier init immediately produces patient-specific outputs since each
    patient has different quantile-normalized protein values.

    Params: `D × n_cls` (2941 × 24 = 70K here). Same output shape as before,
    downstream unchanged. `n_heads` / `n_layers` args kept for backward
    compat with the config yaml but no longer used.
    """

    def __init__(self, local_hidden: int = 32, n_cls: int = 24,
                 n_heads: int = 4, n_layers: int = 3,
                 protein_dim: int = 2941):
        super().__init__()
        self.n_cls = n_cls
        self.local_hidden = local_hidden
        # Linear over D axis: mixes 2941 protein rows into n_cls latent rows.
        self.protein_mixer = nn.Linear(protein_dim, n_cls)

    def forward(self, x_BSDH: torch.Tensor) -> torch.Tensor:
        B, S, D, H = x_BSDH.shape
        # (B, S, D, H) → (B*S, H, D) so Linear(D, n_cls) mixes the D axis
        x = x_BSDH.reshape(B * S, D, H).transpose(1, 2)   # (B*S, H, D)
        mixed = self.protein_mixer(x)                      # (B*S, H, n_cls)
        pooled = mixed.transpose(1, 2).reshape(B * S, self.n_cls * H)  # (B*S, N*H)
        return pooled.reshape(B, S, self.n_cls * H)        # (B, S, N*H)


class ICLTransformerV9FeatAttn(nn.Module):
    def __init__(
        self,
        protein_dim: int = 2941,
        hidden_dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        dropout: float = 0.2,
        dim_feedforward: int = 3072,
        per_prot_hidden: int = 32,
        per_prot_heads: int = 4,
        per_prot_layers: int = 3,
        n_cls_tokens: int = 4,       # sample-level CLS (v7 style)
        n_cls_feat: int = 24,        # feature-level CLS; must satisfy n_cls_feat * per_prot_hidden == hidden_dim
        n_feat_layers: int = 3,      # feature transformer depth
        feat_heads: int = 4,
    ):
        super().__init__()
        assert n_cls_feat * per_prot_hidden == hidden_dim, (
            f"n_cls_feat ({n_cls_feat}) * per_prot_hidden ({per_prot_hidden}) "
            f"must equal hidden_dim ({hidden_dim}) for concat to yield the right size"
        )
        self.hidden_dim = hidden_dim
        self.n_cls_tokens = n_cls_tokens
        self.per_prot_hidden = per_prot_hidden

        # v8 additions kept
        self.protein_id_emb = nn.Embedding(protein_dim, per_prot_hidden)
        nn.init.normal_(self.protein_id_emb.weight, std=0.02)

        # Input lift: scalar → per_prot_hidden
        self.in_proj = nn.Linear(1, per_prot_hidden)

        # Per-protein attn WITHOUT downcast (matrix in, matrix out)
        self.per_protein_attn_layers = nn.ModuleList([
            PerProteinAttentionBlockV9(local_hidden=per_prot_hidden, n_heads=per_prot_heads)
            for _ in range(per_prot_layers)
        ])

        # v9 NEW: feature pool produces (B, K+1, hidden_dim) directly.
        # 2026-09-07: CLS-attention pool did not train; replaced with a direct
        # Linear(D, n_cls) mixer inside FeatureAttentionPool. n_heads/n_layers
        # are kept for compat but unused.
        self.feat_pool = FeatureAttentionPool(
            local_hidden=per_prot_hidden,
            n_cls=n_cls_feat,
            n_heads=feat_heads,
            n_layers=n_feat_layers,
            protein_dim=protein_dim,
        )

        # No more Linear(2941, 768) — feat_pool directly outputs hidden_dim per patient

        # Same as v7/v8 downstream
        self.label_emb = nn.Embedding(3, hidden_dim)
        self.query_pos = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query_pos, std=0.02)

        self.cls_tokens = nn.Parameter(torch.zeros(1, n_cls_tokens, hidden_dim))
        nn.init.normal_(self.cls_tokens, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, 1)

        nn.init.normal_(self.label_emb.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _quantile_normalize(ctx: torch.Tensor, qry: torch.Tensor):
        B, K, D = ctx.shape
        ranks = ctx.argsort(dim=1).argsort(dim=1).float()
        ctx_uni = (ranks + 0.5) / K - 0.5
        ctx_sorted, _ = ctx.sort(dim=1)
        qry_expanded = qry.unsqueeze(1)
        qry_ranks = (ctx_sorted < qry_expanded).float().sum(dim=1, keepdim=True)
        qry_uni = (qry_ranks + 0.5) / K - 0.5
        return ctx_uni, qry_uni

    def forward(self, batch: dict) -> torch.Tensor:
        ctx_proteins = batch['context_proteins']
        ctx_labels   = batch['context_labels']
        qry_proteins = batch['query_proteins']

        B, K, D = ctx_proteins.shape

        # v7 Trick 4: quantile normalization
        ctx_norm, qry_norm = self._quantile_normalize(ctx_proteins, qry_proteins)

        # Stack ctx + qry as single (B, K+1, D) scalar tensor
        stacked = torch.cat([ctx_norm, qry_norm], dim=1)   # (B, K+1, D)

        # Lift scalar → 32-d + add protein_id_emb (v8 style)
        x = stacked.unsqueeze(-1)                           # (B, K+1, D, 1)
        x = self.in_proj(x)                                 # (B, K+1, D, H_local)
        protein_ids = torch.arange(D, device=stacked.device)
        id_emb = self.protein_id_emb(protein_ids)           # (D, H_local)
        x = x + id_emb.view(1, 1, D, -1)                    # broadcast

        # v7 Tricks 1+2: 3 layers of per-protein cross-sample attention (matrix, no downcast)
        for attn_layer in self.per_protein_attn_layers:
            x = attn_layer(x)                               # (B, K+1, D, H_local)

        # v9 NEW: feature attention pool → (B, K+1, hidden_dim)
        patient_tokens = self.feat_pool(x)                  # (B, K+1, 768)

        # Split ctx / qry, add label embeddings (same as v7/v8)
        ctx_tok = patient_tokens[:, :K, :]
        qry_tok = patient_tokens[:, K:, :]

        ctx_emb = ctx_tok + self.label_emb(ctx_labels.long())
        qry_emb = qry_tok + self.label_emb(
            torch.full((B, 1), 2, dtype=torch.long, device=qry_proteins.device)
        ) + self.query_pos

        # v7 Trick 3: multi-CLS + main transformer
        cls = self.cls_tokens.expand(B, -1, -1)
        seq = torch.cat([cls, ctx_emb, qry_emb], dim=1)
        hidden = self.transformer(seq)

        cls_hidden = hidden[:, :self.n_cls_tokens, :]
        pooled = cls_hidden.mean(dim=1)

        return self.output_proj(pooled).squeeze(-1)
