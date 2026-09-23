"""
ICL Transformer v8 = v7 kitchen sink + protein identity embedding.

Only change vs v7: at each per-protein attention layer, after the shared
Linear(1, H_local) lift, add a per-protein learnable identity vector so
each cell knows WHICH protein it belongs to (not just "some protein with
this quantile-normalized value").

Rationale: v7's Linear(1, 32) is shared across all 2941 proteins. Two
proteins with the same normalized value produce identical H_local vectors,
which loses biological identity. Adding a per-protein 32-d embedding lets
the attention layer condition its cross-sample computation on protein ID.

Downstream (main transformer) unchanged — this is a strictly local change
inside the per-protein attention block.
"""
import torch
import torch.nn as nn


class PerProteinAttentionBlockWithID(nn.Module):
    """Per-protein cross-sample attention with an external protein-ID embedding.

    Takes the same (B, S, D) scalar input as v6/v7's PerProteinAttentionBlock,
    plus an extra (D, H_local) tensor of protein-identity vectors that get
    added after the shared Linear(1, H_local) lift.
    """

    def __init__(self, local_hidden: int = 32, n_heads: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(1, local_hidden)
        self.attn_layer = nn.TransformerEncoderLayer(
            d_model=local_hidden, nhead=n_heads,
            dim_feedforward=local_hidden * 2,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.out_proj = nn.Linear(local_hidden, 1)
        nn.init.zeros_(self.out_proj.weight)   # residual: start as identity
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_BSD: torch.Tensor, protein_id_emb_DH: torch.Tensor) -> torch.Tensor:
        """x_BSD: (B, S=K+1, D=2941), protein_id_emb_DH: (D, H_local)."""
        B, S, D = x_BSD.shape
        x = x_BSD.permute(0, 2, 1).reshape(B * D, S, 1)   # (B*D, S, 1)
        h = self.in_proj(x)                                # (B*D, S, H_local)
        # Broadcast protein ID: for each of the B batches, D proteins each get their vector
        # (D, H_local) -> (B*D, 1, H_local) so it adds across the S dimension
        id_h = protein_id_emb_DH.unsqueeze(0).expand(B, D, -1).reshape(B * D, 1, -1)
        h = h + id_h                                       # (B*D, S, H_local)
        h = self.attn_layer(h)                             # (B*D, S, H_local)
        delta = self.out_proj(h)                           # (B*D, S, 1)  starts at 0
        y = x + delta                                      # residual
        y = y.reshape(B, D, S, 1).squeeze(-1).permute(0, 2, 1)   # (B, S, D)
        return y


class ICLTransformerV8ProtID(nn.Module):
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
        n_cls_tokens: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_cls_tokens = n_cls_tokens

        # NEW in v8: shared protein-identity embedding used by every per-protein attn layer
        self.protein_id_emb = nn.Embedding(protein_dim, per_prot_hidden)
        nn.init.normal_(self.protein_id_emb.weight, std=0.02)

        self.per_protein_attn_layers = nn.ModuleList([
            PerProteinAttentionBlockWithID(local_hidden=per_prot_hidden,
                                            n_heads=per_prot_heads)
            for _ in range(per_prot_layers)
        ])

        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.label_emb    = nn.Embedding(3, hidden_dim)
        self.query_pos    = nn.Parameter(torch.zeros(1, 1, hidden_dim))
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

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.protein_proj.weight)
        nn.init.zeros_(self.protein_proj.bias)
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

        # v7 Tricks 1+2 + v8 protein ID: 3 layers of per-protein attn conditioned on protein identity
        stacked = torch.cat([ctx_norm, qry_norm], dim=1)   # (B, K+1, D)
        protein_ids = torch.arange(D, device=ctx_proteins.device)
        protein_id_emb = self.protein_id_emb(protein_ids)   # (D, H_local)
        for attn_layer in self.per_protein_attn_layers:
            stacked = attn_layer(stacked, protein_id_emb)
        ctx_norm, qry_norm = stacked[:, :K, :], stacked[:, K:, :]

        ctx_emb = self.protein_proj(ctx_norm) + self.label_emb(ctx_labels.long())
        qry_emb = self.protein_proj(qry_norm) + self.label_emb(
            torch.full((B, 1), 2, dtype=torch.long, device=qry_proteins.device)
        ) + self.query_pos

        # v7 Trick 3: multi-CLS
        cls = self.cls_tokens.expand(B, -1, -1)
        seq = torch.cat([cls, ctx_emb, qry_emb], dim=1)

        hidden = self.transformer(seq)

        cls_hidden = hidden[:, :self.n_cls_tokens, :]
        pooled = cls_hidden.mean(dim=1)

        return self.output_proj(pooled).squeeze(-1)
