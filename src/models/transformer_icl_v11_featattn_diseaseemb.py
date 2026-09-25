"""
ICL Transformer v11 = v9 (feat-attn pool, no downcast) + Exp-A diseaseemb.

Motivation:
  v9's feature-attention pool (Linear(D, n_cls) + concat) gives the strongest
  I/C results. v10's diseaseemb (ICD-10 prefix embedding) gives the strongest
  G lift over v7/v8. v11 stacks the two so a single architecture can plausibly
  win on all three blocks (I / C / G) without per-block model swaps.

Forward changes vs v9:
  + After `feat_pool` outputs patient tokens and after label_emb addition,
    add a per-query ICD-10 prefix embedding (same lookup as v10) to every
    context and query token.
  + Batch must include `query_disease_codes` (already provided by
    collate_fn_no_id).

Constructor takes prefix_vocab + code_to_prefix_ids like v10.
"""
import torch
import torch.nn as nn
from typing import Dict, List

from models.transformer_icl_v9_featattn import (
    PerProteinAttentionBlockV9, FeatureAttentionPool,
)


class ICLTransformerV11FeatAttnDiseaseEmb(nn.Module):
    def __init__(
        self,
        prefix_vocab: Dict[str, int],
        code_to_prefix_ids: Dict[str, List[int]],
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
        n_cls_feat: int = 24,
        n_feat_layers: int = 3,
        feat_heads: int = 4,
    ):
        super().__init__()
        assert n_cls_feat * per_prot_hidden == hidden_dim, (
            f"n_cls_feat ({n_cls_feat}) * per_prot_hidden ({per_prot_hidden}) "
            f"must equal hidden_dim ({hidden_dim})"
        )
        self.hidden_dim = hidden_dim
        self.n_cls_tokens = n_cls_tokens
        self.per_prot_hidden = per_prot_hidden
        self.prefix_vocab = prefix_vocab
        self.code_to_prefix_ids = code_to_prefix_ids

        # ---- Prefix-embedding buffers (Exp A / v10) ----
        max_len = max(len(v) for v in code_to_prefix_ids.values())
        code_ids = sorted(code_to_prefix_ids.keys())
        self.code_to_row = {c: i for i, c in enumerate(code_ids)}
        prefix_idx = torch.zeros(len(code_ids), max_len, dtype=torch.long)
        prefix_mask = torch.zeros(len(code_ids), max_len, dtype=torch.float32)
        for c, r in self.code_to_row.items():
            ids = code_to_prefix_ids[c]
            prefix_idx[r, :len(ids)] = torch.tensor(ids)
            prefix_mask[r, :len(ids)] = 1.0
        self.register_buffer('prefix_idx',  prefix_idx)
        self.register_buffer('prefix_mask', prefix_mask)
        self.prefix_emb = nn.Embedding(len(prefix_vocab), hidden_dim)

        # ---- v8/v9 stack ----
        self.protein_id_emb = nn.Embedding(protein_dim, per_prot_hidden)
        nn.init.normal_(self.protein_id_emb.weight, std=0.02)

        self.in_proj = nn.Linear(1, per_prot_hidden)

        self.per_protein_attn_layers = nn.ModuleList([
            PerProteinAttentionBlockV9(local_hidden=per_prot_hidden,
                                       n_heads=per_prot_heads)
            for _ in range(per_prot_layers)
        ])

        self.feat_pool = FeatureAttentionPool(
            local_hidden=per_prot_hidden,
            n_cls=n_cls_feat,
            n_heads=feat_heads,
            n_layers=n_feat_layers,
            protein_dim=protein_dim,
        )

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

        nn.init.normal_(self.label_emb.weight,  std=0.02)
        nn.init.normal_(self.prefix_emb.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def _lookup_disease_emb(self, codes: List[str], device) -> torch.Tensor:
        rows = torch.tensor([self.code_to_row[c] for c in codes],
                            dtype=torch.long, device=device)
        idx  = self.prefix_idx[rows]
        mask = self.prefix_mask[rows]
        emb  = self.prefix_emb(idx)
        return (emb * mask.unsqueeze(-1)).sum(dim=1)   # (B, hidden_dim)

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
        codes        = batch['query_disease_codes']

        B, K, D = ctx_proteins.shape

        # v7 quantile norm
        ctx_norm, qry_norm = self._quantile_normalize(ctx_proteins, qry_proteins)

        # Lift + protein_id_emb (v8/v9)
        stacked = torch.cat([ctx_norm, qry_norm], dim=1)     # (B, K+1, D)
        x = stacked.unsqueeze(-1)                            # (B, K+1, D, 1)
        x = self.in_proj(x)                                  # (B, K+1, D, H_local)
        protein_ids = torch.arange(D, device=stacked.device)
        id_emb = self.protein_id_emb(protein_ids)
        x = x + id_emb.view(1, 1, D, -1)

        # 3-layer per-protein attn (matrix in / matrix out)
        for attn_layer in self.per_protein_attn_layers:
            x = attn_layer(x)

        # v9 feature attention pool → (B, K+1, hidden_dim)
        patient_tokens = self.feat_pool(x)

        ctx_tok = patient_tokens[:, :K, :]
        qry_tok = patient_tokens[:, K:, :]

        ctx_emb = ctx_tok + self.label_emb(ctx_labels.long())
        qry_emb = qry_tok + self.label_emb(
            torch.full((B, 1), 2, dtype=torch.long, device=qry_proteins.device)
        ) + self.query_pos

        # v11 NEW: Exp-A disease embedding injection (like v10)
        d_emb = self._lookup_disease_emb(codes, qry_proteins.device)   # (B, hidden_dim)
        ctx_emb = ctx_emb + d_emb.unsqueeze(1)
        qry_emb = qry_emb + d_emb.unsqueeze(1)

        cls = self.cls_tokens.expand(B, -1, -1)
        seq = torch.cat([cls, ctx_emb, qry_emb], dim=1)
        hidden = self.transformer(seq)

        cls_hidden = hidden[:, :self.n_cls_tokens, :]
        pooled = cls_hidden.mean(dim=1)
        return self.output_proj(pooled).squeeze(-1)
