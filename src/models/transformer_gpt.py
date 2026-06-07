"""
In-Context Transformer with GPT-2 backbone for disease prediction.
Reconstructed from checkpoint state dict (original source lost during env corruption).
"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class InContextTransformerGPT(nn.Module):
    def __init__(
        self,
        protein_dim: int = 2941,
        hidden_dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        dropout: float = 0.2,
        temperature: float = 1.0,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Project raw protein features to hidden dim
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)

        # Label embedding: 0=negative, 1=positive (query gets label 0 as placeholder)
        self.label_emb = nn.Embedding(2, hidden_dim)

        # Custom positional encoding (buffer, not trained)
        self.register_buffer(
            'pos_encoding',
            self._sinusoidal_pe(max_seq_len, hidden_dim).unsqueeze(0)  # (1, max_seq_len, hidden_dim)
        )

        # GPT-2 backbone (vocab_size=1 since we always pass inputs_embeds)
        gpt_config = GPT2Config(
            vocab_size=1,
            n_positions=max_seq_len,
            n_embd=hidden_dim,
            n_layer=n_layers,
            n_head=n_heads,
            attn_pdrop=dropout,
            resid_pdrop=dropout,
            embd_pdrop=0.0,
        )
        self.gpt = GPT2Model(gpt_config)

        # Binary classification head
        self.output_proj = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        """Sinusoidal positional encoding (max_len, d_model)."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, batch: dict) -> torch.Tensor:
        """
        batch keys:
            context_proteins  (B, K, protein_dim)
            context_labels    (B, K)  int64
            query_proteins    (B, protein_dim)
            target_label      (B,)    (unused in forward; used for loss)

        Returns:
            logits (B,)
        """
        ctx_proteins = batch['context_proteins']   # (B, K, D)
        ctx_labels   = batch['context_labels']     # (B, K)
        qry_proteins = batch['query_proteins']     # (B, D)

        B, K, _ = ctx_proteins.shape

        # --- Embed context tokens ---
        ctx_emb  = self.protein_proj(ctx_proteins)          # (B, K, H)
        ctx_emb += self.label_emb(ctx_labels.long())        # (B, K, H)

        # --- Embed query token (label unknown → use index 0) ---
        qry_emb  = self.protein_proj(qry_proteins)          # (B, H)
        qry_emb += self.label_emb(
            torch.zeros(B, dtype=torch.long, device=qry_proteins.device)
        )                                                    # (B, H)
        qry_emb  = qry_emb.unsqueeze(1)                    # (B, 1, H)

        # --- Concatenate: context tokens then query token ---
        seq = torch.cat([ctx_emb, qry_emb], dim=1)         # (B, K+1, H)

        # --- Add custom positional encoding ---
        seq_len = seq.size(1)
        seq = seq + self.pos_encoding[:, :seq_len, :]

        # --- GPT-2 forward (internally adds wpe as well) ---
        out = self.gpt(inputs_embeds=seq)
        hidden = out.last_hidden_state                      # (B, K+1, H)

        # --- Predict from last position (query token) ---
        query_hidden = hidden[:, -1, :]                     # (B, H)
        logits = self.output_proj(query_hidden / self.temperature).squeeze(-1)  # (B,)

        return logits
