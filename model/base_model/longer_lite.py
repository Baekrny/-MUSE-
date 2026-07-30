import math

import torch
import torch.nn.functional as F
from torch import nn


class GroupPoolTA(nn.Module):
    def __init__(
        self, input_dim, model_dim, max_len, group_size, residual_init=0.0
    ):
        super().__init__()
        if (
            isinstance(residual_init, bool)
            or not isinstance(residual_init, (int, float))
            or not math.isfinite(residual_init)
        ):
            raise ValueError("residual_init must be a finite number")
        self.token_projection = nn.Linear(input_dim, model_dim)
        self.position_embedding = nn.Embedding(max_len, model_dim)
        self.target_projection = nn.Linear(input_dim, model_dim)
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.group_size = group_size
        self.residual_init = float(residual_init)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.token_projection.weight)
        nn.init.zeros_(self.token_projection.bias)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.target_projection.weight)
        nn.init.zeros_(self.target_projection.bias)
        nn.init.constant_(self.residual_scale, self.residual_init)

    def _project_groups(self, history, mask):
        batch_size, seq_len, _ = history.shape
        padded_len = math.ceil(seq_len / self.group_size) * self.group_size
        if padded_len > self.position_embedding.num_embeddings:
            raise ValueError(
                f"padded sequence length {padded_len} exceeds max_len "
                f"{self.position_embedding.num_embeddings}"
            )

        left_pad = padded_len - seq_len
        history = F.pad(history, (0, 0, left_pad, 0))
        mask = F.pad(mask.to(dtype=torch.bool), (left_pad, 0), value=False)

        positions = torch.arange(
            self.position_embedding.num_embeddings - padded_len,
            self.position_embedding.num_embeddings,
            device=history.device,
        )
        projected = self.token_projection(history)
        projected = projected + self.position_embedding(positions).unsqueeze(0)

        group_count = padded_len // self.group_size
        projected = projected.reshape(
            batch_size, group_count, self.group_size, -1
        )
        grouped_mask = mask.reshape(batch_size, group_count, self.group_size)
        return projected, grouped_mask

    def merge(self, history, mask):
        projected, grouped_mask = self._project_groups(history, mask)
        weights = grouped_mask.unsqueeze(-1).to(dtype=projected.dtype)
        counts = weights.sum(dim=2)
        merged = (projected * weights).sum(dim=2) / counts.clamp_min(1.0)
        group_valid = grouped_mask.any(dim=2)
        merged = merged.masked_fill(~group_valid.unsqueeze(-1), 0.0)
        return merged, group_valid

    def forward(self, target, history, mask):
        merged, group_valid = self.merge(history, mask)
        if target.dim() == 3:
            target = target.squeeze(1)
        query = self.target_projection(target)
        logits = torch.bmm(merged, query.unsqueeze(-1)).squeeze(-1)
        logits = logits / math.sqrt(merged.shape[-1])
        logits = logits.masked_fill(~group_valid, float("-inf"))

        has_history = group_valid.any(dim=1, keepdim=True)
        safe_logits = torch.where(has_history, logits, torch.zeros_like(logits))
        weights = torch.softmax(safe_logits, dim=1)
        weights = weights.masked_fill(~group_valid, 0.0)
        interest = torch.bmm(weights.unsqueeze(1), merged).squeeze(1)
        return interest.masked_fill(~has_history, 0.0)

    def combine_interest(self, base_interest, full_history_interest):
        return base_interest + self.residual_scale * full_history_interest


class InnerTransTA(GroupPoolTA):
    def __init__(
        self,
        input_dim,
        model_dim,
        max_len,
        group_size,
        num_heads,
        transform_chunk_size=50000,
        residual_init=0.0,
    ):
        super().__init__(
            input_dim,
            model_dim,
            max_len,
            group_size,
            residual_init=residual_init,
        )
        if (
            not isinstance(transform_chunk_size, int)
            or isinstance(transform_chunk_size, bool)
            or transform_chunk_size <= 0
        ):
            raise ValueError("transform_chunk_size must be a positive integer")
        self.transform_chunk_size = transform_chunk_size
        self.inner_transformer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=2 * model_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )

    def merge(self, history, mask):
        projected, grouped_mask = self._project_groups(history, mask)
        batch_size, group_count, group_size, model_dim = projected.shape
        grouped_tokens = projected.reshape(
            batch_size * group_count, group_size, model_dim
        )
        flat_mask = grouped_mask.reshape(batch_size * group_count, group_size)

        all_padding = ~flat_mask.any(dim=1)
        safe_tokens = grouped_tokens.clone()
        safe_padding_mask = ~flat_mask.clone()
        safe_tokens[all_padding] = 0.0
        safe_padding_mask[all_padding, 0] = False
        transformed_chunks = []
        for start in range(0, safe_tokens.shape[0], self.transform_chunk_size):
            stop = start + self.transform_chunk_size
            transformed_chunks.append(
                self.inner_transformer(
                    safe_tokens[start:stop],
                    src_key_padding_mask=safe_padding_mask[start:stop],
                )
            )
        transformed = torch.cat(transformed_chunks, dim=0)

        weights = flat_mask.unsqueeze(-1).to(dtype=transformed.dtype)
        counts = weights.sum(dim=1)
        merged = (transformed * weights).sum(dim=1) / counts.clamp_min(1.0)
        merged = merged.masked_fill(all_padding.unsqueeze(-1), 0.0)
        return (
            merged.reshape(batch_size, group_count, model_dim),
            grouped_mask.any(dim=2),
        )


class GlobalTokenLongerLite(InnerTransTA):
    def __init__(
        self,
        input_dim,
        user_dim,
        model_dim,
        max_len,
        group_size,
        recent_queries,
        num_heads,
        transform_chunk_size=50000,
        residual_init=0.0,
    ):
        super().__init__(
            input_dim=input_dim,
            model_dim=model_dim,
            max_len=max_len,
            group_size=group_size,
            num_heads=num_heads,
            transform_chunk_size=transform_chunk_size,
            residual_init=residual_init,
        )
        group_count = math.ceil(max_len / group_size)
        if (
            not isinstance(recent_queries, int)
            or isinstance(recent_queries, bool)
            or recent_queries < 0
            or recent_queries > group_count
        ):
            raise ValueError(
                f"recent_queries must be an integer between 0 and {group_count}"
            )

        self.recent_queries = recent_queries
        self.target_projection_global = nn.Linear(input_dim, model_dim)
        self.user_projection = nn.Linear(user_dim, model_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, model_dim))
        self.cross_attention = nn.MultiheadAttention(
            model_dim, num_heads, batch_first=True
        )
        self.query_transformer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=2 * model_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, target, user, history, mask):
        merged, group_valid = self.merge(history, mask)
        if target.dim() == 3:
            target = target.squeeze(1)
        if user.dim() == 3:
            user = user.squeeze(1)

        batch_size = merged.shape[0]
        target_query = self.target_projection_global(target).unsqueeze(1)
        user_query = self.user_projection(user).unsqueeze(1)
        cls_query = self.cls_token.expand(batch_size, -1, -1)

        if self.recent_queries:
            recent = merged[:, -self.recent_queries :]
            recent_valid = group_valid[:, -self.recent_queries :]
        else:
            recent = merged[:, :0]
            recent_valid = group_valid[:, :0]
        queries = torch.cat([target_query, user_query, cls_query, recent], dim=1)
        query_valid = torch.cat(
            [
                torch.ones(
                    batch_size, 3, dtype=torch.bool, device=group_valid.device
                ),
                recent_valid,
            ],
            dim=1,
        )

        all_padding = ~group_valid.any(dim=1)
        safe_merged = merged.clone()
        safe_group_valid = group_valid.clone()
        safe_merged[all_padding] = 0.0
        safe_group_valid[all_padding, 0] = True
        query_outputs, _ = self.cross_attention(
            queries,
            safe_merged,
            safe_merged,
            key_padding_mask=~safe_group_valid,
            need_weights=False,
        )
        query_outputs = self.query_transformer(
            query_outputs,
            src_key_padding_mask=~query_valid,
        )
        interest = query_outputs[:, 2]
        return interest.masked_fill(all_padding.unsqueeze(-1), 0.0)
