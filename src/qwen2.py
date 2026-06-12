"""
qwen2.py  –  Modified Qwen2 attention for TurboRAG-style independent KV-cache stitching.

Keys are stored WITHOUT RoPE applied so that, at query time, RoPE can be applied
using the reordered global position IDs rather than the per-chunk local IDs.
This matches the TurboRAG "reordered positions" design.
"""

from typing import Optional, Tuple
import torch
import math
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2RotaryEmbedding,
    repeat_kv,
    rotate_half,
    Qwen2DecoderLayer,
    Qwen2Model,
    Qwen2ForCausalLM,
    Qwen2Config,
)
from transformers.cache_utils import Cache


def apply_single_rotary_pos_emb(inputs, cos, sin, unsqueeze_dim=1):
    """Apply RoPE to a single tensor.
    cos/sin shape: [batch, seq_len, head_dim] – already index-selected for the
    relevant position_ids.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (inputs * cos) + (rotate_half(inputs) * sin)


class Qwen2ModifiedAttention(Qwen2Attention):
    """
    Drop-in replacement for Qwen2Attention that stores raw (un-rotated) keys in
    the KV cache and applies RoPE at attention time using the *global* position IDs.
    This is required for TurboRAG's cache-stitching to work correctly across chunks.
    """

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        # Keep a local rotary_emb for re-applying RoPE to the full key sequence
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads,           self.head_dim).transpose(1, 2)
        key_states   = key_states  .view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings

        # ── Apply RoPE only to queries using the shared position_embeddings ──
        query_states = apply_single_rotary_pos_emb(query_states, cos, sin)

        # ── Determine RoPE regime based on cache state ──
        # FIX-BUGB: Three regimes for key RoPE application:
        #
        #   Regime 1 — First decode step after stitching:
        #     _rope_applied_to_prefix == False.  ALL keys in the cache are un-rotated
        #     (TurboRAG design).  Apply RoPE to the FULL key sequence (cached + new)
        #     using global position IDs, then mark the flag as True.
        #
        #   Regime 2 — Incremental decode steps:
        #     _rope_applied_to_prefix == True (or flag absent = standard model use).
        #     Cached keys already have RoPE.  Apply RoPE to NEW keys only BEFORE
        #     storing them in the cache.
        #
        #   Regime 3 — No cache (chunk build phase):
        #     past_key_values is None.  Do NOT apply RoPE to keys — they are stored
        #     un-rotated by design for TurboRAG cache stitching.

        rope_applied = getattr(past_key_values, "_rope_applied_to_prefix", True) if past_key_values is not None else True

        if past_key_values is not None and not rope_applied:
            # ── Regime 1: first decode after stitching ──
            # Store un-rotated new keys in cache first
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            # key_states is now [all_cached_unrotated | new_unrotated]

            # Apply RoPE to the FULL key sequence using global position IDs
            total_kv_len = key_states.shape[-2]
            full_position_ids = torch.arange(
                0, total_kv_len,
                dtype=torch.long, device=query_states.device
            ).unsqueeze(0)
            cos_keys, sin_keys = self.rotary_emb(hidden_states, full_position_ids)
            key_states = apply_single_rotary_pos_emb(key_states, cos_keys, sin_keys)

            # FIX-BUGF: Write rotated keys BACK into the cache.
            # Without this, subsequent Regime-2 decode steps read un-rotated
            # keys from the cache, producing garbage attention after token 1.
            past_key_values.key_cache[self.layer_idx] = key_states

            # Mark flag after ALL layers have processed on this step.
            # We set it on every layer's pass; after the forward completes for all
            # layers, the flag is True and subsequent steps use Regime 2.
            past_key_values._rope_applied_to_prefix = True

            kv_seq_len = total_kv_len

        elif past_key_values is not None:
            # ── Regime 2: incremental decode (cached keys already have RoPE) ──
            cached_len = past_key_values.get_seq_length(self.layer_idx)

            # Apply RoPE to new keys using their global positions BEFORE storing
            new_pos_ids = torch.arange(
                cached_len, cached_len + q_len,
                dtype=torch.long, device=key_states.device
            ).unsqueeze(0)
            cos_new, sin_new = self.rotary_emb(hidden_states, new_pos_ids)
            key_states = apply_single_rotary_pos_emb(key_states, cos_new, sin_new)

            # Store RoPE-applied keys in cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

            kv_seq_len = key_states.shape[-2]

        else:
            # ── Regime 3: chunk build phase — NO RoPE on keys ──
            # Keys are stored un-rotated by DynamicCache (TurboRAG design).
            # chunk_cache.py extracts them via to_legacy_cache() after the forward.
            kv_seq_len = key_states.shape[-2]

        key_states   = repeat_kv(key_states,   self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, "
                f"but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output  = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # ── FIX: return past_key_values as 3rd element ──
        # Qwen2DecoderLayer.forward unpacks all 3 values from the attention return.
        return attn_output, attn_weights, past_key_values


class Qwen2ModifiedDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen2ModifiedAttention(config, layer_idx)


class Qwen2ModifiedModel(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2ModifiedDecoderLayer(config, layer_idx)
             for layer_idx in range(config.num_hidden_layers)]
        )


class Qwen2ModifiedForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.model = Qwen2ModifiedModel(config)