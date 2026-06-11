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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads,           self.head_dim).transpose(1, 2)
        key_states   = key_states  .view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_values is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. "
                    f"If you are using {self.__class__.__name__} for auto-regressive "
                    f"decoding with k/v caching, please make sure to initialize the "
                    f"attention class with a layer index."
                )
            kv_seq_len += past_key_values.get_seq_length(self.layer_idx)

        # ── Apply RoPE only to queries using the shared position_embeddings ──
        cos, sin = position_embeddings
        query_states = apply_single_rotary_pos_emb(query_states, cos, sin)

        # ── Update cache with RAW (un-rotated) keys ──
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            full_position_ids = torch.arange(
                0, past_key_values.get_seq_length(self.layer_idx),
                dtype=torch.long, device=query_states.device
            ).unsqueeze(0)
        else:
            full_position_ids = position_ids

        # ── Apply RoPE to ALL keys using their actual global positions ──
        cos_keys, sin_keys = self.rotary_emb(hidden_states, full_position_ids)
        key_states = apply_single_rotary_pos_emb(key_states, cos_keys, sin_keys)

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

        return attn_output, attn_weights


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
