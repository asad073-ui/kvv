from typing import Optional, Tuple

import torch
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


def apply_single_rotary_pos_emb(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (t * cos) + (rotate_half(t) * sin)


class Qwen2ModifiedAttention(Qwen2Attention):
    

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # 4.51.3's Qwen2Attention.__init__ does not set these; we need them.
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        # Separate RotaryEmbedding instance for re-applying RoPE to the full key
        # sequence.  Config-driven, so it matches the model's rope settings.
        self.rotary_emb_full = Qwen2RotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,   # singular — matches 4.51.3
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,                                  # absorbs position_ids/use_cache
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  # 2-tuple — matches 4.51.3
        output_attentions = bool(kwargs.get("output_attentions", False))

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads,           self.head_dim).transpose(1, 2)
        key_states   = key_states  .view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings

        query_states = apply_single_rotary_pos_emb(query_states, cos, sin)

        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        total_kv_len = key_states.shape[-2]
        full_position_ids = torch.arange(
            0, total_kv_len, dtype=torch.long, device=hidden_states.device
        ).unsqueeze(0)
        cos_k, sin_k = self.rotary_emb_full(hidden_states, full_position_ids)
        key_states = apply_single_rotary_pos_emb(key_states, cos_k, sin_k)

        # GQA expansion 
        key_states_exp   = repeat_kv(key_states,   self.num_key_value_groups)
        value_states_exp = repeat_kv(value_states, self.num_key_value_groups)

        #  Eager scaled dot-product attention 
        attn_weights = torch.matmul(query_states, key_states_exp.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states_exp.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output  = torch.matmul(attn_weights, value_states_exp)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, (attn_weights if output_attentions else None)


class Qwen2ModifiedDecoderLayer(Qwen2DecoderLayer):
    """Replaces only self_attn with the modified attention; forward is inherited."""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen2ModifiedAttention(config, layer_idx)


class Qwen2ModifiedModel(Qwen2Model):
    """Replaces decoder layers with modified versions; everything else inherited."""

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2ModifiedDecoderLayer(config, layer_idx)
             for layer_idx in range(config.num_hidden_layers)]
        )


class Qwen2ModifiedForCausalLM(Qwen2ForCausalLM):
    """Entry point used by chunk_cache.py and evaluate.py."""

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.model = Qwen2ModifiedModel(config)
