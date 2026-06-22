
from __future__ import annotations
import torch
from typing import Tuple, Dict, Any, List


# Group size for group-wise INT4 quantization.  head_dim (128 on Qwen2.5) is
# split into contiguous groups of this many channels, each with its own scale.
# 64 divides 128 cleanly (2 scales/vector), localises outlier damage to a
# 64-wide group, and keeps the storage overhead over INT8 negligible.  Do not
# drop below 32 or the scale overhead erodes the win over INT8.
INT4_GROUP_SIZE = 64


# Low-level quantisation helpers


def quantize_int8(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    
    orig_dtype = tensor.dtype
    t = tensor.float()  # work in fp32

    t_min = t.min(dim=-1, keepdim=True).values
    t_max = t.max(dim=-1, keepdim=True).values

    scale      = (t_max - t_min) / 255.0
    scale      = scale.clamp(min=1e-8)
    zero_point = (-t_min / scale).round().clamp(0, 255).to(torch.int32)

    quantized = ((t / scale) + zero_point).round().clamp(0, 255).to(torch.uint8)

    return {
        "quantized":   quantized,
        "scale":       scale.to(torch.float32),
        "zero_point":  zero_point,
        "shape":       list(tensor.shape),
        "dtype":       str(orig_dtype),
    }


def dequantize_int8(data: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Reconstruct FP32 tensor from INT8 quantized dict."""
    q   = data["quantized"].float()
    s   = data["scale"]
    zp  = data["zero_point"].float()
    out = (q - zp) * s
    # Restore original dtype
    target_dtype = _str_to_dtype(data["dtype"])
    return out.to(target_dtype)


def quantize_int4(tensor: torch.Tensor, group_size: int = INT4_GROUP_SIZE) -> Dict[str, torch.Tensor]:
    """
    Group-wise symmetric INT4 quantization along the last dimension (head_dim).

    The head_dim vector is split into contiguous groups of ``group_size``
    channels, and a separate absmax scale is fitted per group.  This localises
    the damage from a single large-magnitude channel to its own 64-wide group
    instead of letting one outlier set the scale for the entire 128-dim vector.
    Under INT4's 16 levels, a per-token (whole-vector) scale driven by an
    outlier rounds every small channel to ~0 — the failure mode that made INT4
    collapse the model.  Per-group scaling is the standard fix (the scheme
    behind KIVI / KVQuant).

    Range: [-8, 7].  Two INT4 values are packed into one uint8 byte along
    head_dim.  head_dim is zero-padded up to a multiple of ``group_size``,
    which also guarantees the even length the nibble-packing requires.

    Returns dict with keys: 'packed', 'scale', 'shape', 'dtype',
    'group_size', 'padded'.  'scale' has shape (..., num_groups, 1).
    """
    orig_dtype = tensor.dtype
    t = tensor.float()

    head_dim   = t.shape[-1]
    group_size = min(group_size, head_dim)            # head_dim < group → 1 group
    pad_len    = (-head_dim) % group_size             # pad up to a multiple of group
    if pad_len:
        t = torch.cat(
            [t, torch.zeros(*t.shape[:-1], pad_len, dtype=t.dtype, device=t.device)],
            dim=-1,
        )
    padded_dim = head_dim + pad_len
    num_groups = padded_dim // group_size

    # (..., num_groups, group_size) — one scale per group from that group's absmax.
    grouped = t.reshape(*t.shape[:-1], num_groups, group_size)
    abs_max = grouped.abs().amax(dim=-1, keepdim=True)        # (..., num_groups, 1)
    scale   = (abs_max / 7.0).clamp(min=1e-8)

    quantized = (grouped / scale).round().clamp(-8, 7).to(torch.int8)
    quantized = quantized.reshape(*t.shape[:-1], padded_dim)  # back to (..., padded_dim)

    # Pack pairs along head_dim into uint8 (padded_dim is even by construction).
    q_uint8 = (quantized & 0x0F).to(torch.uint8)
    packed  = (q_uint8[..., 0::2] | (q_uint8[..., 1::2] << 4))

    return {
        "packed":       packed,
        "scale":        scale.to(torch.float32),
        "shape":        list(tensor.shape),
        "dtype":        str(orig_dtype),
        "group_size":   group_size,
        "padded":       bool(pad_len),
    }


def dequantize_int4(data: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Reconstruct tensor from a group-wise INT4 packed dict."""
    packed     = data["packed"]
    scale      = data["scale"]                       # (..., num_groups, 1)
    orig_shape = data["shape"]
    head_dim   = orig_shape[-1]
    group_size = int(data.get("group_size", head_dim))

    lo = (packed & 0x0F).to(torch.int8)              # lower nibble
    hi = ((packed >> 4) & 0x0F).to(torch.int8)       # upper nibble

    # Restore sign for 4-bit signed
    lo[lo > 7] -= 16
    hi[hi > 7] -= 16

    unpacked   = torch.stack([lo, hi], dim=-1).flatten(start_dim=-2).float()  # (..., padded_dim)
    padded_dim = unpacked.shape[-1]
    num_groups = padded_dim // group_size

    # Reshape into groups, apply each group's scale, flatten, then strip padding.
    grouped = unpacked.reshape(*unpacked.shape[:-1], num_groups, group_size)
    out     = (grouped * scale).reshape(*unpacked.shape[:-1], padded_dim)
    out     = out[..., :head_dim]

    target_dtype = _str_to_dtype(data["dtype"])
    return out.to(target_dtype)


def _str_to_dtype(s: str) -> torch.dtype:
    mapping = {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
    }
    return mapping.get(s, torch.float32)



# Layer-cache-level API  (operates on a legacy KV-cache tuple for one chunk)


def compress_kvcache(legacy_cache: tuple, precision: str) -> Any:
    """
    Compress a full per-chunk legacy KV cache.

    legacy_cache: tuple of (key, value) tensors per layer, as returned by
                  DynamicCache.to_legacy_cache().
    precision: 'fp16' | 'int8' | 'int4'

    Returns a list of dicts (one per layer), each with 'k' and 'v' sub-dicts.
    For 'fp16', k/v are plain tensors.  For 'int8'/'int4' they are quantization dicts.
    """
    precision = precision.lower()
    if precision not in ("fp16", "int8", "int4"):
        raise ValueError(f"precision must be fp16, int8, or int4; got {precision!r}")

    compressed = []
    for layer_k, layer_v in legacy_cache:
        if precision == "fp16":
            # ROOT-CAUSE-1 FIX: store the C1 "reference" cache in bfloat16, not
            # float16.  Qwen2.5 is a bfloat16-native model; downcasting its KV
            # values to fp16 overflows the bf16 dynamic range and reintroduces the
            # NaN/Inf corruption this baseline is supposed to be free of.  The
            # "fp16" precision label is kept for storage-layout compatibility, but
            # the on-disk dtype is now bfloat16 (still 2 bytes — cache_size_bytes
            # is unchanged).
            compressed.append({"k": layer_k.to(torch.bfloat16),
                                "v": layer_v.to(torch.bfloat16)})
        elif precision == "int8":
            compressed.append({"k": quantize_int8(layer_k),
                                "v": quantize_int8(layer_v)})
        elif precision == "int4":
            compressed.append({"k": quantize_int4(layer_k),
                                "v": quantize_int4(layer_v)})
    return compressed


def decompress_kvcache(compressed: list, precision: str) -> tuple:
    """
    Decompress a compressed per-chunk KV cache back to a legacy cache tuple.

    Returns: tuple of (key_tensor, value_tensor) per layer.
    """
    precision = precision.lower()
    legacy = []
    for layer_data in compressed:
        if precision == "fp16":
            # ROOT-CAUSE-1 FIX: keep the reference cache in bfloat16 (see
            # compress_kvcache).  Casting to float16 here would re-corrupt the
            # bf16-native values even though they were stored cleanly.
            k = layer_data["k"].to(torch.bfloat16)
            v = layer_data["v"].to(torch.bfloat16)
        elif precision == "int8":
            k = dequantize_int8(layer_data["k"])
            v = dequantize_int8(layer_data["v"])
        elif precision == "int4":
            k = dequantize_int4(layer_data["k"])
            v = dequantize_int4(layer_data["v"])
        else:
            raise ValueError(f"Unknown precision {precision!r}")
        legacy.append((k, v))
    return tuple(legacy)



# Storage size utility


def cache_size_bytes(compressed: list, precision: str) -> int:
    """Return total byte size of a compressed KV cache list."""
    total = 0
    precision = precision.lower()
    for layer_data in compressed:
        for key in ("k", "v"):
            d = layer_data[key]
            if precision == "fp16":
                total += d.numel() * 2  # float16 = 2 bytes
            elif precision == "int8":
                total += d["quantized"].numel()                   # 1 byte/element
                total += d["scale"].numel() * 4                   # float32
                total += d["zero_point"].numel() * 4              # int32
            elif precision == "int4":
                # packed: 2 nibbles per uint8 byte → numel() is the true byte count.
                # scale:  now PER-GROUP (num_groups per head_dim vector, not 1),
                #         so numel() is larger than the old per-token scheme.
                #         Reading numel() keeps the reported size honest automatically.
                total += d["packed"].numel()                      # 1 byte / packed pair
                total += d["scale"].numel() * 4                   # float32, per group
    return total
