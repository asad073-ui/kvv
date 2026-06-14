
from __future__ import annotations
import torch
from typing import Tuple, Dict, Any, List



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


def quantize_int4(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Per-token symmetric INT4 quantization along the last dimension (head_dim).
    Range: [-8, 7].  Two INT4 values packed into one uint8 byte.

    Returns dict with keys: 'packed', 'scale', 'shape', 'dtype'
    """
    orig_dtype = tensor.dtype
    t = tensor.float()

    abs_max = t.abs().max(dim=-1, keepdim=True).values
    scale   = (abs_max / 7.0).clamp(min=1e-8)

    quantized = (t / scale).round().clamp(-8, 7).to(torch.int8)

    # Pack pairs along head_dim into uint8
    head_dim = quantized.shape[-1]
    pad = (head_dim % 2 != 0)
    if pad:
        quantized = torch.cat(
            [quantized, torch.zeros(*quantized.shape[:-1], 1, dtype=torch.int8, device=quantized.device)],
            dim=-1
        )
    q_uint8 = (quantized & 0x0F).to(torch.uint8)
    packed  = (q_uint8[..., 0::2] | (q_uint8[..., 1::2] << 4))

    return {
        "packed":       packed,
        "scale":        scale.to(torch.float32),
        "shape":        list(tensor.shape),
        "dtype":        str(orig_dtype),
        "padded":       pad,
    }


def dequantize_int4(data: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Reconstruct FP32 tensor from INT4 packed dict."""
    packed = data["packed"]
    scale  = data["scale"]
    orig_shape = data["shape"]
    head_dim   = orig_shape[-1]

    lo = (packed & 0x0F).to(torch.int8)          # lower nibble
    hi = ((packed >> 4) & 0x0F).to(torch.int8)   # upper nibble

    # Restore sign for 4-bit signed
    lo[lo > 7] -= 16
    hi[hi > 7] -= 16

    unpacked = torch.stack([lo, hi], dim=-1).flatten(start_dim=-2).float()
    if data.get("padded"):
        unpacked = unpacked[..., :head_dim]

    out = unpacked * scale
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
            compressed.append({"k": layer_k.to(torch.float16),
                                "v": layer_v.to(torch.float16)})
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
            
            k = layer_data["k"].to(torch.float16)
            v = layer_data["v"].to(torch.float16)
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
                total += d["packed"].numel()                      # 0.5 bytes avg → stored as uint8
                total += d["scale"].numel() * 4                   # float32
    return total
