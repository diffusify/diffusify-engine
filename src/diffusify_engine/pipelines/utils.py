
import torch
import logging
import safetensors.torch
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

def soft_empty_cache(force=False):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def get_fp_maxval(bits=8, mantissa_bit=3, sign_bits=1):
    _bits = torch.tensor(bits)
    _mantissa_bit = torch.tensor(mantissa_bit)
    _sign_bits = torch.tensor(sign_bits)
    M = torch.clamp(torch.round(_mantissa_bit), 1, _bits - _sign_bits)
    E = _bits - _sign_bits - M
    bias = 2 ** (E - 1) - 1
    mantissa = 1
    for i in range(mantissa_bit - 1):
        mantissa += 1 / (2 ** (i+1))
    maxval = mantissa * 2 ** (2**E - 1 - bias)
    return maxval

def quantize_to_fp8(x, bits=8, mantissa_bit=3, sign_bits=1):
    """
    Default is E4M3.
    """
    bits = torch.tensor(bits)
    mantissa_bit = torch.tensor(mantissa_bit)
    sign_bits = torch.tensor(sign_bits)
    M = torch.clamp(torch.round(mantissa_bit), 1, bits - sign_bits)
    E = bits - sign_bits - M
    bias = 2 ** (E - 1) - 1
    mantissa = 1
    for i in range(mantissa_bit - 1):
        mantissa += 1 / (2 ** (i+1))
    maxval = mantissa * 2 ** (2**E - 1 - bias)
    minval = - maxval
    minval = - maxval if sign_bits == 1 else torch.zeros_like(maxval)
    input_clamp = torch.min(torch.max(x, minval), maxval)
    log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(input_clamp)) + bias)).detach(), 1.0)
    log_scales = 2.0 ** (log_scales - M - bias.type(x.dtype))
    # dequant
    qdq_out = torch.round(input_clamp / log_scales) * log_scales
    return qdq_out, log_scales

def fp8_tensor_quant(x, scale, bits=8, mantissa_bit=3, sign_bits=1):
    for i in range(len(x.shape) - 1):
        scale = scale.unsqueeze(-1)
    new_x = x / scale
    quant_dequant_x, log_scales = quantize_to_fp8(new_x, bits=bits, mantissa_bit=mantissa_bit, sign_bits=sign_bits)
    return quant_dequant_x, scale, log_scales

def fp8_activation_dequant(qdq_out, scale, dtype):
    qdq_out = qdq_out.type(dtype)
    quant_dequant_x = qdq_out * scale.to(dtype)
    return quant_dequant_x

def fp8_linear_forward(cls, original_dtype, input):
    weight_dtype = cls.weight.dtype
    #####
    if cls.weight.dtype != torch.float8_e4m3fn:
        maxval = get_fp_maxval()
        scale = torch.max(torch.abs(cls.weight.flatten())) / maxval
        linear_weight, scale, log_scales = fp8_tensor_quant(cls.weight, scale)
        linear_weight = linear_weight.to(torch.float8_e4m3fn)
        weight_dtype = linear_weight.dtype
    else:
        scale = cls.fp8_scale.to(cls.weight.device)
        linear_weight = cls.weight
    #####

    if weight_dtype == torch.float8_e4m3fn and cls.weight.sum() != 0:
        if True or len(input.shape) == 3:
            cls_dequant = fp8_activation_dequant(linear_weight, scale, original_dtype)
            if cls.bias != None:
                output = F.linear(input, cls_dequant, cls.bias)
            else:
                output = F.linear(input, cls_dequant)
            return output
        else:
            return cls.original_forward(input.to(original_dtype))
    else:
        return cls.original_forward(input)

def convert_fp8_linear(module, original_dtype):
    setattr(module, "fp8_matmul_enabled", True)
    script_directory = os.path.dirname(os.path.abspath(__file__))
    fp8_map_path = os.path.join(script_directory,"models/hunyuan/config/fp8_map.safetensors")
    if os.path.exists(fp8_map_path):
        fp8_map = load_torch_file(fp8_map_path, safe_load=True)
    else:
        raise ValueError(f"Invalid fp8_map path: {fp8_map_path}.")
    for key, layer in module.named_modules():
        if isinstance(layer, nn.Linear) and ('double_blocks' in key or 'single_blocks' in key):
            original_forward = layer.forward
            layer.weight = torch.nn.Parameter(layer.weight.to(torch.float8_e4m3fn))
            setattr(layer, "fp8_scale", fp8_map[key].to(dtype=original_dtype))
            setattr(layer, "original_forward", original_forward)
            setattr(layer, "forward", lambda input, m=layer: fp8_linear_forward(m, original_dtype, input))

def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                logging.warning("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        if "global_step" in pl_sd:
            logging.debug(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            if len(pl_sd) == 1:
                key = list(pl_sd.keys())[0]
                sd = pl_sd[key]
                if not isinstance(sd, dict):
                    sd = pl_sd
            else:
                sd = pl_sd
    return sd

# TODO: useful for future model inspection
# def calculate_parameters(sd, prefix=""):
#     params = 0
#     for k in sd.keys():
#         if k.startswith(prefix):
#             w = sd[k]
#             params += w.nelement()
#     return params

# TODO: useful for future model inspection
# def weight_dtype(sd, prefix=""):
#     dtypes = {}
#     for k in sd.keys():
#         if k.startswith(prefix):
#             w = sd[k]
#             dtypes[w.dtype] = dtypes.get(w.dtype, 0) + w.numel()
#     if len(dtypes) == 0:
#         return None
#     return max(dtypes, key=dtypes.get)
