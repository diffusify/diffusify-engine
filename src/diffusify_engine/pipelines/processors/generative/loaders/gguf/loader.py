import torch
import gguf

from .dequant import is_quantized
from .ops import GGMLTensor

IMG_ARCH_LIST = {"hyvideo"}  # Add "hyvideo" for Hunyuan Video models

def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))

def load_gguf_unet(model_path, device, offload_device, handle_prefix="model.diffusion_model.", return_arch=False):
    """
    Loads a GGUF UNET model, adapting logic from gguf_sd_loader in ComfyUI-GGUF.
    """
    reader = gguf.GGUFReader(model_path)

    # filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    # detect and verify architecture
    compat = None
    arch_str = None
    arch_field = reader.get_field("general.architecture")
    if arch_field is not None:
        if len(arch_field.types) != 1 or arch_field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF general.architecture key: expected string, got {arch_field.types!r}")
        arch_str = str(arch_field.parts[arch_field.data[-1]], encoding="utf-8")
        if arch_str not in IMG_ARCH_LIST:
            raise ValueError(f"Unexpected architecture type in GGUF file, expected one of hyvideo but got {arch_str!r}")

    # main loading loop
    state_dict = {}
    qtype_dict = {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        tensor_type_str = str(tensor.tensor_type)
        torch_tensor = torch.from_numpy(tensor.data) # mmap

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        # add to state dict
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
        state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    # mark largest tensor for vram estimation
    qsd = {k:v for k,v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    # sanity check debug print
    print("\nggml_sd_loader:")
    for k,v in qtype_dict.items():
        print(f" {k:30}{v:3}")

    if return_arch:
        return (state_dict, arch_str)
    return state_dict
