import torch

from .dequant import is_quantized, dequantize_tensor

def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item

class GGUFModelPatcher(torch.nn.Module):
    def __init__(self, model, load_device, offload_device):
        super().__init__()
        self.model = model
        self.patches = {}
        self.backup = {}
        self.load_device = load_device
        self.offload_device = offload_device
        self.patch_on_device = False
        self.tensor_map = {}  # Add a dictionary to store tensors

    def add_tensor(self, key, tensor):
        self.tensor_map[key] = tensor

    def patch_model(self, device, dtype):
        if self.model is None:
            raise ValueError("Model not set for patching.")
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                self.patch_linear(name, module, device, dtype)
        # Add similar logic for other layer types (e.g., Conv2d, Embedding, LayerNorm) if needed.

    def patch_linear(self, module_name, module, device, dtype):
        weight_key = f"{module_name}.weight"
        bias_key = f"{module_name}.bias"

        # Retrieve weight and bias from tensor_map
        weight = self.tensor_map.get(weight_key)
        bias = self.tensor_map.get(bias_key)

        if weight is not None:
            setattr(module, "weight", torch.nn.Parameter(weight.to(device, dtype=dtype), requires_grad=False))

        if bias is not None:
            setattr(module, "bias", torch.nn.Parameter(bias.to(device, dtype=dtype), requires_grad=False))

    def get_weight(self, key, device, dtype, original_weight):
        if key not in self.patches:
            return original_weight
        weight = self.patches[key]
        if is_quantized(weight):
            weight = dequantize_tensor(weight, dtype=dtype).to(device)
        return weight

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        p = set()
        for k in patches:
            if isinstance(k, str):
                key = k
            else:
                key = k[0]
            if key in self.model.state_dict():
                p.add(k)
                current_patches = self.patches.get(key, [])
                current_patches.append((strength_patch, patches[k], strength_model))
                self.patches[key] = current_patches
        return list(p)
