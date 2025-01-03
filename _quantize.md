    
    # TODO: fp8_scaled
    # for name, param in transformer.named_parameters():
    #     dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
    #     print(f"set module to device: {name} - {dtype_to_use}")
    #     set_module_tensor_to_device(transformer, name, device=device, dtype=dtype_to_use, value=sd[name])
    # convert_fp8_linear(transformer, base_dtype, MODEL_MAP_PATH)

    # TODO: torchao pre-quantize linear weights
    # quant_func = fpx_weight_only(3, 2)  # (3, 1) # FP5 # FP6 (3 exponent bits, 2 mantissa bits)
    # for name, param in transformer.named_parameters():
    #     dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
    #     if name in sd:
    #         # Access the dtype directly from the state dict entry
    #         state_dict_dtype = sd[name].dtype
    #         print(f"{name}: model param dtype [{param.dtype}], sd dtype [{state_dict_dtype}]")
    #     else:
    #         print(f"{name} not in state dict (problem)")
    #     if dtype_to_use == quant_dtype:
    #         # if any(parent_name in name for parent_name in ["single_blocks", "double_blocks"]):
    #         parent_name = name.rsplit('.', 1)[0]
    #         module = transformer.get_submodule(parent_name)
    #         if isinstance(module, torch.nn.Linear):
    #             print(f"quantizing: {parent_name}")
    #             module._quantized = True
    #             weight_fp8 = sd[name]
    #             temp_layer = torch.nn.Linear(weight_fp8.shape[1], weight_fp8.shape[0], bias=False, device=offload_device)
    #             with torch.no_grad():
    #                 temp_layer.weight.copy_(weight_fp8)
    #             quantize_(temp_layer, quant_func)
    #             weight_fp6_cpu = temp_layer.weight.detach()
    #             set_module_tensor_to_device(
    #                 transformer, 
    #                 parent_name + ".weight", 
    #                 device=device, 
    #                 # dtype=base_dtype,
    #                 value=weight_fp6_cpu
    #             )
    #             del temp_layer
    #             soft_empty_cache()
    #             print(f"quantized: {parent_name}")
    #     else:
    #         set_module_tensor_to_device(transformer, name, device=device, dtype=dtype_to_use, value=sd[name])
    #         print(f"sent to device: {name}")
    # # Example of modifying the forward method of a quantized layer:
    # for name, module in transformer.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         if hasattr(module, 'weight') and hasattr(module, "_quantized") and module._quantized:  # Check if it's a quantized layer
    #             module.original_forward = module.forward
    #             def new_forward(self, x, *args, **kwargs):
    #                 # Dequantize on-the-fly
    #                 dequantized_weight = self.weight.dequantize()
    #                 # Perform the linear operation with the dequantized weight
    #                 return torch.nn.functional.linear(x, dequantized_weight, self.bias)
    #             module.forward = new_forward.__get__(module)
    #         else:
    #             print(f"maybe not quantized: {name}")
    #


    # 1. Load weights into the model (on offload_device initially)
    for name, param in transformer.named_parameters():
        if name in sd:
            set_module_tensor_to_device(transformer, name, device=offload_device, dtype=base_dtype, value=sd[name])
        else:
            print(f"{name} not in state dict (problem)")

    # 2. Define the quantization function
    quant_func = fpx_weight_only(3, 1) # FP6 (3 exponent bits, 2 mantissa bits)

    # 3. Define a filter function to select layers for quantization
    def quant_filter(module: torch.nn.Module, fqn: str) -> bool:
        # Quantize only Linear layers that are meant to be quantized
        is_match = isinstance(module, torch.nn.Linear) and any(keyword in fqn for keyword in ["single_blocks", "double_blocks"])
        print(f"quantizing: {module._get_name()} | {fqn} | {is_match}")
        return is_match

    # 5. Move non-quantized parameters to the target device
    for name, param in transformer.named_parameters():
        if any(keyword in name for keyword in params_to_keep):
            print(f"moving to device: {name}")
            set_module_tensor_to_device(transformer, name, device=device, dtype=base_dtype)

    # 4. Quantize the model in-place
    quantize_(transformer, quant_func, filter_fn=quant_filter, device=device)