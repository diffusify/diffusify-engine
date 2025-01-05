import torch
import torch.nn as nn

from einops import rearrange
from contextlib import contextmanager
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer
from .attention import attention, get_cu_seqlens
from .enhance import get_feta_scores
from .posemb_layers import apply_rotary_emb
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from .mlp_layers import MLP, MLPEmbedder, FinalLayer
from .modulate_layers import ModulateDiT, modulate, apply_gate
from .token_refiner import SingleTokenRefiner
from .norm_layers import RMSNorm

from typing import List, Tuple, Optional, Union, Dict

@contextmanager
def init_weights_on_device(device=torch.device("meta"), include_buffers:bool = False):
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer
    
    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)
            
    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper
    
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}
    
    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)

class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        attention_mode: str = "flash_attn_varlen",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.attention_mode = attention_mode

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        frames: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )

        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            
        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
                
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        feta_scores = get_feta_scores(img_q, img_k, frames)
    
        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)

        attn = attention(
            q,
            k,
            v,
            heads = self.heads_num,
            mode=self.attention_mode,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=img_k.shape[0],
            attn_mask=attn_mask
        )

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]
        img_attn *= feta_scores

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt

class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        attention_mode: str = "flash_attn_varlen",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.attention_mode = attention_mode

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(
            hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs
        )
        # proj and mlp_out
        self.linear2 = nn.Linear(
            hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        frames: int,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        stg_mode: Optional[str] = None,
    ) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            # assert (
            #     img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            # ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        # feta scores
        # feta_scores = get_feta_scores(img_q, img_k, frames)

        # Compute attention.
        # assert (
        #    cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        # ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
        if stg_mode is not None:
            if stg_mode == "STG-A":
                attn = attention(
                    q,
                    k,
                    v,
                    heads = self.heads_num,
                    mode=self.attention_mode,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    batch_size=x.shape[0],
                    do_stg=True,
                    txt_len=txt_len,
                    attn_mask=attn_mask
                )
                output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
                return x + apply_gate(output, gate=mod_gate)
            elif stg_mode == "STG-R":
                attn = attention(
                    q,
                    k,
                    v,
                    heads = self.heads_num,
                    mode=self.attention_mode,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    batch_size=x.shape[0],
                    attn_mask=attn_mask
                )
                # Compute activation in mlp stream, cat again and run second linear layer.
                output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
                output = apply_gate(output, gate=mod_gate)
                batch_size = output.shape[0]
                output[:batch_size-1, :, :] = 0
                return x + output
        else:
            attn = attention(
                q,
                k,
                v,
                heads = self.heads_num,
                mode=self.attention_mode,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=x.shape[0],
                attn_mask=attn_mask
            )

            # Compute activation in mlp stream, cat again and run second linear layer.
            output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
            # output *= feta_scores
            output = x + apply_gate(output, gate=mod_gate)
            
            return output

class HYVideoDiffusionTransformer(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model.
    device: torch.device
        The device of the model.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # for modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        main_device: Optional[torch.device] = None,
        offload_device: Optional[torch.device] = None,
        attention_mode: str = "flash_attn_varlen"
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list

        self.main_device = main_device
        self.offload_device = offload_device
        self.attention_mode = attention_mode

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
        )

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )

        # text modulation
        self.vector_in = MLPEmbedder(
            self.text_states_dim_2, self.hidden_size, **factory_kwargs
        )

        # guidance modulation
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )

        # double blocks
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    attention_mode=attention_mode,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        # single blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    attention_mode=attention_mode,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.double_blocks_to_swap = -1
        self.single_blocks_to_swap = -1
        self.offload_txt_in = False
        self.offload_img_in = False

    # thanks @2kpr for the initial block swap code!
    def block_swap(self, double_blocks_to_swap, single_blocks_to_swap, offload_txt_in=False, offload_img_in=False):
        print(f"Swapping {double_blocks_to_swap + 1} double blocks and {single_blocks_to_swap + 1} single blocks")
        self.double_blocks_to_swap = double_blocks_to_swap
        self.single_blocks_to_swap = single_blocks_to_swap
        self.offload_txt_in = offload_txt_in
        self.offload_img_in = offload_img_in
        for b, block in enumerate(self.double_blocks):
            if b > self.double_blocks_to_swap:
                #print(f"Moving double_block {b} to main device")
                block.to(self.main_device)
            else:
                #print(f"Moving double_block {b} to offload_device")
                block.to(self.offload_device)
        for b, block in enumerate(self.single_blocks):
            if b > self.single_blocks_to_swap:
                block.to(self.main_device)
            else:
                block.to(self.offload_device)

    def enable_auto_offload(self, dtype=torch.bfloat16, device="cuda"):
        def cast_to(weight, dtype=None, device=None, copy=False):
            if device is None or weight.device == device:
                if not copy:
                    if dtype is None or weight.dtype == dtype:
                        return weight
                return weight.to(dtype=dtype, copy=copy)

            r = torch.empty_like(weight, dtype=dtype, device=device)
            r.copy_(weight)
            return r

        def cast_weight(s, input=None, dtype=None, device=None):
            if input is not None:
                if dtype is None:
                    dtype = input.dtype
                if device is None:
                    device = input.device
            weight = cast_to(s.weight, dtype, device)
            return weight

        def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
            if input is not None:
                if dtype is None:
                    dtype = input.dtype
                if bias_dtype is None:
                    bias_dtype = dtype
                if device is None:
                    device = input.device
            weight = cast_to(s.weight, dtype, device)
            bias = cast_to(s.bias, bias_dtype, device) if s.bias is not None else None
            return weight, bias

        class quantized_layer:
            class Linear(torch.nn.Linear):
                def __init__(self, *args, dtype=torch.bfloat16, device="cuda", **kwargs):
                    super().__init__(*args, **kwargs)
                    self.dtype = dtype
                    self.device = device

                def block_forward_(self, x, i, j, dtype, device):
                    weight_ = cast_to(
                        self.weight[j * self.block_size: (j + 1) * self.block_size, i * self.block_size: (i + 1) * self.block_size],
                        dtype=dtype, device=device
                    )
                    if self.bias is None or i > 0:
                        bias_ = None
                    else:
                        bias_ = cast_to(self.bias[j * self.block_size: (j + 1) * self.block_size], dtype=dtype, device=device)
                    x_ = x[..., i * self.block_size: (i + 1) * self.block_size]
                    y_ = torch.nn.functional.linear(x_, weight_, bias_)
                    del x_, weight_, bias_
                    torch.cuda.empty_cache()
                    return y_
                
                def block_forward(self, x, **kwargs):
                    # This feature can only reduce 2GB VRAM, so we disable it.
                    y = torch.zeros(x.shape[:-1] + (self.out_features,), dtype=x.dtype, device=x.device)
                    for i in range((self.in_features + self.block_size - 1) // self.block_size):
                        for j in range((self.out_features + self.block_size - 1) // self.block_size):
                            y[..., j * self.block_size: (j + 1) * self.block_size] += self.block_forward_(x, i, j, dtype=x.dtype, device=x.device)
                    return y
                    
                def forward(self, x, **kwargs):
                    weight, bias = cast_bias_weight(self, x, dtype=self.dtype, device=self.device)
                    return torch.nn.functional.linear(x, weight, bias)

            class RMSNorm(torch.nn.Module):
                def __init__(self, module, dtype=torch.bfloat16, device="cuda"):
                    super().__init__()
                    self.module = module
                    self.dtype = dtype
                    self.device = device
                    
                def forward(self, hidden_states, **kwargs):
                    input_dtype = hidden_states.dtype
                    variance = hidden_states.to(torch.float32).square().mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(variance + self.module.eps)
                    hidden_states = hidden_states.to(input_dtype)
                    if self.module.weight is not None:
                        weight = cast_weight(self.module, hidden_states, dtype=torch.bfloat16, device="cuda")
                        hidden_states = hidden_states * weight
                    return hidden_states
                
            class Conv3d(torch.nn.Conv3d):
                def __init__(self, *args, dtype=torch.bfloat16, device="cuda", **kwargs):
                    super().__init__(*args, **kwargs)
                    self.dtype = dtype
                    self.device = device
                    
                def forward(self, x):
                    weight, bias = cast_bias_weight(self, x, dtype=self.dtype, device=self.device)
                    return torch.nn.functional.conv3d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
                
            class LayerNorm(torch.nn.LayerNorm):
                def __init__(self, *args, dtype=torch.bfloat16, device="cuda", **kwargs):
                    super().__init__(*args, **kwargs)
                    self.dtype = dtype
                    self.device = device
                    
                def forward(self, x):
                    if self.weight is not None and self.bias is not None:
                        weight, bias = cast_bias_weight(self, x, dtype=self.dtype, device=self.device)
                        return torch.nn.functional.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
                    else:
                        return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        
        def replace_layer(model, dtype=torch.bfloat16, device="cuda"):
                for name, module in model.named_children():
                    if isinstance(module, torch.nn.Linear):
                        with init_weights_on_device():
                            new_layer = quantized_layer.Linear(
                                module.in_features, module.out_features, bias=module.bias is not None,
                                dtype=dtype, device=device
                            )
                        new_layer.load_state_dict(module.state_dict(), assign=True)
                        setattr(model, name, new_layer)
                    elif isinstance(module, torch.nn.Conv3d):
                        with init_weights_on_device():
                            new_layer = quantized_layer.Conv3d(
                                module.in_channels, module.out_channels, kernel_size=module.kernel_size, stride=module.stride,
                                dtype=dtype, device=device
                            )
                        new_layer.load_state_dict(module.state_dict(), assign=True)
                        setattr(model, name, new_layer)
                    elif isinstance(module, RMSNorm):
                        new_layer = quantized_layer.RMSNorm(
                            module,
                            dtype=dtype, device=device
                        )
                        setattr(model, name, new_layer)
                    elif isinstance(module, torch.nn.LayerNorm):
                        with init_weights_on_device():
                            new_layer = quantized_layer.LayerNorm(
                                module.normalized_shape, elementwise_affine=module.elementwise_affine, eps=module.eps,
                                dtype=dtype, device=device
                            )
                        new_layer.load_state_dict(module.state_dict(), assign=True)
                        setattr(model, name, new_layer)
                    else:
                        replace_layer(module, dtype=dtype, device=device)

        replace_layer(self, dtype=dtype, device=device)

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        stg_mode: str = None,
        stg_block_idx: int = -1,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # prep vars
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        
        # feta calc set
        frames = img.shape[2]

        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        if text_states_2 is not None:
            vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if guidance is not None:
            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        if self.offload_txt_in:
            self.txt_in.to(self.main_device)
        if self.offload_img_in:
            self.img_in.to(self.main_device)

        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )
        if self.offload_txt_in:
            self.txt_in.to(self.offload_device, non_blocking=True)
        if self.offload_img_in:
            self.img_in.to(self.offload_device, non_blocking=True)

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]
        max_seqlen_q = max_seqlen_kv = img_seq_len + txt_seq_len

        if self.attention_mode == "sdpa" or self.attention_mode == "comfy":
            cu_seqlens_q, cu_seqlens_kv = None, None
            # Create a square boolean mask filled with False
            attn_mask = torch.zeros((1, max_seqlen_q, max_seqlen_q), dtype=torch.bool, device=text_mask.device)

            # Calculate the valid attention regions
            text_len = text_mask[0].sum().item()
            total_len = text_len + img_seq_len

            # Allow attention to all tokens up to total_len
            attn_mask[0, :total_len, :total_len] = True
        else:
            attn_mask = None
            # Compute cu_squlens for flash attention
            cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
            cu_seqlens_kv = cu_seqlens_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        for b, block in enumerate(self.double_blocks):
            if b <= self.double_blocks_to_swap and self.double_blocks_to_swap >= 0:
                #print(f"Moving double_block {b} to main device")
                block.to(self.main_device)
            double_block_args = [
                img,
                txt,
                vec,
                frames,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                freqs_cis,
                attn_mask
            ]

            img, txt = block(*double_block_args)
            if b <= self.double_blocks_to_swap and self.double_blocks_to_swap >= 0:
                #print(f"Moving double_block {b} to offload device")
                block.to(self.offload_device, non_blocking=True)

        # Merge txt and img to pass through single stream blocks.
        x = torch.cat((img, txt), 1)
        if len(self.single_blocks) > 0:
            for b, block in enumerate(self.single_blocks):
                if b <= self.single_blocks_to_swap and self.single_blocks_to_swap >= 0:
                    #print(f"Moving single_block {b} to main device")
                    #mm.soft_empty_cache()
                    block.to(self.main_device)
                curr_stg_mode = stg_mode if b == stg_block_idx else None
                single_block_args = [
                    x,
                    vec,
                    frames,
                    txt_seq_len,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    (freqs_cos, freqs_sin),
                    attn_mask,
                    curr_stg_mode,
                ]

                x = block(*single_block_args)
                if b <= self.single_blocks_to_swap and self.single_blocks_to_swap >= 0:
                    #print(f"Moving single_block {b} to offload device")
                    #mm.soft_empty_cache()
                    block.to(self.offload_device, non_blocking=True)

        img = x[:, :img_seq_len, ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return img

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_qkv.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_qkv.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1.parameters())
                    + sum(p.numel() for p in block.linear2.parameters())
                    for block in self.single_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts

#################################################################################
#                             HunyuanVideo Configs                              #
#################################################################################

HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
}
