import os
import sys
import traceback
import gc
import glob
import json
import ffmpeg

import torch
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm

from typing import List

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from diffusers.video_processor import VideoProcessor

from .processors.generative.diffusion.hunyuan.modules.models import HYVideoDiffusionTransformer
from .processors.generative.diffusion.hunyuan.hunyuan_video_pipeline import HunyuanVideoPipeline
from .processors.generative.diffusion.hunyuan.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from .processors.generative.diffusion.hunyuan.text_encoder.encoder import TextEncoder
from .processors.generative.diffusion.hunyuan.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from .processors.generative.diffusion.hunyuan.modules.posemb_layers import get_nd_rotary_pos_embed
from .processors.generative.diffusion.hunyuan.helpers import align_to
from .utils import convert_fp8_linear, load_torch_file, soft_empty_cache

# Configuration
MODEL_PATH = "/home/ubuntu/share/comfyui/models/diffusion_models/hunyuan-video-720-fp8.pt"
VAE_PATH = "/home/ubuntu/share/comfyui/models/vae/hunyuan-video-vae-bf16.safetensors"
LLM_PATH = "/home/ubuntu/share/comfyui/models/llm/llava-llama-3-8b-text-encoder-tokenizer"
CLIP_PATH = "/home/ubuntu/share/comfyui/models/clip/clip-vit-large-patch14"
PROMPT = "a warmly lit café at night, with hanging spherical Edison bulbs casting a soft glow over polished wooden tables and velvet booths, large floor-to-ceiling windows reveal a rainy European street outside where raindrops streak the glass and blurred headlights of passing cars create a dreamy ambiance, the camera zooms slowly on a steaming cup of coffee, next to flickering candles, and pastries."
NEGATIVE_PROMPT = None
INPUT_FRAMES_PATH = "/home/ubuntu/share/tests-frames"
OUTPUT_VIDEO = "output_video.mp4"
WIDTH = 960
HEIGHT = 544
NUM_FRAMES = 35
STEPS = 30
CFG_SCALE = 1.0
CFG_SCALE_START = 0.9
CFG_SCALE_END = 1.0
EMBEDDED_GUIDANCE_SCALE = 5.0
FLOW_SHIFT = 5.0
SEED = 348273
DENOISE_STRENGTH = 1.0
BASE_DTYPE = torch.bfloat16
QUANT_DTYPE = torch.float8_e4m3fn
PARAMS_TO_KEEP = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
SWAP_DOUBLE_BLOCKS = 4
SWAP_SINGLE_BLOCKS = 0
OFFLOAD_TXT_IN = False
OFFLOAD_IMG_IN = False

PROMPT_TEMPLATE_ENCODE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)

PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)  

PROMPT_TEMPLATE = {
    "dit-llm-encode": {
        "template": PROMPT_TEMPLATE_ENCODE,
        "crop_start": 36,
    },
    "dit-llm-encode-video": {
        "template": PROMPT_TEMPLATE_ENCODE_VIDEO,
        "crop_start": 95,
    },
}

def load_video_frames(video_path):
    image_files = sorted(glob.glob(os.path.join(video_path, "*.png")))  # Assuming PNG images
    if not image_files:
        raise ValueError(f"No image files found in the specified folder: {video_path}")
    frames = []
    for i, image_file in tqdm(enumerate(image_files), desc="Loading frames", total=len(image_files)):
        try:
            img = Image.open(image_file).convert("RGB")
            frames.append(img)
        except Exception as e:
            print(f"Warning: Could not load image {image_file}. Skipping. Error: {e}")
    if not frames:
        raise ValueError("No valid images were loaded.")
    return frames

def save_video_ffmpeg(frames, output_path, fps=24):
    """
    Save a sequence of frames as a video file using python-ffmpeg directly.
    """
    try:
        # Create a temporary directory to store frames
        temp_frame_dir = "temp_frames"
        os.makedirs(temp_frame_dir, exist_ok=True)

        # Save frames as PNG files in the temporary directory
        for idx, frame in enumerate(frames):
            frame.save(os.path.join(temp_frame_dir, f"frame{idx:08d}.png"))

        # Setup video encoder using python-ffmpeg
        video_stream = ffmpeg.input(
            os.path.join(temp_frame_dir, 'frame%08d.png'), r=fps
        )

        stream = ffmpeg.output(
            video_stream,
            output_path,
            vcodec='libx264',
            pix_fmt='yuv420p',
            crf=12
        )

        stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)

        # Clean up temporary directory
        for f in os.listdir(temp_frame_dir):
            os.remove(os.path.join(temp_frame_dir, f))
        os.rmdir(temp_frame_dir)

        print(f"Video saved successfully to: {output_path}")
        return output_path

    except ffmpeg.Error as e:
        print(f"Error creating video: {e.stderr.decode()}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Error saving video: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def load_vae(vae_path, device, offload_device, precision):
    try:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        script_directory = os.path.dirname(os.path.abspath(__file__))
        vae_config = os.path.join(script_directory, "models/hunyuan/config/hy_vae_config.json")

        # Load VAE configuration
        with open(vae_config, 'r') as f:
            vae_config = json.load(f)

        # Load VAE state dict
        vae_state_dict = load_torch_file(vae_path, device=offload_device)

        # Initialize and load VAE
        vae = AutoencoderKLCausal3D.from_config(vae_config)
        vae.load_state_dict(vae_state_dict)
        vae.requires_grad_(False)
        vae.eval()
        vae.to(device=device, dtype=dtype)

        del vae_state_dict

        return vae

    except Exception as e:
        print(f"Error in load_vae: {str(e)}")
        raise

def encode_video(vae, frames, device, offload_device):
    try:
        # Convert PIL Images to tensors and normalize
        tensor_frames = []
        for frame in frames:
            frame = T.functional.pil_to_tensor(frame)
            tensor_frames.append(frame)

        # Assuming frames are PIL Images
        vae.to(device)
        vae.enable_tiling()

        # Stack frames into a single tensor
        tensor_frames = torch.stack(tensor_frames, dim=0).to(device).unsqueeze(0)  # Shape: [1, T, C, H, W]
        tensor_frames = tensor_frames.permute(0, 2, 1, 3, 4).float()  # Shape: [1, C, T, H, W]
        tensor_frames = (tensor_frames / 255.0) * 2.0 - 1.0
        tensor_frames = tensor_frames.to(vae.dtype)

        # Encode
        generator = torch.Generator(device=torch.device("cpu"))
        latents = vae.encode(tensor_frames).latent_dist.sample(generator)
        latents = latents * vae.config.scaling_factor

        # Offload VAE
        latents.to(offload_device)
        vae.to(offload_device)

        return latents
    except Exception as e:
        print(f"Error in encode_video: {str(e)}")
        raise

def decode_video(vae, latents, device, offload_device):
    try:
        vae.to(device)
        vae.enable_tiling()

        # # Handle input dimensions
        # if len(latents.shape) == 4:
        #     if isinstance(vae, AutoencoderKLCausal3D):
        #         latents = latents.unsqueeze(2)
        # elif len(latents.shape) != 5:
        #     raise ValueError(
        #         f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
        #     )

        latents = latents / vae.config.scaling_factor
        latents = latents.to(vae.dtype).to(device)

        # Create CPU generator for reproducibility
        generator = torch.Generator(device=torch.device("cpu"))

        # Decode latents
        with torch.no_grad():
            video = vae.decode(latents, return_dict=False, generator=generator)[0]

        video_processor = VideoProcessor(vae_scale_factor=8)
        video_processor.config.do_resize = False
        video = video_processor.postprocess_video(video=video, output_type="pt")

        frames = video.squeeze(0)
        frames = frames.permute(0, 2, 3, 1).cpu().float()  # [F, H, W, C]

        # Convert tensors to PIL Images
        pil_frames = []
        # os.makedirs("debug_frames", exist_ok=True)
        for idx, frame in enumerate(frames):
            frame_uint8 = (frame * 255).round().to(torch.uint8).numpy()
            pil_frame = Image.fromarray(frame_uint8, mode='RGB')
            # pil_frame.save(f"debug_frames/frame_{idx:04d}.png")
            pil_frames.append(pil_frame)

        vae.to(offload_device)

        return pil_frames

    except Exception as e:
        print(f"Error in decode_video: {str(e)}")
        raise

def load_text_encoder(precision, device, offload_device, apply_final_norm=False, hidden_state_skip_layer=2):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

    text_encoder_2 = TextEncoder(
        text_encoder_path=CLIP_PATH,
        text_encoder_type="clipL",
        max_length=77,
        text_encoder_precision=precision,
        tokenizer_type="clipL",
        logger=None,
        device=device
    )

    text_encoder = TextEncoder(
        text_encoder_path=LLM_PATH,
        text_encoder_type="llm",
        max_length=256,
        text_encoder_precision=precision,
        tokenizer_type="llm",
        hidden_state_skip_layer=hidden_state_skip_layer,
        apply_final_norm=apply_final_norm,
        logger=None,
        device=device,
        dtype=dtype
    )

    return text_encoder, text_encoder_2

def encode_text(text_encoder_1, text_encoder_2, device, offload_device, prompt, negative_prompt, cfg_scale=1.0, prompt_template="video", custom_prompt_template=None):
    if prompt_template != "disabled":
        if prompt_template == "custom":
            prompt_template_dict = custom_prompt_template
        elif prompt_template == "video":
            prompt_template_dict = PROMPT_TEMPLATE["dit-llm-encode-video"]
        elif prompt_template == "image":
            prompt_template_dict = PROMPT_TEMPLATE["dit-llm-encode"]
        else:
            raise ValueError(f"Invalid prompt_template: {prompt_template_dict}")
        assert (
            isinstance(prompt_template_dict, dict)
            and "template" in prompt_template_dict
        ), f"`prompt_template` must be a dictionary with a key 'template', got {prompt_template_dict}"
        assert "{}" in str(prompt_template_dict["template"]), (
            "`prompt_template['template']` must contain a placeholder `{}` for the input text, "
            f"got {prompt_template_dict['template']}"
        )
    else:
        prompt_template_dict = None

    def encode_prompt(prompt, negative_prompt, text_encoder):
        text_inputs = text_encoder.text2tokens(prompt, prompt_template=prompt_template_dict)
        prompt_outputs = text_encoder.encode(text_inputs, prompt_template=prompt_template_dict, device=device)
        prompt_embeds = prompt_outputs.hidden_state
        attention_mask = prompt_outputs.attention_mask

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            bs_embed, seq_len = attention_mask.shape
            attention_mask = attention_mask.repeat(1, 1)
            attention_mask = attention_mask.view(
                bs_embed * 1, seq_len
            )

        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        batch_size = 1
        num_videos_per_prompt = 1
        if cfg_scale > 1.0:
            print('encoding negative prompt')
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = text_encoder.text2tokens(
                uncond_tokens,
                prompt_template=prompt_template_dict
            )

            negative_prompt_outputs = text_encoder.encode(
                uncond_input,
                prompt_template=prompt_template_dict,
                device=device
            )

            negative_prompt_embeds = negative_prompt_outputs.hidden_state
            negative_attention_mask = negative_prompt_outputs.attention_mask

            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(
                    1, num_videos_per_prompt
                )
                negative_attention_mask = negative_attention_mask.view(
                    batch_size * num_videos_per_prompt, seq_len
                )

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=text_encoder.dtype, device=device
            )

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, -1
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt, 1
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, seq_len, -1
                )
        else:
            negative_prompt_embeds = None
            negative_attention_mask = None

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )
    
    # encode prompt
    text_encoder_1.to(device)
    prompt_embeds, negative_prompt_embeds, attention_mask, negative_attention_mask = encode_prompt(prompt, negative_prompt, text_encoder_1)
    if text_encoder_2 is not None:
        text_encoder_2.to(device)
        prompt_embeds_2, negative_prompt_embeds_2, attention_mask_2, negative_attention_mask_2 = encode_prompt(prompt, negative_prompt, text_encoder_2)

    # offload text encoders
    text_encoder_1.to(offload_device)
    text_encoder_2.to(offload_device)

    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "attention_mask": attention_mask,
        "negative_attention_mask": negative_attention_mask,
        "prompt_embeds_2": prompt_embeds_2,
        "negative_prompt_embeds_2": negative_prompt_embeds_2,
        "attention_mask_2": attention_mask_2,
        "negative_attention_mask_2": negative_attention_mask_2
    }

def get_rotary_pos_embed_hyvideo(transformer, latent_video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2
        rope_theta = 225
        patch_size = transformer.patch_size
        rope_dim_list = transformer.rope_dim_list
        hidden_size = transformer.hidden_size
        heads_num = transformer.heads_num
        head_dim = hidden_size // heads_num

        # 884
        latents_size = [latent_video_length, height // 8, width // 8]

        if isinstance(patch_size, int):
            assert all(s % patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // patch_size for s in latents_size]
        elif isinstance(patch_size, list):
            assert all(
                s % patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [
                s // patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis

        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

def load_model(model_path, device, offload_device, base_dtype):
    in_channels = out_channels = 16
    factor_kwargs = {"device": device, "dtype": base_dtype}

    HUNYUAN_VIDEO_CONFIG = {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    }
    
    with init_empty_weights():
        transformer = HYVideoDiffusionTransformer(
            in_channels=in_channels,
            out_channels=out_channels,
            attention_mode='sageattn_varlen',
            main_device=device,
            offload_device=offload_device,
            **HUNYUAN_VIDEO_CONFIG,
            **factor_kwargs
        )

    transformer.eval()
    
    sd = load_torch_file(model_path, device=offload_device, safe_load=True)

    dtype = torch.float8_e4m3fn
    params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
    for name, param in transformer.named_parameters():
        dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
        set_module_tensor_to_device(transformer, name, device=device, dtype=dtype_to_use, value=sd[name])

    # convert scaled
    convert_fp8_linear(transformer, base_dtype)

    pipeline = HunyuanVideoPipeline(
        transformer=transformer,
        scheduler=FlowMatchDiscreteScheduler(
            shift=9.0,
            reverse=True,
            solver="euler",
        ),
        progress_bar_config=None,
        base_dtype=base_dtype
    )

    return pipeline

def sample_video(pipeline, text_embeddings, latents, device, offload_device, width, height, num_frames, steps, embedded_guidance_scale, cfg_scale, flow_shift, seed, denoise_strength):
    cfg = cfg_scale
    cfg_start_percent = CFG_SCALE_START
    cfg_end_percent = CFG_SCALE_END

    generator = torch.Generator(device=torch.device("cpu")).manual_seed(seed)

    target_height = align_to(height, 16)
    target_width = align_to(width, 16)

    freqs_cos, freqs_sin = get_rotary_pos_embed_hyvideo(
        pipeline.transformer, num_frames, target_height, target_width
    )
    n_tokens = freqs_cos.shape[0]

    pipeline.scheduler.shift = flow_shift

    # enable swapping
    for name, param in pipeline.transformer.named_parameters():
        if "single" not in name and "double" not in name:
            param.data = param.data.to(device)
    pipeline.transformer.block_swap(
        SWAP_DOUBLE_BLOCKS - 1,
        SWAP_SINGLE_BLOCKS - 1,
        offload_txt_in = OFFLOAD_TXT_IN,
        offload_img_in = OFFLOAD_IMG_IN,
    )

    gc.collect()
    soft_empty_cache()

    pipeline.transformer.to(device)

    out_latents = pipeline(
        num_inference_steps=steps,
        height=target_height,
        width=target_width,
        video_length=num_frames,
        guidance_scale=cfg,
        cfg_start_percent=cfg_start_percent,
        cfg_end_percent=cfg_end_percent,
        embedded_guidance_scale=embedded_guidance_scale,
        latents=latents,
        denoise_strength=denoise_strength,
        prompt_embed_dict=text_embeddings,
        generator=generator,
        freqs_cis=(freqs_cos, freqs_sin),
        n_tokens=n_tokens,
        # stg_mode=None,
        # stg_block_idx=-1,
        # stg_scale=0.0,
        # stg_start_percent=0.0,
        # stg_end_percent=1.0,
    )

    pipeline.transformer.to(offload_device)
    gc.collect()

    return out_latents

class GenerativePipeline:
    def __init__(self, *args, **kwargs):
        # test
        print(f'{args}, {kwargs}')

        # Define devices for each  model in the pipeline
        # and then load them passing the device itself as the offload device
        # effectively disabling offloading.
        offload_device = torch.device("cpu")
        vae_device = torch.device("cuda:0")
        txt_encoder_device = torch.device("cuda:0")
        model_device = torch.device("cuda:0")
        sample_device = torch.device("cuda:0")
        
        # # 1.a. Load input video
        # frames = load_video_frames(INPUT_FRAMES_PATH)
        # print(f"Loaded {len(frames)} frames.")

        # # 1.b. Encode input video
        # vae = load_vae(VAE_PATH, vae_device, offload_device, "bf16")
        # latents = encode_video(vae, frames, vae_device, offload_device)

        # 2. Encode prompt
        text_encoder, text_encoder_2 = load_text_encoder("fp16", txt_encoder_device, offload_device)
        text_embeddings = encode_text(text_encoder, text_encoder_2, txt_encoder_device, offload_device, PROMPT, NEGATIVE_PROMPT, CFG_SCALE)

        # # # Test
        # # new_frames = decode_video(vae, latents, vae_device, offload_device)
        # # print("Decoded latents into frames. (test mode)")

        # 3. Sample
        latents = None #temp
        pipeline = load_model(MODEL_PATH, model_device, offload_device, torch.bfloat16)        
        new_latents = sample_video(pipeline, text_embeddings, latents, sample_device, offload_device, WIDTH, HEIGHT, NUM_FRAMES, STEPS, EMBEDDED_GUIDANCE_SCALE, CFG_SCALE, FLOW_SHIFT, SEED, DENOISE_STRENGTH)

        # 4. Decode
        vae = load_vae(VAE_PATH, vae_device, offload_device, "bf16")
        new_frames = decode_video(vae, new_latents, vae_device, offload_device)

        # 5. Combine frames and save
        save_video_ffmpeg(new_frames, OUTPUT_VIDEO, fps=24)
