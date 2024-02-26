from io import BytesIO
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline, AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from PIL import Image, ImageSequence

def text_to_gif_animateLCM(prompt, negative_prompt="", num_frames=16, guidance_scale=2.0, num_inference_steps=6):
    """
    Generates an animated GIF based on a given prompt using Animate Diff Pipeline.
    """
    adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

    pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    pipe.set_adapters(["lcm-lora"], [0.8])

    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(0),
    )
    frames = output.frames[0]

    # Convert frames to a GIF and save it to a BytesIO object
    gif_bytes = frames_to_gif_bytes(frames)
    
    return gif_bytes

def frames_to_gif_bytes(frames, duration=100):
    """
    Save frames to a GIF file stored in a BytesIO object.

    Args:
    - frames: A list of PIL.Image objects.
    - duration: Duration for each frame in milliseconds.

    Returns:
    - A BytesIO object containing the GIF.
    """
    gif_bytes = BytesIO()
    frames[0].save(gif_bytes, format='GIF', save_all=True, append_images=frames[1:], duration=duration, loop=0)
    gif_bytes.seek(0)  # Rewind to the start of the BytesIO object
    return gif_bytes