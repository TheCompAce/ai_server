import io
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def generate_image_with_lighting(prompt, num_inference_steps=4, guidance_scale=0):
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"

    # Load the UNet model
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float8_e4m3fn)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))

    # Initialize the pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)
    
    # Configure the scheduler for "trailing" timesteps
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # Generate an image based on some input; the function returns a PIL Image object
    pil_image = generate_image_with_lighting(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    
    # Convert PIL Image to BytesIO object
    img_io = io.BytesIO()
    pil_image.save(img_io, 'PNG', quality=70)
    img_io.seek(0)
    
    # Send the image as a response
    return img_io