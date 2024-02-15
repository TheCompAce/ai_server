
from datetime import datetime
import io

import torch
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS


def text_to_image_wuerstchen(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # At the beginning of your server.py script, initialize the wuerstchen pipeline
    wuerstchen_pipeline = AutoPipelineForText2Image.from_pretrained("warp-ai/wuerstchen", torch_dtype=torch.float16).to(device)

    # Generate images using the wuerstchen pipeline
    images = wuerstchen_pipeline(
        prompt,
        height=1024,
        width=1536,
        prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
        prior_guidance_scale=4.0,
        num_images_per_prompt=1,
    ).images

    # For simplicity, let's send only the first image
    img_io = io.BytesIO()
    images[0].save(img_io, 'PNG')
    img_io.seek(0)

    return img_io
