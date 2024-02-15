# Initialize the text-to-image pipeline
from datetime import datetime
import io
import os
import torch

from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from flask import request, send_file

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def text_to_image_sd(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16",safety_checker=None)  
    pipe.to(device)

    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

    image.seek(0)

    return image

    
