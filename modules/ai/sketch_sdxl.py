import io
from flask import request
import torch
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from controlnet_aux.pidi import PidiNetDetector


def image_sketch_sdxl(image, prompt, negative_prompt = ''):
    # Initialize the sketch generation pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
    ).to(device)
    euler_a = EulerAncestralDiscreteScheduler.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16", 
    ).to(device)
    # pipe.enable_xformers_memory_efficient_attention()  # Commented out due to compatibility issues
    pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators").to(device)


    # Prepare and condition the image
    image = pidinet(
        image, detect_resolution=1024, image_resolution=1024, apply_filter=True
    )

    with torch.no_grad():
        gen_images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=30,
            adapter_conditioning_scale=0.9,
            guidance_scale=7.5,
        ).images[0]

    # Convert PIL image to byte array
    img_io = io.BytesIO()
    gen_images.save(img_io, 'PNG')
    img_io.seek(0)

    return img_io