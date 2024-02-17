import torch
from diffusers import StableDiffusionXLPipeline, KDPM2AncestralDiscreteScheduler, AutoencoderKL

def text_to_image_proteus(prompt, negative_prompt="", width=1024, height=1024, guidance_scale=7, num_inference_steps=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the VAE component with float16 for efficiency
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    ).to(device)

    # Configure the pipeline with the Proteus model, VAE, and scheduler
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "dataautogpt3/ProteusV0.3", 
        vae=vae,
        torch_dtype=torch.float16
    ).to(device)
    
    # Use the KDPM2AncestralDiscreteScheduler configured from the pipeline's config
    pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # Generate the image with the provided parameters
    image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]
    
    return image