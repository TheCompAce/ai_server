# Initialize the text-to-image pipeline
from diffusers import StableDiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# It's good practice to set these configurations, but they're not directly related to the error
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def text_to_image_sd15(prompt):
    # Ensure model_id is correct and model supports expected output
    model_id = "runwayml/stable-diffusion-v1-5"
    try:
        
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        pipe = pipe.to(device)

        image = pipe(prompt).images[0]  
        
        return image 
        
    except Exception as e:
        # Handle other exceptions gracefully
        print(f"An error occurred: {e}")
        return None
