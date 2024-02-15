import io
import torch
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionImageVariationPipeline

def image_variation_sd(file):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the Stable Diffusion pipeline
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers",
        revision="v2.0",
        torch_dtype=torch.float16,
        safety_checker = None,
        requires_safety_checker = False
    ).to(device)

    image = Image.open(file.stream)

    # Transform the image
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        ),
        transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]
        ),
    ])
    inp = tform(image).to(device).unsqueeze(0)

    # Generate image variation
    with torch.no_grad():
        out = sd_pipe(inp, guidance_scale=1)
    result_image = out["images"][0]

    # Convert PIL image to byte array
    img_io = io.BytesIO()
    result_image.save(img_io, 'PNG')
    img_io.seek(0)

    return img_io