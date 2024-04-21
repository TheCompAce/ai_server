import io
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, resize
from diffusers import AutoPipelineForInpainting

def inpaint_image_with_mask(image_stream, mask_stream, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the inpainting pipeline
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)

    # Load the image and mask from streams and convert them
    original_image = Image.open(image_stream).convert("RGB")
    mask_image_stream = Image.open(mask_stream).convert("RGB")

    # Store the original dimensions
    original_dimensions = original_image.size

    # Resize image and mask for the model input
    image = original_image.resize((1024, 1024))
    mask_image = mask_image_stream.resize((1024, 1024))

    # Perform the inpainting
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=8.0,
        num_inference_steps=20,
        strength=0.99,
        generator=torch.Generator(device=device).manual_seed(0)
    )

    # Convert the tensor output to PIL Image
    result_image = to_pil_image(result.images[0])

    # Resize the inpainted image back to the original dimensions
    result_image_resized = resize(result_image, original_dimensions)

    # Convert PIL image to byte array for web transmission
    img_io = io.BytesIO()
    result_image_resized.save(img_io, 'PNG')
    img_io.seek(0)

    return img_io
