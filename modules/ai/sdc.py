import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

def text_to_image_sdc(prompt, negative_prompt="", num_images_per_prompt=2, guidance_scale=4.0, num_inference_steps_prior=20, num_inference_steps_decoder=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the prior and decoder models, adjusting for the appropriate torch_dtype for efficiency
    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to(device)
    decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=torch.float16).to(device)

    # Generate image embeddings using the prior model
    prior_output = prior(
        prompt=prompt,
        height=1024,
        width=1024,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        num_inference_steps=num_inference_steps_prior
    )

    # Decode the embeddings into images using the decoder model
    decoder_output = decoder(
        image_embeddings=prior_output.image_embeddings.half(),
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,  # Typically, guidance scale for decoder is set to 0.0 as per best practices
        output_type="pil",
        num_inference_steps=num_inference_steps_decoder
    ).images

    return decoder_output