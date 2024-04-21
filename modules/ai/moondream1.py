from huggingface_hub import snapshot_download
from moondream import VisionEncoder, TextModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

def vision_moondream1(image, text):
    # Initialize the models outside the function
    model_path = snapshot_download("vikhyatk/moondream1")
    vision_encoder = VisionEncoder(model_path)
    text_model = TextModel(model_path)

    image_embeds = vision_encoder(image)
    result = text_model.answer_question(image_embeds, text)

    return result

def vision_moondream2(image, question):
    # Download the model and tokenizer from Hugging Face Hub
    model_id = "vikhyatk/moondream2"
    revision = "2024-03-13"
    # Ensure the model and tokenizer are downloaded only once, if not already available
    model_path = snapshot_download(model_id, revision=revision)
    
    # Initialize the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    
    enc_image = model.encode_image(image)
    
    # Answer the question based on the encoded image
    answer = model.answer_question(enc_image, question, tokenizer)
    
    return answer