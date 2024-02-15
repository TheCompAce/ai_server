from huggingface_hub import snapshot_download
from moondream import VisionEncoder, TextModel

def vision_moondream1(image, text):
    # Initialize the models outside the function
    model_path = snapshot_download("vikhyatk/moondream1")
    vision_encoder = VisionEncoder(model_path)
    text_model = TextModel(model_path)

    image_embeds = vision_encoder(image)
    result = text_model.answer_question(image_embeds, text)

    return result