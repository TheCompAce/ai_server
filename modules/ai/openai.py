import base64
import requests
import os
import io
from PIL import Image

from openai import BadRequestError, OpenAI

def text_to_image_openai(prompt, settings):
    # Read OpenAI API key from environment variable
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        return "OpenAI API key not found", 400

    # OpenAI API URL for image generation
    url = "https://api.openai.com/v1/images/generations"

    # Headers including the Authorization with your OpenAI API key
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    model = settings.get("dalle_model", "dall-e-3")

    # Request data
    data = {
        "model": model,
        "prompt": prompt,
        "response_format": "b64_json",
        "n": 1,
        "size": "1024x1024"
    }

    # Send POST request to OpenAI API
    response = requests.post(url, headers=headers, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        # Extract image URL from the response
        response_data = response.json()["data"][0]["b64_json"]

        # Decode the Base64 string
        image_data = base64.b64decode(response_data)

        # Convert the binary data to an image
        return Image.open(io.BytesIO(image_data))
    else:
        return "Error calling OpenAI API: " + response.text, response.status_code

def image_variation_openai(file_path, settings):
    # Read OpenAI API key from environment variable
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        return {
            "error": {
                "code": "401",
                "message": "OpenAI API key not found",
                "param": "Authorization",
                "type": "authentication_error"
            }
        }, 401

    try:
        # Initialize a variable to hold a potentially converted file stream
        converted_image_stream = None

        # Verify image format and size, and convert if necessary
        with Image.open(file_path) as img:
            if img.format in ['JPEG', 'JPG']:
                # Convert JPEG/JPG to PNG
                bytes_io = io.BytesIO()
                img.save(bytes_io, format='PNG')
                bytes_io.seek(0)
                converted_image_stream = bytes_io
                img_byte_size = bytes_io.getbuffer().nbytes
            else:
                if img.format != 'PNG':
                    raise ValueError("Uploaded image must be a PNG.")
                img_byte_size = os.path.getsize(file_path)

            if img_byte_size > 4 * 1024 * 1024:  # 4 MB in bytes
                raise ValueError("Image must be less than 4 MB.")

        # If checks pass, proceed with API call
        client = OpenAI(api_key=openai_api_key)
        if converted_image_stream:
            image_data = converted_image_stream.read()
        else:
            with open(file_path, 'rb') as image_file:
                image_data = image_file.read()

        response = client.images.create_variation(
            image=image_data,
            n=settings.get('n', 2),
            size=settings.get('size', "1024x1024")
        )

        # Correctly handle response to get the image URL
        # This part assumes you have a valid image URL in response
        if hasattr(response, 'data') and response.data:
            image_url = response.data[0].url  # Adjust based on actual structure

            # Download the image from the URL
            response = requests.get(image_url)
            if response.status_code == 200:
                return io.BytesIO(response.content)
            else:
                raise ValueError("Failed to download the image.")
        else:
            raise ValueError("Invalid response structure.")
    except BadRequestError as e:
        print(str(e))
        return {"error": str(e)}, 400
    except Exception as e:
        print(str(e))
        return {
            "error": {
                "code": None,
                "message": str(e),
                "param": None,
                "type": "invalid_request_error"
            }
        }, 400
    
def speak(text):
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        return "OpenAI API key not found", 400

    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    data = {
        "model": "tts-1",
        "input": text,
        "voice": "alloy",
        "response_format": "mp3"  # Ensure this is correctly spelled
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        # Since response.content is base64, decode it directly
        print(response.raw)
        # audio_data = base64.b64decode(response.content)
        audio_data = response.content
        
        audio_io = io.BytesIO(audio_data)
        audio_io.seek(0)
        

        return audio_io
    else:
        return "Error calling OpenAI API: " + response.text, response.status_code
    
import requests
import os
import json

def ask(system, user, settings):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        return "OpenAI API key not found", 400

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    data = {
        "model": settings.get("openai_model", "gpt-3.5-turbo"),
        "messages": messages  # conversation should be a list of message dicts
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:

        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error calling OpenAI API: {response.text}", response.status_code

def ask_json(system, user, settings):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        return "OpenAI API key not found", 400

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    data = {
        "model": settings.get("openai_json_model", "gpt-3.5-turbo-1106"),
        "response_format": { "type": "json_object" },
        "messages": messages  # conversation should be a list of message dicts
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:

        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error calling OpenAI API: {response.text}", response.status_code

def vision_openai(image, prompt, settings):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        return "OpenAI API key not found", 400

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    # Convert the PIL Image object to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # Save image to buffer in PNG format
    image_bytes = buffer.getvalue()  # Get bytes from buffer

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": "data:image/png;base64," + image_base64}
        ]}
    ]

    data = {
        "model": settings.get("openai_vision_model", "gpt-4-vision-preview"),
        "messages": messages  # conversation should be a list of message dicts
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:

        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error calling OpenAI API: {response.text}", response.status_code
    

def create_embeddings(input_text, model_id="text-embedding-ada-002", encoding_format="float"):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        return "OpenAI API key not found", 400

    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": input_text,
        "model": model_id,
        "encoding_format": encoding_format
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        # Extract just the embedding vector from the response
        embedding_data = response.json()["data"]
        # Assuming there's only one input and hence one embedding in the response
        embedding_vector = embedding_data[0]["embedding"] if embedding_data else []
        return embedding_vector
    else:
        return f"Error calling OpenAI API: {response.text}", response.status_code
