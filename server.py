from datetime import datetime
import json
from modules.ai_main import text_to_image, image_to_depth, detect_image, variation_image, sketch_image, music_generate, text_to_speech, ask_llm, get_vision, ask_llm_json, speech_to_text, ask_llm_embed, text_to_sound, transform_image

from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, send_file

import io

import os

settings ={}

def get_settings():
    if os.path.exists('settings.json'):
        with open('settings.json', 'r') as file:
            return json.load(file)

settings = get_settings()

# Function to clear the generated_images folder
def clear_generated_images_folder():
    folder = 'generated_images'
    if os.path.exists(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

# Clear the folder when the application starts
clear_generated_images_folder()

app = Flask(__name__, static_folder='public')

# Serve HTML, JS, CSS, and image files from the static folder
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory(app.static_folder, path)


# Endpoint to process the image and text
@app.route('/vision', methods=['POST'])
def vision():
    settings = get_settings()
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    text = request.form.get('text', '')
    if file and allowed_file(file.filename):
        # Assuming 'file' is the uploaded file object from Flask request
        file_stream = file.stream  # Get the file stream
        image = Image.open(file_stream)  # Open the image directly from the stream

        
        result = get_vision(image, text, settings)

        return jsonify({"result": result})
    else:
        return jsonify({"error": "Invalid request"}), 400

# Endpoint to process a question for the LLM
@app.route('/ask', methods=['POST'])
def ask():
    settings = get_settings()
    data = request.json
    system = data.get('system', '')
    user = data.get('user', '')

    response = ask_llm(system, user, settings)
    return jsonify({"response": response})

# Endpoint to process a question for the LLM
@app.route('/ask/json', methods=['POST'])
def ask_json():
    settings = get_settings()
    data = request.json
    system = data.get('system', '')
    user = data.get('user', '')

    response = ask_llm_json(system, user, settings)
    return jsonify({"response": response})

# Endpoint to process a question for the LLM
@app.route('/ask/embed', methods=['POST'])
def ask_embed():
    settings = get_settings()
    data = request.json
    text = data.get('text', '')

    response = ask_llm_embed(text, settings)
    return jsonify({"response": response})

@app.route('/music', methods=['POST'])
def generate_music():
    prompt = request.form['prompt']

    audio_io = music_generate(prompt)

    return send_file(audio_io, mimetype='audio/wav', as_attachment=True, download_name='music.wav')

@app.route('/speak', methods=['POST'])
def speak():
    settings = get_settings()
    data = request.json
    text = data.get('prompt', '')
    audio_io = text_to_speech(text, settings)

    return send_file(audio_io, mimetype='audio/mp3', as_attachment=True, download_name='generated_speech.mp3')

@app.route('/sound', methods=['POST'])
def sound_effect():
    settings = get_settings()
    data = request.json
    text = data.get('prompt', '')
    audio_io = text_to_sound(text, settings)

    return send_file(audio_io, mimetype='audio/mp3', as_attachment=True, download_name='generated_sound.mp3')

@app.route('/hear', methods=['POST'])
def hear():
    settings = get_settings()
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    audio_bytes = audio_file.read()

    response = speech_to_text(audio_bytes, settings)

    return jsonify({"response": response})

@app.route('/image', methods=['POST'])
def generate_image_from_text():
    
    data = request.get_json()  # Use get_json() to parse JSON data
    
    if not data or 'prompt' not in data:
        return "Invalid request", 400  # Return a Bad Request error if no prompt
    
    prompt = data['prompt']
    # Process the prompt to gen
    
    # Assuming text_to_image returns a PIL Image object
    image = text_to_image(prompt, settings)
    
    # Convert PIL image to byte array
    img_io = io.BytesIO()
    image.save(img_io, 'PNG', quality=70)
    img_io.seek(0)
    
    # Generate a date/time formatted image name for the download
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f"{timestamp}.png"
    
    return send_file(
        img_io,
        mimetype='image/png',
        as_attachment=True,
        download_name=image_name,
    )

@app.route('/image/depth', methods=['POST'])
def depth():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    
    depth_image = image_to_depth(image_file, settings)

    # Convert depth_image to byte array
    img_io = io.BytesIO()
    depth_image.save(img_io, 'PNG', quality=70)
    img_io.seek(0)

    # Generate a date/time formatted image name for the download
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    depth_image_name = f"depth_{timestamp}.png"
    
    return send_file(
        img_io,
        mimetype='image/png',
        as_attachment=True,
        download_name=depth_image_name
    )

@app.route('/image/detect', methods=['POST'])
def image_detect():

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    detections = detect_image(file, settings)
    return jsonify(detections)

# Endpoint to process the image and text
@app.route('/image/transform', methods=['POST'])
def image_transform():
    settings = get_settings()
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    text = request.form.get('text', '')
    if file and allowed_file(file.filename):
        # Assuming 'file' is the uploaded file object from Flask request
        file_stream = file.stream  # Get the file stream
        image = Image.open(file_stream)  # Open the image directly from the stream

        # Generate a date/time formatted image name for the download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tranform_image_name = f"tranform_{timestamp}.png"
        img_io = transform_image(image, text, settings)
        print(img_io)
        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name=tranform_image_name)
    else:
        return jsonify({"error": "Invalid request"}), 400

@app.route('/image/variation', methods=['POST'])
def image_variation():
    settings = get_settings()

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    img_io = variation_image(file, settings)
        
    # Generate a date/time formatted image name for the download
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    variation_image_name = f"variation_{timestamp}.png"

    return send_file(img_io, mimetype='image/png', as_attachment=True, download_name=variation_image_name)

@app.route('/image/sketch', methods=['POST'])
def image_sketch():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    image = Image.open(file.stream)

    # Process the image through the sketch pipeline
    prompt = request.form.get('prompt', '')
    negative_prompt = request.form.get('negative_prompt', '')

    img_io = sketch_image(image, prompt, negative_prompt)

    # Generate a date/time formatted image name for the download
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sketch_image_name = f"variation_{timestamp}.png"

    return send_file(img_io, mimetype='image/png', as_attachment=True, download_name=sketch_image_name)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    app.run(debug=True)
