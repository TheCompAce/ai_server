import os
import io
import sys
import signal

from datetime import datetime
import json
from modules.ai_main import text_to_gif, text_to_image, image_to_depth, detect_image, variation_image, sketch_image, music_generate, text_to_speech, ask_llm, get_vision, ask_llm_json, speech_to_text, ask_llm_embed, text_to_sound, transform_image, inpaint_image, parlor_text_to_speech
from modules.database import DatabaseManager


from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.security import check_password_hash, generate_password_hash
from flask_cors import CORS



db_manager = DatabaseManager('database.json')  # Adjust path as necessary
db_manager.load_config()
db_manager.build()

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
0
# Clear the folder when the application starts
clear_generated_images_folder()


# app = Flask(__name__)
app = Flask(__name__, static_folder='public')
CORS(app, resources={r"/": {"origins": "*"}})  # Adjust origins as needed


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

@app.route('/speak/parlor', methods=['POST'])
def speak_parlor():
    data = request.get_json()
    prompt = data.get('prompt', '')
    description = data.get('description', '')

    # You may want to add error handling or validation here
    
    audio_io = parlor_text_to_speech(prompt, description, settings)

    return send_file(audio_io, mimetype='audio/wav', as_attachment=True, download_name='parlor_speech.wav')


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
    settings = get_settings()
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

@app.route('/gif', methods=['POST'])
def generate_gif_from_text():
    
    data = request.get_json()  # Use get_json() to parse JSON data
    
    if not data or 'prompt' not in data:
        return "Invalid request", 400  # Return a Bad Request error if no prompt
    
    prompt = data['prompt']
    # Process the prompt to gen
    
    # Assuming text_to_image returns a PIL Image object
    gif_io = text_to_gif(prompt, settings)  # Assuming this returns a BytesIO object
    
    # Generate a date/time formatted image name for the download
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_name = f"{timestamp}.gif"
    
    return send_file(
        gif_io,
        mimetype='image/gif',
        as_attachment=True,
        download_name=gif_name,
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
        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name=tranform_image_name)
    else:
        return jsonify({"error": "Invalid request"}), 400
    
@app.route('/image/inpaint', methods=['POST'])
def image_inpaint():
    settings = get_settings()  # Retrieve settings if they are used within inpaint_image
    
    # Ensure both 'image' and 'mask' are present in the request files
    if 'image' not in request.files or 'mask' not in request.files:
        return jsonify({"error": "No image or mask file provided"}), 400

    image_file = request.files.get('image')
    mask_file = request.files.get('mask')

    if not image_file:
        print("Missing image file")
        return "Missing image file", 400
    
    if not mask_file:
        print("Missing mask file")
        return "Missing mask file", 400
    
    # Optional: You can add checks here to ensure the files are of allowed types
    
    prompt = request.form.get('prompt', '')  # Retrieve the prompt from form data
    
    # Call the inpaint_image function
    inpainted_image = inpaint_image(image_file.stream, mask_file.stream, prompt, settings)
    
    # Convert the inpainted PIL Image to a byte array for response
    img_io = io.BytesIO()
    inpainted_image.save(img_io, 'PNG', quality=70)
    img_io.seek(0)
    
    # You can add logic here to generate a dynamic filename based on the request
    filename = "inpainted_image.png"
    
    return send_file(
        img_io,
        mimetype='image/png',
        as_attachment=True,
        download_name=filename
    )

# Endpoint to process the remove background
@app.route('/image/removebg', methods=['POST'])
def image_removebg():
    settings = get_settings()
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    range_val = float(request.form.get('range', '.2')) / 100
    if file and allowed_file(file.filename):
        # Assuming 'file' is the uploaded file object from Flask request
        file_stream = file.stream  # Get the file stream
        image = Image.open(file_stream)  # Open the image directly from the stream

        # Generate a date/time formatted image name for the download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        remove_bg_name = f"removebg_{timestamp}.png"

        remove_bg_image = image_to_depth(file, settings)

        # now take the red channel and use it for the alpha of the image passed in
        # We need to base this off our "range_val" (between 0 to 1) based on the remove_bg_image.chennel Level
        # We also need to make sure the image is RGBA
        image = image.convert('RGBA')
        remove_bg_image = remove_bg_image.convert('RGBA')

        # Create a new image for the alpha channel with the same dimensions
        alpha = Image.new('L', image.size, 255)  # Start with a fully opaque alpha channel

        # Apply the specified range to adjust the alpha channel based on the depth image
        depth_pixels = remove_bg_image.load()
        alpha_pixels = [image.width * y + x for y in range(image.height) for x in range(image.width)]

        # Iterate through the image, pixel by pixel        
        for y in range(image.height):
            for x in range(image.width):
                # Scale the depth pixel value based on the range_val
                if depth_pixels[x, y][0] < (range_val * 255):
                    alpha_pixels[y * image.width + x] = 0
                else:
                    alpha_pixels[y * image.width + x] = 255

        alpha.putdata(alpha_pixels)

        # Combine the original image with the new alpha channel
        image.putalpha(alpha)

        # Convert PIL image to byte array with alpha
        img_io = io.BytesIO()
        image.save(img_io, 'PNG', quality=70)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name=remove_bg_name)
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

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not all([username, email, password]):
        return jsonify({"error": "Missing username, email, or password"}), 400

    # Check if user already exists
    existing_user = db_manager.read_user_by_email(email)
    if existing_user:
        return jsonify({"error": "User already exists"}), 400

    # Hash password
    hashed_password = generate_password_hash(password, method='sha256')

    # Create user
    user_id = db_manager.create_user({'username': username, 'email': email, 'password_hash': hashed_password})
    
    return jsonify({"message": "User created successfully", "user_id": user_id}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not all([email, password]):
        return jsonify({"error": "Missing email or password"}), 400
    
    # Fetch user by email
    user = db_manager.read_user_by_email(email)
    
    if user and check_password_hash(user['password_hash'], password):
        # Successfully authenticated
        # Here, implement session creation or token generation as per your requirement
        return jsonify({"message": "Login successful", "user_id": user['id']}), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    app.run(debug=True)

@app.teardown_appcontext
def shutdown_session(exception=None):
    db_manager.close_connections()

def signal_handler(sig, frame):
    print('Shutting down gracefully...')
    db_manager.close_connections()
    sys.exit(0)