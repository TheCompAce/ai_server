
# AI Server

A comprehensive AI service framework that integrates various AI functionalities, including image processing, speech-to-text, text-to-speech, and more, leveraging a Flask server as its backbone.

## Overview

This project is a Flask-based server designed to handle multiple AI-related tasks such as image generation, depth estimation, object detection, image variation, sketch generation, music generation, text-to-speech, and speech-to-text conversion. It uses a modular architecture, where `server.py` serves as the entry point and `ai_main.py` directs specific tasks to their respective modules.

## Features

- **Image Processing**: Generate images from text, detect objects, create image variations, convert images to sketches, and estimate image depth.
- **Speech Processing**: Convert text to speech, and speech to text.
- **Music Generation**: Create music based on text prompts.
- **LLM Queries**: Interact with large language models to process and answer queries.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/TheCompAce/ai_server
   cd ai_server
   ```

2. **Install Dependencies**
   Ensure you have Python 3.8+ installed. Then, install the required Python packages.
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**
   Set up necessary environment variables, such as `OPENAI_API_KEY`, for modules that require external API access.

4. **Running the Server**
   ```bash
   python server.py
   ```

## Usage

The server exposes various endpoints for interacting with AI models:

- **/vision**: Process images for various tasks.
- **/ask** and **/ask/json**: Query a language model.
- **/music**: Generate music from a prompt.
- **/speak**: Convert text to speech.
- **/hear**: Convert speech to text.
- **/image**: Generate images from text prompts.

### Example Requests

```python
import requests

# Generate an image from text
response = requests.post('http://localhost:5000/image', data={'prompt': 'A scenic view of the mountains'})
```

## Modules Overview

- `ai_main.py` directs tasks to specific AI models, such as `text_to_image`, `speech_to_text`, etc.
- Modules like `moondream1.py`, `openai.py`, `sdc.py`, and `whisper.py` are used for specific AI tasks, interfacing with models from Hugging Face, OpenAI, and others.

## Acknowledgments

This project integrates various AI technologies and models, including OpenAI's GPT, Stable Diffusion for image generation, Whisper for speech-to-text, and more. Special thanks to the creators and maintainers of these models and the Flask framework for making web server integration straightforward.
