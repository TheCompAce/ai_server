import io
from flask import request
import torch
import soundfile as sf
from transformers import AutoProcessor, BarkModel

# Define the default voice preset
DEFAULT_VOICE_PRESET = "v2/en_speaker_6"

# Function to synthesize speech with an optional voice preset
def speak(text, voice_preset=None):
    if not voice_preset:
        voice_preset = DEFAULT_VOICE_PRESET  # Use default if no voice preset is specified

    # Initialize the Bark processor and model
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")

    # Process the text input with the specified voice preset
    inputs = processor(text, voice_preset=voice_preset)

    # Generate the audio array from the model
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()  # Squeeze to convert from tensor to numpy array

    # Convert the audio array to a byte stream
    audio_io = io.BytesIO()
    sf.write(audio_io, audio_array, samplerate=24000, format='wav')  # Writing to audio stream with 24kHz sample rate
    audio_io.seek(0)  # Reset the stream to the beginning

    return audio_io  # Return the audio byte stream
