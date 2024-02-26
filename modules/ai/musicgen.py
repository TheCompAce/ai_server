import io
from flask import request
import torch
from transformers import pipeline
import soundfile as sf

def generate_music(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize the music generation pipeline
    synthesiser = pipeline("text-to-audio", "facebook/musicgen-stereo-large", device=device, torch_dtype=torch.float16)

    music = synthesiser(prompt, forward_params={"max_new_tokens": 512})
    
    # Convert audio array to byte stream
    audio_io = io.BytesIO()
    sf.write(audio_io, music["audio"][0].T, music["sampling_rate"], format='wav')
    audio_io.seek(0)

    return audio_io