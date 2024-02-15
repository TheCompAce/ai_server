import io
from datasets import load_dataset
from flask import request
import torch
from transformers import pipeline
import soundfile as sf

def speak(text):
    # Initialize the speech synthesis pipeline
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

    # Load speaker embeddings
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

    # Convert audio array to byte stream
    audio_io = io.BytesIO()
    sf.write(audio_io, speech["audio"], samplerate=speech["sampling_rate"], format='wav')
    audio_io.seek(0)

    return audio_io
