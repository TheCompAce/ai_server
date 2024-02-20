import io
from diffusers import AudioLDM2Pipeline
import soundfile as sf
import torch

def text_to_sound_audioldm(prompt):
    """
    Generates sound based on a text description using the AudioLDM2Pipeline model.

    :param prompt: A string description of the desired sound.
    :return: io.BytesIO object containing the generated audio in WAV format.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = "cvssp/audioldm2"

    # Initialize the pipeline
    AudioLDM2_pipe = AudioLDM2Pipeline.from_pretrained(repo_id)
    AudioLDM2_pipe = AudioLDM2_pipe.to(device)

    # Generate audio based on the prompt
    audio = AudioLDM2_pipe(prompt, num_inference_steps=200, audio_length_in_s=10.0).audios[0]

    # Convert the generated audio to a byte stream
    audio_io = io.BytesIO()
    sf.write(audio_io, audio, 16000, format='wav')  # Corrected line
    audio_io.seek(0)  # Rewind the buffer to the beginning
    
    return audio_io
