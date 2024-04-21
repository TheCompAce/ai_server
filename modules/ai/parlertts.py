import io
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

def generate_speech(text, description):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")
    
    # Process inputs
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    
    # Generate speech
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    
    # Convert audio array to byte stream for playback or storage
    audio_io = io.BytesIO()
    sf.write(audio_io, audio_arr, model.config.sampling_rate, format='wav')
    audio_io.seek(0)
    
    return audio_io
