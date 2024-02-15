from modules.ai import sdlx_turbo
from modules.ai import sd15
from modules.ai import sd_turbo
from modules.ai import sdc
from modules.ai import wuerstchen
from modules.ai import sketch_sdxl
from modules.ai import dpt
from modules.ai import detr
from modules.ai import sd_variations
from modules.ai import musicgen
from modules.ai import speecht5_tts
from modules.ai import tinyllama
from modules.ai import moondream1
from modules.ai import whisper
from modules.ai import openai

def text_to_image(prompt, settings):
    if (settings.get("use_sdxl", False) == True):
        return sdlx_turbo.text_to_image_sdxl(prompt)
    if (settings.get("use_sd15", False) == True):
        return sd15.text_to_image_sd15(prompt)
    elif (settings.get("use_sd", True) == True):
        return sd_turbo.text_to_image_sd(prompt)
    elif (settings.get("use_sdc", True) == True):
        return sdc.text_to_image_sdc(prompt)
    else:
        return openai.text_to_image_openai(prompt, settings)

def image_to_depth(image):
    return dpt.image_to_depth_dpt(image)

def detect_image(image):
    return detr.detect_image_detr(image)

def variation_image(file, settings):
    if (settings.get("use_sd_variation", True) == True):
        return sd_variations.image_variation_sd(file)
    else:
        return openai.image_variation_openai(file, settings)
    
def sketch_image(image, prompt, negative_prompt = ''):
    return sketch_sdxl.image_sketch_sdxl(image, prompt, negative_prompt)

def music_generate(prompt, settings = {}):
    return musicgen.generate_music(prompt)

def text_to_speech(prompt, settings = {}):
    if (settings.get("use_speecht5", True) == True):
        return speecht5_tts.speak(prompt)
    else:
        return openai.speak(prompt)
    
def ask_llm(system, user, settings):
    if (settings.get("use_tinyllama", True) == True):
        return tinyllama.ask_tinyllama(system, user)
    else:
        return openai.ask(system, user, settings)
    
def ask_llm_json(system, user, settings):
    return openai.ask_json(system, user, settings)
    
def get_vision(image, prompt, settings = {}):
    if (settings.get("use_moondream", True) == True):
        return moondream1.vision_moondream1(image, prompt)
    else:
        return openai.vision_openai(image, prompt, settings)

def speech_to_text(audio, settings = {}):
    return whisper.speech_to_text_whisper(audio)
    