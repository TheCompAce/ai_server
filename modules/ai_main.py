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
from modules.ai import nomic
from modules.ai import openai
from modules.ai import proteus
from modules.ai import audioldm
from modules.cache import Cache

def check_cache(settings, key):
    cache = Cache()
    if (settings.get("use_cache", False) == True):
        return cache.get(key)
    else:
        return None

def set_cache(settings, key, value, is_value_binary=False):
    cache = Cache()
    if (settings.get("use_cache", False) == True):
        cache.set(key, value, is_value_binary)

def text_to_image(prompt, settings):
    cache_val = check_cache(settings, ("tti", prompt))
    if (cache_val!= None):
        return cache_val
    
    if (settings.get("use_sdc", False) == True):
        reval = sdc.text_to_image_sdc(prompt)    
    if (settings.get("use_sd15", False) == True):
        reval = sd15.text_to_image_sd15(prompt)
    elif (settings.get("use_sd", False) == True):
        reval = sd_turbo.text_to_image_sd(prompt)
    elif (settings.get("use_proteus", False) == True):
        reval = proteus.text_to_image_proteus(prompt)
    elif (settings.get("use_sdxl", True) == True):
        reval = sdlx_turbo.text_to_image_sdxl(prompt)
    else:
        reval = openai.text_to_image_openai(prompt, settings)
    
    set_cache(settings, ("tti", prompt), reval, True)

    return reval

def image_to_depth(image, settings = {}):
    cache_val = check_cache(settings, ("ttd", image))
    if (cache_val!= None):
        return cache_val
    
    reval = dpt.image_to_depth_dpt(image)

    set_cache(settings, ("ttd", image), reval, True)
    return reval

def detect_image(image, settings = {}):
    cache_val = check_cache(settings, ("di", image))
    if (cache_val!= None):
        return cache_val
    
    reval = detr.detect_image_detr(image)

    set_cache(settings, ("di", image), reval, True)    
    return reval

def variation_image(image, settings = {}):
    cache_val = check_cache(settings, ("vi", image))
    if (cache_val!= None):
        return cache_val
    
    if (settings.get("use_sd_variation", True) == True):
        reval = sd_variations.image_variation_sd(image)
    else:
        reval = openai.image_variation_openai(image, settings)

    set_cache(settings, ("vi", image), reval, True)
    return reval
    
def sketch_image(image, prompt, negative_prompt = '', settings = {}):
    cache_val = check_cache(settings, ("ski", (image, prompt, negative_prompt)))
    if (cache_val!= None):
        return cache_val
    
    reval = sketch_sdxl.image_sketch_sdxl(image, prompt, negative_prompt)

    set_cache(settings, ("ski", (image, prompt, negative_prompt)), reval, True)
    return reval

def music_generate(prompt, settings = {}):
    cache_val = check_cache(settings, ("mg", prompt))
    if (cache_val!= None):
        return cache_val
    
    reval = musicgen.generate_music(prompt)

    set_cache(settings, ("mg", prompt), reval, True)
    return reval

def text_to_speech(prompt, settings = {}):
    cache_val = check_cache(settings, ("tts", prompt))
    if (cache_val!= None):
        return cache_val
    
    if (settings.get("use_speecht5", True) == True):
        reval = speecht5_tts.speak(prompt)
    else:
        reval = openai.speak(prompt)

    set_cache(settings, ("tts", prompt), reval, True)
    return reval
    
def ask_llm(system, user, settings):
    cache_val = check_cache(settings, ("llm", (system, user)))
    if (cache_val!= None):
        return cache_val
    
    if (settings.get("use_tinyllama", True) == True):
        reval = tinyllama.ask_tinyllama(system, user)
    else:
        reval = openai.ask(system, user, settings)
        
    set_cache(settings, ("llm", (system, user)), reval, True)
    return reval
    
def ask_llm_json(system, user, settings):
    cache_val = check_cache(settings, ("llm_json", (system, user)))
    if (cache_val!= None):
        return cache_val
    
    reval = openai.ask_json(system, user, settings)
        
    set_cache(settings, ("llm_json", (system, user)), reval, True)
    return reval
    
def get_vision(image, prompt, settings = {}):
    cache_val = check_cache(settings, ("vis", (image, prompt)))
    if (cache_val!= None):
        return cache_val
    
    if (settings.get("use_moondream", True) == True):
        reval = moondream1.vision_moondream1(image, prompt)
    else:
        reval = openai.vision_openai(image, prompt, settings)
        
    set_cache(settings, ("vis", (image, prompt)), reval, False)
    return reval

def speech_to_text(audio, settings = {}):
    cache_val = check_cache(settings, ("stt", audio))
    if (cache_val!= None):
        return cache_val
    
    reval = whisper.speech_to_text_whisper(audio)
        
    set_cache(settings, ("stt", audio), reval, False)
    return reval
    
def ask_llm_embed(prompt, settings = {}):
    cache_val = check_cache(settings, ("embed", prompt))
    if (cache_val!= None):
        return cache_val
    
    if (settings.get("use_nomic", True) == True):
        reval = nomic.get_embeddings(prompt)
    else:
        reval = openai.create_embeddings(prompt)
        
    set_cache(settings, ("embed", prompt), reval, True)
    return reval
    
def text_to_sound(prompt, settings = {}):
    cache_val = check_cache(settings, ("ttsd", prompt))
    if (cache_val!= None):
        return cache_val

    reval = audioldm.text_to_sound_audioldm(prompt)
        
    set_cache(settings, ("ttsd", prompt), reval, True)
    return reval
