@echo off
if not exist env (
    echo Creating virtual environment...
    python -m venv env
    echo.

    echo Activating virtual environment...
    call env\Scripts\activate

    echo Updating pip...
    python -m pip install --upgrade pip

    call env\Scripts\deactivate

    echo Activating virtual environment...
    call env\Scripts\activate

    echo Installing dependencies...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  | find /V "already satisfied"
    pip install ".[xformers]" --upgrade --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt | find /V "already satisfied"
    pip install git+https://github.com/kashif/diffusers.git@wuerstchen-v3
    pip install git+https://github.com/facebookresearch/audiocraft.git

    REM python -m spacy download en_core_web_sm
    REM pip install git+https://github.com/suno-ai/bark.git

    
)
call env\Scripts\activate
pip install -r requirements.txt | find /V "already satisfied"


echo Starting ai_server...

python server.py %*