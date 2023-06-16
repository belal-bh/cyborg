from pathlib import Path
from dotenv import dotenv_values

_config = {
    **dotenv_values(".env.default"),
    **dotenv_values(".env")
}

# BASE Directory
BASE_DIR = Path(__file__).resolve().parent

# Audio files directory
AUDIO_DIR = BASE_DIR.joinpath(_config["AUDIO_DIR"])

# Whisper transribed output directory
TRANSCRIBE_DIR = BASE_DIR.joinpath(_config['TRANSCRIBE_DIR'])

# Supported audio file extensions (TODO: More will be added later)
SUPPORTED_AUDIO_FILE_EXTENSIONS = ['.mp3', '.m4a', '.wav']

# Whisper Model to be used
WHISPER_MODEL_NAME = _config["WHISPER_MODEL_NAME"]
