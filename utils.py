from pathlib import Path
from config import AUDIO_DIR, SUPPORTED_AUDIO_FILE_EXTENSIONS, TRANSCRIBE_DIR


def get_audio_files(audio_dir=AUDIO_DIR):
    audio_files = []
    path = Path(audio_dir)
    for file in path.glob('**/*'):
        if file.is_file() and file.suffix.lower() in SUPPORTED_AUDIO_FILE_EXTENSIONS:
            audio_files.append(file)
    return audio_files


def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)


def save_transcripsion(file_name, content):
    file_path = TRANSCRIBE_DIR.joinpath(file_name)
    write_to_file(file_path, content)


print("audio_files paths:", [f.name for f in get_audio_files()])
