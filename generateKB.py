import whisper
from utils import get_audio_files, save_transcripsion
from config import WHISPER_MODEL_NAME

model = whisper.load_model(WHISPER_MODEL_NAME)


def transcribe_audios(audio_files):
    # decode the audio
    options = whisper.DecodingOptions(fp16=False)

    for audio_file in audio_files:
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(audio_file)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # detect the spoken language
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        result = whisper.decode(model, mel, options)

        # print the recognized text
        print(result.text)

        # get the audio file name from path and save the transcription
        file_name = audio_file.name.split('.')[0] + '.txt'
        save_transcripsion(file_name, result.text)


def main():
    audio_files = get_audio_files()
    print("audio_files", audio_files)
    transcribe_audios(audio_files)


if __name__ == "__main__":
    main()
