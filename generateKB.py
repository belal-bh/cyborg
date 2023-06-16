import whisper
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from utils import (
    get_audio_files,
    save_transcripsion,
    load_documents
)
from config import (
    WHISPER_MODEL_NAME,
    DEVICE_TYPE,
    TRANSCRIBE_DIR,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL_NAME,
    KB_DIR,
    CHROMA_SETTINGS
)

model = whisper.load_model(WHISPER_MODEL_NAME)


def transcribe_audios(audio_files):
    # decode the audio
    options = whisper.DecodingOptions(fp16=(DEVICE_TYPE == "cuda"))

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

    SOURCE_DIRECTORIES = [TRANSCRIBE_DIR, DOCUMENTS_DIR]

    # Load documents and split in chunks
    print(f"Loading documents from {SOURCE_DIRECTORIES} folders")
    documents = load_documents(SOURCE_DIRECTORIES)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(
        f"Loaded {len(documents)} documents from  {SOURCE_DIRECTORIES} folders")
    print(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE_TYPE},
    )
    # change the embedding type here if you are running into issues.
    # These are much smaller embeddings and will work for most appications
    # If you use HuggingFaceEmbeddings, make sure to also use the same in the
    # other files

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=str(KB_DIR.absolute()),
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None


if __name__ == "__main__":
    main()
