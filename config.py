import os
from pathlib import Path
from dotenv import dotenv_values
from chromadb.config import Settings
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader, UnstructuredExcelLoader

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

# Additional documents for knowledge base
DOCUMENTS_DIR = BASE_DIR.joinpath(_config["DOCUMENTS_DIR"])

# Supported audio file extensions (TODO: More will be added later)
SUPPORTED_AUDIO_FILE_EXTENSIONS = ['.mp3', '.m4a', '.wav']

# Whisper Model to be used
WHISPER_MODEL_NAME = _config["WHISPER_MODEL_NAME"]

# Device type i.e 'cuda' or 'cpu'
DEVICE_TYPE = _config["DEVICE_TYPE"]

# documents reader currently supported document types with reader
SUPPORTED_DOCUMENT_MAP = {
    ".txt": TextLoader,
    '.pdf': PDFMinerLoader,
    '.csv': CSVLoader,
    '.xls': UnstructuredExcelLoader,
    '.xlxs': UnstructuredExcelLoader
}

# Default Instructor Model
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"

# Persisted Knoledge Bage Directory
KB_DIR = BASE_DIR.joinpath("DB")

# KB Threads
KB_THREADS = os.cpu_count() or 8

# Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=str(KB_DIR),
    anonymized_telemetry=False
)

# default model_id and Basename (See details at load_model.py)
LLM_MODEL_ID = "TheBloke/WizardLM-7B-uncensored-GPTQ"
LLM_MODEL_BASENAME = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
