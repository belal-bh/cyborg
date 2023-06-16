import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from langchain.docstore.document import Document

from config import (
    AUDIO_DIR,
    SUPPORTED_AUDIO_FILE_EXTENSIONS,
    TRANSCRIBE_DIR,
    SUPPORTED_DOCUMENT_MAP,
    KB_THREADS,
)


def get_audio_files(audio_dir=AUDIO_DIR):
    audio_files = get_supported_file_paths(
        audio_dir, SUPPORTED_AUDIO_FILE_EXTENSIONS)
    return audio_files


def get_supported_file_paths(files_dir, extensions):
    file_paths = []
    path = Path(files_dir)
    for file in path.glob('**/*'):
        if file.is_file() and file.suffix.lower() in extensions:
            file_paths.append(file)
    return file_paths


def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)


def save_transcripsion(file_name, content):
    file_path = TRANSCRIBE_DIR.joinpath(file_name)
    write_to_file(file_path, content)


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    loader_class = SUPPORTED_DOCUMENT_MAP.get(file_extension)
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]


def load_document_batch(filepaths):
    print("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name)
                   for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)


def load_documents(source_dirs):
    print('source_dirs', source_dirs)
    # Loads all documents from the source documents directories
    paths = []
    for source_dir in source_dirs:
        all_files = os.listdir(source_dir)
        for file_path in all_files:
            file_extension = os.path.splitext(file_path)[1]
            source_file_path = os.path.join(source_dir, file_path)
            if file_extension in SUPPORTED_DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    # Have at least one worker and at most KB_THREADS workers
    n_workers = min(KB_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i: (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            contents, _ = future.result()
            docs.extend(contents)

    return docs
