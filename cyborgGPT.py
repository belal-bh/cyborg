import click
from get_retrieval_qa import get_retrieval_qa

from config import DEVICE_TYPE


@click.command()
@click.option(
    "--device_type",
    default=DEVICE_TYPE or "cuda",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
def main(device_type, show_sources):
    """
    This function implements the information retrieval task.


    1. Loads an embedding model, can be HuggingFaceInstructEmbeddings or HuggingFaceEmbeddings
    2. Loads the existing vectorestore that was created by generateKB.py
    3. Loads the local LLM using load_model function - You can now set different LLMs.
    4. Setup the Question Answer retreival chain.
    5. Question answers.
    """

    print(f"Running on: {device_type}")
    print(f"Display Source Documents set to: {show_sources}")
    qa = get_retrieval_qa(device_type)

    # Interactive questions and answers
    print("----------------------------------Cyborg---------------------------")
    print("Starting QA Session with Cyborg...(Type 'exit' to end session)...")
    print("...................................................................")
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            print("...................................................................")
            print("Cyborg: Ending Session, Good Luck!...")
            print("----------------------------------Cyborg---------------------------")
            break
        # Get the answer from the chain
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if show_sources:  # this is a flag that you can set to disable showing answers.
            # # Print the relevant sources used for the answer
            print(
                "----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print(
                "----------------------------------SOURCE DOCUMENTS---------------------------")


if __name__ == "__main__":
    main()
