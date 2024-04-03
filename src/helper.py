from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone.data.index import Index
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import pinecone
import os
from dotenv import load_dotenv


def load_pdfs(data: str) -> list:
    """
    The function `load_data` takes a string representing a directory path, loads all PDF files within
    that directory using a PyPDFLoader, and returns a list of loaded data.
    
    :param data: The `data` parameter is a string that represents the directory path where the PDF files
    are located
    :type data: str
    :return: A list of PDF files loaded using the DirectoryLoader with the specified parameters.
    """
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def text_split(extracted_data: list) -> list:
    """
    The function `text_split` takes a list of extracted data and splits the text into chunks of 500
    characters with a 20-character overlap using a RecursiveCharacterTextSplitter.
    
    :param extracted_data: The `extracted_data` parameter is expected to be a list of text documents
    that you want to split into smaller chunks for further processing or analysis. The `text_split`
    function takes this list of text documents as input and uses a `RecursiveCharacterTextSplitter`
    object to split each document into
    :type extracted_data: list
    :return: The function `text_split` is returning the result of splitting the extracted data into
    chunks using a `RecursiveCharacterTextSplitter` with a chunk size of 500 and an overlap of 20. The
    return value is a list of the split documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(extracted_data)


def download_hugging_face_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    This Python function downloads Hugging Face embeddings using a specified model name.
    
    :param model_name: The `model_name` parameter in the `download_hugging_face_embeddings` function is
    a string that specifies the name of the Hugging Face model to be downloaded. By default, it is set
    to "sentence-transformers/all-MiniLM-L6-v2". This parameter allows you to specify a, defaults to
    sentence-transformers/all-MiniLM-L6-v2
    :type model_name: str (optional)
    :return: An instance of the `HuggingFaceEmbeddings` class initialized with the specified
    `model_name`.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def pinecone_init():
    """
    The function `pinecone_init` initializes a connection to a Pinecone Database using environment
    variables for API key, environment, and index name.
    :return: The function `pinecone_init()` is returning the Pinecone index object after setting up the
    connection to the Pinecone Database and retrieving the current state of the Vector Database.
    """
    PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME = get_secrets()
    try:
        print("Setting up the connection to Pinecone Database...")
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    except Exception as e:
        print(f"Error setting up the connection to Pinecone Database: {e}")
        exit()
    index = pc.Index(PINECONE_INDEX_NAME)
    print("Current State of the Vector Database: ->")
    print(index.describe_index_stats())
    return index
    

def store_data(text_chunks: list, embeddings: HuggingFaceEmbeddings, index: Index, batch_size: int = 300) -> bool:
    PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME = get_secrets()
    print("Storing the data on the Pinecone Vector DataBase")
    total_documents = len(text_chunks)
    with tqdm(total=total_documents, desc="Storing documents") as pbar:
        for i in range(0, total_documents, batch_size):
            batch = text_chunks[i:i + batch_size]
            try:
                PineconeVectorStore.from_documents(batch, embeddings, index_name=PINECONE_INDEX_NAME)
                pbar.update(len(batch))
            except Exception as e:
                print(f"Error storing documents: {e}")
                return False
    print("Finished Uploading the data on the Pinecone Vector DataBase.")
    print("State of DataBase After Storing: -> ")
    print(index.describe_index_stats())
    return True


def get_chunks_from_pdf(path: str) -> list:
    try:
        print("Extracting data...")
        extracted_data = load_pdfs(data="data")
    except Exception as e:
        print(f"Error during loading the data: {e}")
        exit()
    print("Creating text chunks...")
    text_chunks = text_split(extracted_data=extracted_data)
    return text_chunks


def get_embeddings(emb_model: str = None) -> HuggingFaceEmbeddings:
    try:
        print("Getting the embedding model...")
        if emb_model is None:
            embeddings = download_hugging_face_embeddings()
        else:
            embeddings = download_hugging_face_embeddings(emb_model)
    except Exception as e:
        print(f"Error during fetching the embedding model: {e}")
        exit()
    return embeddings


def get_secrets() -> tuple:
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    return PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME


def load_vectorstore(prompt_template: str, embeddings: HuggingFaceEmbeddings) -> tuple:
    PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME = get_secrets()
    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["question", "context", "history"])
    return vectorstore, PROMPT


    