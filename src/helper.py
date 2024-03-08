from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import os
from dotenv import load_dotenv
load_dotenv()


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

def  download_hugging_face_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
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

# def pinecone_init():
#     pinecone_key = os.getenv("PINECONE_API_KEY")
#     pinecone_env = os.getenv("PINECONE_ENV")
#     pinecone_index = os.getenv("PINECONE_INDEX_NAME")
#     pc = pinecone.Pinecone(api_key=pinecone_key, environment=pinecone_env)
#     index=pc.Index(pinecone_index)
#     print("Stats of the Vector Database: ->")
#     print("================================================")
#     print(index.describe_index_stats())
    

# def push_data_on_pinecone(text_chunks: list, embeddings: HuggingFaceEmbeddings, ):
    