import os
import pinecone
from tqdm import tqdm
from dotenv import load_dotenv
import src.helper as helper
from langchain_pinecone import PineconeVectorStore

def store_vectors():
    """
    The function `store_vectors` extracts data from PDFs, creates text chunks, fetches an embedding
    model, sets up a connection to Pinecone Database, and stores the data in the database.
    """
    chunks = helper.get_chunks_from_pdf(path="data")
    embeddings = helper.get_embeddings()
    index = helper.pinecone_init()
    done = helper.store_data(text_chunks=chunks, embeddings=embeddings, index=index)
    if done == False:
        exit()

if __name__ == "__main__":
    store_vectors()
