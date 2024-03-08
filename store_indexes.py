import os
import pinecone
from tqdm import tqdm
from dotenv import load_dotenv
from src.helper import load_pdfs, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore

def store_vectors():
    """
    The function `store_vectors` extracts data from PDFs, creates text chunks, fetches an embedding
    model, sets up a connection to Pinecone Database, and stores the data in the database.
    """
    
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    try:
        print("Extracting data...")
        extracted_data = load_pdfs(data="data")
    except Exception as e:
        print(f"Error during loading the data: {e}")
        exit()

    print("Creating text chunks...")
    text_chunks = text_split(extracted_data=extracted_data)

    try:
        print("Getting the embedding model...")
        # The line `embeddings = download_hugging_face_embeddings()` is calling a function named
        # `download_hugging_face_embeddings()` to fetch an embedding model from the Hugging Face model
        # hub. This function likely downloads a pre-trained language model or embedding model from
        # Hugging Face's repository, which will be used to convert the text data into numerical
        # vectors for storage in the Pinecone Database.
        embeddings = download_hugging_face_embeddings()
    except Exception as e:
        print(f"Error during fetching the embedding model: {e}")
        exit()

    try:
        print("Setting up the connection to Pinecone Database...")
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    except Exception as e:
        print(f"Error setting up the connection to Pinecone Database: {e}")
        exit()

    print ("State of DataBase Before Storing: -> ")
    index = pc.Index(PINECONE_INDEX_NAME)
    print (index.describe_index_stats())
    print ("==================================================")
    print("Storing the data on the Pinecone Database...")
    batch_size = 300
    total_documents = len(text_chunks)

    with tqdm(total=total_documents, desc="Storing documents") as pbar:
        for i in range(0, total_documents, batch_size):
            batch = text_chunks[i:i+batch_size]
            try:
                PineconeVectorStore.from_documents(batch, embeddings, index_name=PINECONE_INDEX_NAME)
                pbar.update(len(batch))
            except Exception as e:
                print(f"Error storing documents: {e}")
                exit()

    print("Done")
    print ("==================================================")
    print ("State of DataBase After Storing: -> ")
    print (index.describe_index_stats())

if __name__ == "__main__":
    store_vectors()
